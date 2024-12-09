import os
import time
import torch
import numpy as np
from model import SASRec
from utils import build_index, data_partition, WarpSampler, evaluate, evaluate_valid
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='beauty_review', help="Dataset name (default: beauty_review)")
    parser.add_argument('--train_dir', default='results', help="Directory to save model and results (default: results)")
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--maxlen', default=50, type=int)
    parser.add_argument('--hidden_units', default=50, type=int)
    parser.add_argument('--num_blocks', default=2, type=int)
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--dropout_rate', default=0.5, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    # parser.add_argument('--device', default='mps',type=str)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help="Device to use (default: cuda if available)")
    parser.add_argument('--inference_only', default=False, type=bool, help="Set to True for inference only mode")
    parser.add_argument('--state_dict_path', default=None, help="Path to saved model state dict for inference")

    return parser.parse_args()

def train_and_infer(args):
    # 데이터셋 준비
    u2i_index, i2u_index = build_index(args.dataset)
    dataset = data_partition(args.dataset)
    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    num_batch = (len(user_train) - 1) // args.batch_size + 1
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))

    f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')
    f.write('epoch (val_ndcg, val_hr) (test_ndcg, test_hr)\n')

    # 모델 준비
    sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
    model = SASRec(usernum, itemnum, args).to(args.device)
    
    # 모델 가중치 초기화
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass

    model.pos_emb.weight.data[0, :] = 0
    model.item_emb.weight.data[0, :] = 0

    model.train()

    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
        except: # in case your pytorch version is not 1.6 etc., pls debug by pdb if load weights failed
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            print('pdb enabled for your quick check, pls type exit() if you do not need it')
            import pdb; pdb.set_trace()

    # 추론만
    if args.inference_only:
        model.eval()
        # 평가 및 후보군 추출
        ndcg, hr, candidates = evaluate(model, dataset, args)
        
        print('test (NDCG@10: %.4f, HR@10: %.4f)' % (ndcg, hr))
        
        # 상위 50개 아이템을 파일로 저장
        with open(os.path.join(args.dataset + '_' + args.train_dir, 'candidates_50_per_user.txt'), 'w') as f:
            for user_id, top50_items in candidates.items():
                item_list = ' '.join(map(str, top50_items))
                f.write(f"{user_id} {item_list}\n")
        
        print("Top 50 candidates for each user have been saved.")

    # 훈론 & 추론
    else:
        # 훈련 과정
        adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
        bce_criterion = torch.nn.BCEWithLogitsLoss()

        best_val_ndcg, best_val_hr = 0.0, 0.0
        best_test_ndcg, best_test_hr = 0.0, 0.0

        T = 0.0
        t0 = time.time()
        
        # 최고 성능 모델 파일 경로 저장 변수
        best_model_path = None  

        for epoch in range(epoch_start_idx, args.num_epochs + 1):
            for step in range(num_batch):
                u, seq, pos, neg = sampler.next_batch()
                u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
                pos_logits, neg_logits = model(u, seq, pos, neg)
                pos_labels = torch.ones(pos_logits.shape, device=args.device)
                neg_labels = torch.zeros(neg_logits.shape, device=args.device)
                adam_optimizer.zero_grad()
                indices = np.where(pos != 0)
                loss = bce_criterion(pos_logits[indices], pos_labels[indices]) 
                loss += bce_criterion(neg_logits[indices], neg_labels[indices])
                for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
                loss.backward()
                adam_optimizer.step()
                print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item())) # expected 0.4~0.6 after init few epochs
                
            if epoch % 20 == 0:
                model.eval()
                t1 = time.time() - t0
                T += t1
                print('Evaluating', end='')
                t_test = evaluate(model, dataset, args)
                t_valid = evaluate_valid(model, dataset, args)
                print('epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)'
                        % (epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1]))

                if t_valid[0] > best_val_ndcg or t_valid[1] > best_val_hr or t_test[0] > best_test_ndcg or t_test[1] > best_test_hr:
                    best_val_ndcg = max(t_valid[0], best_val_ndcg)
                    best_val_hr = max(t_valid[1], best_val_hr)
                    best_test_ndcg = max(t_test[0], best_test_ndcg)
                    best_test_hr = max(t_test[1], best_test_hr)
                    folder = args.dataset + '_' + args.train_dir
                    fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
                    fname = fname.format(epoch, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
                    best_model_path = os.path.join(folder, fname)  # 최고 성능 모델 경로 업데이트
                    torch.save(model.state_dict(), best_model_path)

                f.write(str(epoch) + ' ' + str(t_valid) + ' ' + str(t_test) + '\n')
                f.flush()
                t0 = time.time()
                model.train()
        
            if epoch == args.num_epochs:
                folder = args.dataset + '_' + args.train_dir
                fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
                fname = fname.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
                torch.save(model.state_dict(), os.path.join(folder, fname))
        
        f.close()
        sampler.close()
        print("Training completed!")
        
        # 훈련 완료 후, best 모델로 추론 수행
        if best_model_path:
            model.load_state_dict(torch.load(best_model_path, map_location=torch.device(args.device), weights_only =True))
            model.eval()

            # 평가 및 후보군 추출
            ndcg, hr, candidates, candidates_valid, candidates_test_valid = evaluate(model, dataset, args)
            print('test (NDCG@10: %.4f, HR@10: %.4f)' % (ndcg, hr))

            ndcg_valid, hr_valid = evaluate_valid(model, dataset, args)
            print('valid (NDCG@10: %.4f, HR@10: %.4f)' % (ndcg_valid, hr_valid))

            # 상위 50개 아이템을 파일로 저장
            with open(os.path.join(args.dataset + '_' + args.train_dir, 'candidates_50_per_user.txt'), 'w') as f:
                for user_id, top50_items in candidates.items():
                    item_list = ' '.join(map(str, top50_items))
                    f.write(f"{user_id} {item_list}\n")
            
            # 상위 50개 valid 아이템 파일로 저장
            with open(os.path.join(args.dataset + '_' + args.train_dir, 'candidates_50_per_user_valid.txt'), 'w') as f:
                for user_id, top50_items in candidates_valid.items():
                    item_list = ' '.join(map(str, top50_items))
                    f.write(f"{user_id} {item_list}\n")            

            # 상위 50개 test + valid 아이템 파일로 저장
            with open(os.path.join(args.dataset + '_' + args.train_dir, 'candidates_50_per_user_test_valid.txt'), 'w') as f:
                for user_id, top50_items in candidates_test_valid.items():
                    item_list = ' '.join(map(str, top50_items))
                    f.write(f"{user_id} {item_list}\n")            
            
            print("Top 50 candidates for each user have been saved.")

            # 후보군을 변수에 저장 (파일 저장 대신)
            top50_candidates = {}

            for user_id, top50_items in candidates.items():
                top50_candidates[user_id] = top50_items
            print("Top 50 candidates saved in memory.")

            # 후보군을 변수에 저장 (파일 저장 대신) valid
            top50_candidates_valid = {}

            for user_id, top50_items in candidates_valid.items():
                top50_candidates_valid[user_id] = top50_items
            print("Top 50 candidates valid saved in memory.")

            # 후보군을 변수에 저장 (파일 저장 대신) valid + test
            top50_candidates_test_valid = {}

            for user_id, top50_items in candidates_test_valid.items():
                top50_candidates_test_valid[user_id] = top50_items
            print("Top 50 candidates valid saved in memory.")

    return top50_candidates, top50_candidates_valid, top50_candidates_test_valid
