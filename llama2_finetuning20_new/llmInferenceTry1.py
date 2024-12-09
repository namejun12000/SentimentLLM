import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from peft import PeftModel
import re
from tqdm import tqdm  # 진행 상황을 표시하기 위한 라이브러리

# PyTorch 시드 고정
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    
# 원본 모델 로드
base_model = "meta-llama/Llama-2-7b-hf"  # 원본 모델 경로
model = AutoModelForCausalLM.from_pretrained(base_model, 
                                             device_map={"": 0}, 
                                             load_in_8bit=False)
tokenizer = AutoTokenizer.from_pretrained(base_model)

# LoRA 어댑터 모델 로드
adapter_path = "./fine_tuned_model_try1"  # 어댑터 모델 경로
model = PeftModel.from_pretrained(model, adapter_path)

# 모델을 GPU로 이동
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# # 데이터셋 로드 (나머지 80%)
# dataset_80 = load_dataset("namejun12000/AW_finetuning_5core_split1_all_final_valid_include", split="train_80")

# 50 % (test)
dataset_50 = load_dataset("namejun12000/AW_finetuning_5core_split1_all_final_valid_include_inference1", split="train_50_second")

# 50 % (valid)
datset_50_valid = load_dataset("namejun12000/AW_finetuning_5core_split1_all_final_valid_include_inference2", split="train_50_second")

# # 전체 데이터 중에서 0.05%만 샘플링 (test)
# sampled_dataset = dataset_50.shuffle(seed=24).select(range(30))
# sampled_dataset_valid = dataset_50_valid.shuffle(seed=25).select(range(100))

def dcg_at_k(scores, k=None):
    if k is not None:
        scores = scores[:k]
    discounts = np.log2(np.arange(2, scores.size + 2))
    return np.sum(scores / discounts)

def ndcg_at_k(top_k_recommended, test_item, k=10):
    # 추천 목록에서 test_item 위치를 찾고, 해당 위치에 점수를 부여
    relevance_scores = [1 if item == test_item else 0 for item in top_k_recommended]
    dcg = dcg_at_k(np.array(relevance_scores), k)
    ideal_relevance = sorted(relevance_scores, reverse=True)
    idcg = dcg_at_k(np.array(ideal_relevance), k)
    return dcg / idcg if idcg > 0 else 0
    
def calculate_hit_ratio(top_k_recommended, test_item):
    # test_item이 top_k_recommended에 있는지 확인
    return int(test_item in top_k_recommended)

# LOO 평가 함수 정의
def evaluate_loo(model, dataset, tokenizer, top_k=10):
    ndcg_total_test, hit_ratio_total_test = 0, 0

    NDCG = 0.0
    HT =0.0

    valid_users = 0
    
    for example in tqdm(dataset, desc="Evaluating LOO", unit="user"):
        user_id = example["input"]["user_id"]
        interaction_history = example["input"]["interaction"]
        candidates = example["input"]["candidates"]
        ground_truth = example["output"]["recommended"]
        sentiments = example['input']['sentiments']

        if len(interaction_history) < 5:
            continue

        # LOO 분할
        test_item = int(interaction_history[-1])
        valid_item = int(interaction_history[-2])
        train_items = interaction_history[:-1]
        sentiment_items = sentiments[:-1]

        # 입력 준비
        input_text = (
            f"[INST] Recommend ten beauty products from the candidate list based on user's interaction history and sentiment label from the reviews.\n"
            f"interaction history: {', '.join(map(str, train_items))}\n"
            f"sentiment label: {', '.join(map(str,sentiment_items))}\n"
            f"candidate list: {', '.join(map(str, candidates))} [/INST]\nRecommended Products: "
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        # 모델 예측
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=70, temperature=0.8, top_k=10, top_p=0.85)
            # outputs = model.generate(**inputs, max_new_tokens=70, temperature=0.01, top_k=1, repetition_penalty=1.2)  # Greedy Decoding 사용
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # print(f"Generated text: {generated_text}")  # 디버깅 출력

            recommended_text = re.search(r"Recommended Products:\s*(.*)", generated_text)
            if recommended_text:
                # 중복된 항목을 제거하고 순서를 유지하여 추천 항목 목록 생성
                recommended_items = []
                for item in re.findall(r"\b\d+\b", recommended_text.group(1)):
                    item = int(item)
                    if item not in recommended_items:  # 중복 체크
                        recommended_items.append(item)
            else:
                recommended_items = []

        # # 디버깅: 추천 결과 출력
        # print(f"User ID: {user_id}")
        # print(f"Test item: {test_item}")
        # print(f"Valid item: {valid_item}")
        # print(f"Recommended items: {recommended_items[:top_k]}")
        # print(f"Ground truth: {ground_truth}")
        # print(f"Train item: {train_items}")

        # 상위 10개 아이템 선택
        top_k_recommended = recommended_items[:top_k]
        
        # 테스트 NDCG 및 Hit Ratio 계산
        ndcg_score_test = ndcg_at_k(top_k_recommended, test_item)
        hit_ratio_test = calculate_hit_ratio(top_k_recommended, test_item)
        ndcg_total_test += ndcg_score_test
        hit_ratio_total_test += hit_ratio_test

        ##### SASRec 평가방식
        # Test Item에 대한 NDCG 및 Hit Ratio 계산
        rank_test = top_k_recommended.index(test_item) if test_item in top_k_recommended else top_k
        if rank_test < top_k:
            NDCG += 1 / np.log2(rank_test + 2)
            HT += 1

        valid_users += 1

    # 평균 NDCG@10과 Hit Ratio@10
    avg_ndcg_test = ndcg_total_test / valid_users if valid_users > 0 else 0
    avg_hit_ratio_test = hit_ratio_total_test / valid_users if valid_users > 0 else 0

    return avg_ndcg_test, avg_hit_ratio_test, NDCG / valid_users, HT / valid_users

# LOO 평가 함수 정의 (valid)
def evaluate_loo2(model, dataset, tokenizer, top_k=10):
    ndcg_total_valid, hit_ratio_total_valid = 0, 0

    NDCG_v = 0.0
    HT_v = 0.0

    valid_users = 0
    
    for example in tqdm(dataset, desc="Evaluating LOO", unit="user"):
        user_id = example["input"]["user_id"]
        interaction_history = example["input"]["interaction"]
        candidates = example["input"]["candidates"]
        ground_truth = example["output"]["recommended"]
        sentiments = example['input']['sentiments']

        if len(interaction_history) < 5:
            continue

        # LOO 분할
        test_item = int(interaction_history[-1])
        valid_item = int(interaction_history[-2])
        train_items = interaction_history[:-2]
        sentiment_items = sentiments[:-2]

        # 입력 준비
        input_text = (
            f"[INST] Recommend ten beauty products from the candidate list based on user's interaction history and sentiment label from the reviews.\n"
            f"interaction history: {', '.join(map(str, train_items))}\n"
            f"sentiment label: {', '.join(map(str,sentiment_items))}\n"
            f"candidate list: {', '.join(map(str, candidates))} [/INST]\nRecommended Products: "
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        # 모델 예측
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=70, temperature=0.8, top_k=10, top_p=0.85)
            # outputs = model.generate(**inputs, max_new_tokens=70, temperature=0.01, top_k=1, repetition_penalty=1.2)  # Greedy Decoding 사용
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # print(f"Generated text: {generated_text}")  # 디버깅 출력

            recommended_text = re.search(r"Recommended Products:\s*(.*)", generated_text)
            if recommended_text:
                # 중복된 항목을 제거하고 순서를 유지하여 추천 항목 목록 생성
                recommended_items = []
                for item in re.findall(r"\b\d+\b", recommended_text.group(1)):
                    item = int(item)
                    if item not in recommended_items:  # 중복 체크
                        recommended_items.append(item)
            else:
                recommended_items = []

        # # 디버깅: 추천 결과 출력
        # print(f"User ID: {user_id}")
        # print(f"Test item: {test_item}")
        # print(f"Valid item: {valid_item}")
        # print(f"Recommended items: {recommended_items[:top_k]}")
        # print(f"Ground truth: {ground_truth}")
        # print(f"Train item: {train_items}")

        # 상위 10개 아이템 선택
        top_k_recommended = recommended_items[:top_k]

        # 검증 NDCG 및 Hit Ratio 계산
        ndcg_score_valid = ndcg_at_k(top_k_recommended, valid_item)
        hit_ratio_valid = calculate_hit_ratio(top_k_recommended, valid_item)
        ndcg_total_valid += ndcg_score_valid
        hit_ratio_total_valid += hit_ratio_valid

        ##### SASRec 평가방식
        # Valid Item에 대한 NDCG 및 Hit Ratio 계산
        rank_valid = top_k_recommended.index(valid_item) if valid_item in top_k_recommended else top_k
        if rank_valid < top_k:
            NDCG_v += 1 / np.log2(rank_valid + 2)
            HT_v += 1


        valid_users += 1

    # 평균 NDCG@10과 Hit Ratio@10
    avg_ndcg_valid = ndcg_total_valid / valid_users if valid_users > 0 else 0
    avg_hit_ratio_valid = hit_ratio_total_valid / valid_users if valid_users > 0 else 0

    return avg_ndcg_valid, avg_hit_ratio_valid, NDCG_v / valid_users, HT_v / valid_users

# 평가 실행
ndcg_test, hit_ratio_test, ndcg_test_sas, hit_ratio_test_sas  = evaluate_loo(model, dataset_50, tokenizer)
# ndcg_valid, hit_ratio_valid, ndcg_valid_sas, hit_ratio_valid_sas = evaluate_loo2(model, dataset_50_valid, tokenizer)

print(f"Test NDCG@10: {ndcg_test:.4f}, Test Hit Ratio@10: {hit_ratio_test:.4f}")
# print(f"Validation NDCG@10: {ndcg_valid:.4f}, Validation Hit Ratio@10: {hit_ratio_valid:.4f}")

print(f"\n\nevalution from SASRec\n")
print(f"Test NDCG@10: {ndcg_test_sas:.4f}, Test Hit ratio@10: {hit_ratio_test_sas:.4f}")
# print(f"Validation NDCG@10: {ndcg_valid_sas:.4f}, Validation Hit Ratio@10: {hit_ratio_valid_sas:.4f}")

# 결과를 텍스트 파일로 저장
with open("inference_results/evaluation_test_results_50.txt", "w") as f:
    f.write(f"Test NDCG@10: {ndcg_test:.4f}\n")
    f.write(f"Test Hit Ratio@10: {hit_ratio_test:.4f}\n")
    f.write(f"Test NDCG@10 (sasrec): {ndcg_test_sas:.4f}\n")
    f.write(f"Test Hit Ratio@10 (sasrec): {hit_ratio_test_sas:.4f}\n")

print("Evaluation results saved to 'evaluation_results_50.txt'")

# # 결과를 텍스트 파일로 저장
# with open("inference_results/evaluation_valid_results_50.txt", "w") as f:
#     f.write(f"Validation NDCG@10: {ndcg_valid:.4f}\n")
#     f.write(f"Validation Hit Ratio@10: {hit_ratio_valid:.4f}\n")
#     f.write(f"Validation NDCG@10 (sasrec): {ndcg_valid_sas:.4f}\n")
#     f.write(f"Validation Hit Ratio@10: {hit_ratio_valid_sas:.4f}\n")

# print("Evaluation results saved to 'evaluation_results_50.txt'")