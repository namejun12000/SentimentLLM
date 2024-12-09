import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue


def build_index(dataset_name):

    ui_mat = np.loadtxt('data/%s.txt' % dataset_name, dtype=np.int32)

    n_users = ui_mat[:, 0].max()
    n_items = ui_mat[:, 1].max()

    u2i_index = [[] for _ in range(n_users + 1)]
    i2u_index = [[] for _ in range(n_items + 1)]

    for ui_pair in ui_mat:
        u2i_index[ui_pair[0]].append(ui_pair[1])
        i2u_index[ui_pair[1]].append(ui_pair[0])

    return u2i_index, i2u_index

# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample(uid):

        # uid = np.random.randint(1, usernum + 1)
        while len(user_train[uid]) <= 1: uid = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[uid][-1]
        idx = maxlen - 1

        ts = set(user_train[uid])
        for i in reversed(user_train[uid][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break

        return (uid, seq, pos, neg)

    np.random.seed(SEED)
    uids = np.arange(1, usernum+1, dtype=np.int32)
    counter = 0
    while True:
        if counter % usernum == 0:
            np.random.shuffle(uids)
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample(uids[counter % usernum]))
            counter += 1
        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


# train/val/test data generation
def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]

# TODO: merge evaluate functions for test and val set
# evaluate on test set
def evaluate(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0
    
    # # 모든 사용자 평가 (샘플링 제거)
    # users = range(1, usernum + 1)

    # # 사용자 80%
    # users = range(4474, 22364)

    # 사용자 50% (15723)
    users = range(11183, 22364)

    # # 사용자 90% (20107)
    # users = range(2237,22364)

    # candidates
    candidates = {}
    candidates_valid = {}
    candidates_test_valid = {}

    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]] # 긍정 샘플

        for i in range(1, itemnum + 1):
            if i not in rated and i != test[u][0]:
                item_idx.append(i)  # 부정 샘플을 전체 아이템에서 추가

        # # 100개의 부정 샘플
        # for _ in range(100):
        #     t = np.random.randint(1, itemnum + 1)
        #     while t in rated: t = np.random.randint(1, itemnum + 1)
        #     item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0] # - for 1st argsort DESC

        predictions = predictions.cpu().detach().numpy() # 추가됨

        rank = predictions.argsort().argsort()[0].item()
        
        # top 50 candidates 설정
        # 첫 번째 인덱스에 test item을 먼저 추가
        unique_top50_items = [test[u][0]]

        # 나머지 상위 49개의 아이템 추가 (중복 피하기)
        for item in np.argsort(predictions):
            if (item_idx[item] not in unique_top50_items and item_idx[item] not in rated and item_idx[item] != valid[u][0]):
                unique_top50_items.append(item_idx[item])
            if len(unique_top50_items) == 50:  # test item 포함 총 50개
                break

        # 사용자별 후보군 저장
        candidates[u] = unique_top50_items

        # valid candiates
        valid_top50_items = [valid[u][0]]
        
        for item in np.argsort(predictions):
            if (item_idx[item] not in valid_top50_items and item_idx[item] not in rated and item_idx[item] != test[u][0]):  # 테스트 아이템 제외
                valid_top50_items.append(item_idx[item])
            if len(valid_top50_items) == 50:
                break
        candidates_valid[u] = valid_top50_items


        # test + valid candidates
        test_valid_top50_items = [test[u][0], valid[u][0]]
        for item in np.argsort(predictions):
            if item_idx[item] not in test_valid_top50_items and item_idx[item] not in rated:
                test_valid_top50_items.append(item_idx[item])
            if len(test_valid_top50_items) == 50:
                break
        candidates_test_valid[u] = test_valid_top50_items

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user, candidates, candidates_valid, candidates_test_valid


# evaluate on val set
def evaluate_valid(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    
    # # 모든 사용자 평가 (샘플링 제거)
    # users = range(1, usernum + 1)

    # # 사용자 80%
    # users = range(4474, 22364)

    # 사용자 50% (15723)
    users = range(11183, 22364)

    # # 사용자 90% (20107)
    # users = range(2237,22364)

    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]

        # for _ in range(100):
        #     t = np.random.randint(1, itemnum + 1)
        #     while t in rated: t = np.random.randint(1, itemnum + 1)
        #     item_idx.append(t)

        for i in range(1, itemnum + 1):
            if i not in rated:
                item_idx.append(i)  # 부정 샘플을 전체 아이템에서 추가

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user
  