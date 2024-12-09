import json
import huggingface_hub
import random
import numpy as np
from datasets import load_dataset
from datasets import DatasetDict, concatenate_datasets

huggingface_hub.login()

# 시드 고정
random.seed(42)
np.random.seed(42)

# user_item_sequences.txt
user_item_seq = {}

with open("data/user_item_sequences.txt", "r") as f:
  user_item_lines = f.readlines()

for line in user_item_lines:
  user_id, *product_ids = line.strip().split()
  user_item_seq[user_id] = product_ids

# candidates_50_per_user.txt
candidates_seq = {}

with open("beauty_review_results/candidates_50_per_user.txt", "r") as f:
  candidates_lines = f.readlines()

for line in candidates_lines:
  user_id, *candidates_ids = line.strip().split()
  candidates_seq[user_id] = candidates_ids
  random.shuffle(candidates_seq[user_id])

# candidates_50_per_user_valid.txt
valid_candidates_seq = {}

with open("beauty_review_results/candidates_50_per_user_valid.txt", "r") as f:
  valid_candidates_lines = f.readlines()

for line in valid_candidates_lines:
  user_id, *candidates_ids = line.strip().split()
  valid_candidates_seq[user_id] = candidates_ids
  random.shuffle(valid_candidates_seq[user_id])

# candidates_50_per_user_test_valid.txt
test_valid_candidates_seq = {}

with open("beauty_review_results/candidates_50_per_user_test_valid.txt", "r") as f:
  test_valid_candidates_lines = f.readlines()

for line in test_valid_candidates_lines:
  user_id, *candidates_ids = line.strip().split()
  test_valid_candidates_seq[user_id] = candidates_ids
  random.shuffle(test_valid_candidates_seq[user_id])

sentiment_seq = {}

with open('goeRobert/sentiment_strength_results.txt', 'r') as f:
  sentiment_lines = f.readlines()

for line in sentiment_lines:
  uid, pid, emotionLabel, emtionStrength, strengthLabel = line.strip().split(maxsplit=4)
  
  # 사용자별 감성 시퀀스 생성
  if uid not in sentiment_seq:
    sentiment_seq[uid] = []
  
  # 시퀀스에 감성 레이블 추가
  sentiment_seq[uid].append(strengthLabel)

# 상위 10개 후보군 선택 함수
def select_top_10_candidates(user_id):
    if user_id not in user_item_seq or user_id not in candidates_seq:
        return None

    # 사용자의 마지막 상호작용 아이템을 정답 레이블로 사용
    true_label = user_item_seq[user_id][-1]
    candidates = candidates_seq[user_id]

    # 정답 레이블이 후보군에 포함되어 있는지 확인
    if true_label in candidates:
        # 정답 레이블을 포함하여 상위 10개 선택
        top_candidates = [true_label] + [c for c in candidates if c != true_label][:9]
    else:
        # 정답 레이블이 없을 경우 단순 상위 10개 선택
        top_candidates = candidates[:10]
    
    return top_candidates

# 상위 10개 후보군 선택 함수 (valid)
def select_top_10_candidates_valid(user_id):
    if user_id not in user_item_seq or user_id not in valid_candidates_seq:
        return None

    # 사용자의 마지막 상호작용 아이템을 정답 레이블로 사용
    true_label = user_item_seq[user_id][-2]
    candidates = candidates_seq[user_id]

    # 정답 레이블이 후보군에 포함되어 있는지 확인
    if true_label in candidates:
        # 정답 레이블을 포함하여 상위 10개 선택
        top_candidates = [true_label] + [c for c in candidates if c != true_label][:9]
    else:
        # 정답 레이블이 없을 경우 단순 상위 10개 선택
        top_candidates = candidates[:10]
    
    return top_candidates

# 상위 10개 후보군 선택 함수 (test + valid)
def select_top_10_candidates_test_valid(user_id):
    if user_id not in user_item_seq or user_id not in test_valid_candidates_seq:
        return None

    # 사용자의 마지막 상호작용 아이템을 test item으로, 그 전 아이템을 valid item으로 사용
    test_item = user_item_seq[user_id][-1]
    valid_item = user_item_seq[user_id][-2]
    candidates = test_valid_candidates_seq[user_id]

    # test item과 valid item을 포함하여 상위 10개 선택
    # test item, valid item을 고정한 뒤 나머지 상위 8개의 후보군을 추가
    remaining_candidate = [c for c in candidates if c != test_item and c != valid_item]
    random.shuffle(remaining_candidate) # 상위 8개 섞기
    top_candidates = [test_item, valid_item] + remaining_candidate[:8]
    
    return top_candidates

# 각 사용자에 대해 상위 10개 후보군을 선택하고 결과 출력
top_10_candidates_per_user = {}
for user_id in user_item_seq:
    top_10_candidates = select_top_10_candidates(user_id)
    if top_10_candidates:
        top_10_candidates_per_user[user_id] = top_10_candidates

print("상위 10개 후보군이 각 사용자에 대해 설정되었습니다.") # top 10 from candiates

# 각 사용자에 대해 상위 10개 후보군을 선택하고 결과 출력 (valid)
top_10_candidates_per_user_valid = {}
for user_id in user_item_seq:
    top_10_candidates = select_top_10_candidates_valid(user_id)
    if top_10_candidates:
        top_10_candidates_per_user_valid[user_id] = top_10_candidates

print("상위 10개 후보군이 각 사용자에 대해 설정되었습니다.") # top 10 from candiates

# 각 사용자에 대해 상위 10개 후보군을 선택하고 결과 출력 (test +valid)
top_10_candidates_per_user_test_valid = {}
for user_id in user_item_seq:
    top_10_candidates = select_top_10_candidates_test_valid(user_id)
    if top_10_candidates:
        top_10_candidates_per_user_test_valid[user_id] = top_10_candidates

print("상위 10개 후보군이 각 사용자에 대해 설정되었습니다. (test+valid)") # top 10 from candiates

# 데이터 구조 초기화
training_data4 = {"data": []}

# create json
for uid in user_item_seq.keys():
  if uid in sentiment_seq and uid in test_valid_candidates_seq:
    training_data4['data'].append({
        "instruction": "Recommend ten beauty products from the candidate list based on user's interaction history and sentiment label from the reviews.",
        "input": {
          "user_id": uid,
          "interaction": user_item_seq[uid],
          "sentiments": sentiment_seq[uid],
          "candidates": test_valid_candidates_seq[uid]
        },
        "output": {
          "recommended": top_10_candidates_per_user_test_valid[uid]
        }
    })

# json 저장
with open("fineTuningDT/fine_tuning_data.json", "w", encoding='utf-8') as json_file:
  json.dump(training_data4, json_file, ensure_ascii=False, indent=4)

print("JSON 파일이 성공적으로 생성되었습니다. (try 4)\n")

data4 = load_dataset("json", data_files="fineTuningDT/fine_tuning_data.json", field="data")
data4.push_to_hub("namejun12000/AW_finetuning_5core_try1_all_final_valid_include")

# 데이터 구조 초기화 (inf1 test)
inference_data = {"data": []}

# create json
for uid in user_item_seq.keys():
  if uid in sentiment_seq and uid in candidates_seq:
    inference_data['data'].append({
        "instruction": "Recommend ten beauty products from the candidate list based on user's interaction history and sentiment label from the reviews.",
        "input": {
          "user_id": uid,
          "interaction": user_item_seq[uid],
          "sentiments": sentiment_seq[uid],
          "candidates": candidates_seq[uid]
        },
        "output": {
          "recommended": top_10_candidates_per_user[uid]
        }
    })

# json 저장
with open("fineTuningDT/inference_data.json", "w", encoding='utf-8') as json_file:
  json.dump(inference_data, json_file, ensure_ascii=False, indent=4)

print("JSON 파일이 성공적으로 생성되었습니다. (inference [test])\n")

data_inf1 = load_dataset("json", data_files="fineTuningDT/inference_data.json", field="data")
data_inf1.push_to_hub("namejun12000/AW_finetuning_5core_try1_all_final_valid_include_inference1")

# 데이터 구조 초기화 (inf2 valid)
inference_data2 = {"data": []}

# create json
for uid in user_item_seq.keys():
  if uid in sentiment_seq and uid in valid_candidates_seq:
    inference_data2['data'].append({
        "instruction": "Recommend ten beauty products from the candidate list based on user's interaction history and sentiment label from the reviews.",
        "input": {
          "user_id": uid,
          "interaction": user_item_seq[uid],
          "sentiments": sentiment_seq[uid],
          "candidates": valid_candidates_seq[uid]
        },
        "output": {
          "recommended": top_10_candidates_per_user_valid[uid]
        }
    })

# json 저장
with open("fineTuningDT/inference_data2.json", "w", encoding='utf-8') as json_file:
  json.dump(inference_data2, json_file, ensure_ascii=False, indent=4)

print("JSON 파일이 성공적으로 생성되었습니다. (Inference [valid])\n")

data_inf2 = load_dataset("json", data_files="fineTuningDT/inference_data2.json", field="data")
data_inf2.push_to_hub("namejun12000/AW_finetuning_5core_try1_all_final_valid_include_inference2")

# 시드 고정
random.seed(42)
np.random.seed(42)

# Custom Dataset into hf
code_dataset4 = "namejun12000/AW_finetuning_5core_try1_all_final_valid_include"
code_dataset_inf1 = "namejun12000/AW_finetuning_5core_try1_all_final_valid_include_inference1"
code_dataset_inf2 = "namejun12000/AW_finetuning_5core_try1_all_final_valid_include_inference2"

# fine-tuning (20%)
dataset_20_try4 = load_dataset(code_dataset4, split="train[:20%]") # 20% 
dataset_80_try4 = load_dataset(code_dataset4, split="train[20%:]") # 80%

# # fine-tuning (50%)
# dataset_50_first_half = load_dataset(code_dataset4, split="train[:50%]")  # First 50%
# dataset_50_second_half = load_dataset(code_dataset4, split="train[50%:]")  # Second 50%

# # fine-tuing (10%)
# dataset_10_try7 = load_dataset(code_dataset4, split="train[:10%]") # 10%
# dataset_90_try7 = load_dataset(code_dataset4, split="train[10%:]") # 90%

# inference (test)
dataset_50_first_half_inf1 = load_dataset(code_dataset_inf1, split="train[:50%]")  # First 50%
dataset_50_second_half_inf1 = load_dataset(code_dataset_inf1, split="train[50%:]")  # Second 50%

# inference (valid)
dataset_50_first_half_inf2 = load_dataset(code_dataset_inf2, split="train[:50%]")  # First 50%
dataset_50_second_half_inf2 = load_dataset(code_dataset_inf2, split="train[50%:]")  # Second 50%

# train[:20%]와 train[20%:]로 분할한 데이터셋을 DatasetDict로 저장합니다.
dataset_dict4 = DatasetDict({
    "train_20": dataset_20_try4,
    "train_80": dataset_80_try4
})

# dataset_dict6 = DatasetDict({
#     "train_50_first": dataset_50_first_half,
#     "train_50_second": dataset_50_second_half
# })

# dataset_dict7 = DatasetDict({
#     "train_10": dataset_10_try7,
#     "train_90": dataset_90_try7
# })

# inference (datasetdict 저장)
dataset_dict_inf1 = DatasetDict({
    "train_50_first": dataset_50_first_half_inf1,
    "train_50_second": dataset_50_second_half_inf1
})

dataset_dict_inf2 = DatasetDict({
    "train_50_first": dataset_50_first_half_inf2,
    "train_50_second": dataset_50_second_half_inf2
})

# Hugging Face Hub에 업로드
dataset_dict4.push_to_hub("namejun12000/AW_finetuning_5core_split1_all_final_valid_include")
# dataset_dict6.push_to_hub("namejun12000/AW_finetuning_5core_split1_all_final_valid_include_50")
# dataset_dict7.push_to_hub("namejun12000/AW_finetuning_5core_split1_all_final_valid_include_10")
dataset_dict_inf1.push_to_hub("namejun12000/AW_finetuning_5core_split1_all_final_valid_include_inference1")
dataset_dict_inf2.push_to_hub("namejun12000/AW_finetuning_5core_split1_all_final_valid_include_inference2")

print("Upload jsons in HuggingFace are completed!\n")