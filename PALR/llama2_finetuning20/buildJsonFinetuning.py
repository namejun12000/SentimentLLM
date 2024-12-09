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

with open("../../data/user_item_sequences.txt", "r") as f:
  user_item_lines = f.readlines()

for line in user_item_lines:
  user_id, *product_ids = line.strip().split()
  user_item_seq[user_id] = product_ids

# user preference1.txt
user_preference_test = {}
with open("../llama2_preference/user_preferences.txt", "r") as f:
  user_pref_lines = f.readlines()

for line in user_pref_lines:
  user_id, preference = line.strip().split(maxsplit=1)
  user_preference_test[user_id] = preference

# user preference2.txt
user_preference_valid = {}
with open("../llama2_preference/user_preferences2.txt", "r") as f:
  user_pref_lines2 = f.readlines()

for line in user_pref_lines2:
  user_id, preference2 = line.strip().split(maxsplit=1)
  user_preference_valid[user_id] = preference2

# product.txt
product_mapping = {}

with open("../../data/product.txt", "r") as f:
    product_lines = f.readlines()

# Parse product ID to product name mapping
for line in product_lines:
    product_id, product_name = line.strip().split(maxsplit=1)
    product_mapping[product_id] = product_name

# Replace product IDs with product names
user_item_name_seq = {}

for user_id, product_ids in user_item_seq.items():
    # Map product IDs to product names
    product_names = [product_mapping.get(product_id, "Unknown") for product_id in product_ids]
    user_item_name_seq[user_id] = product_names

# candidates_50_per_user.txt
candidates_seq = {}

with open("../../beauty_review_results/candidates_50_per_user.txt", "r") as f:
  candidates_lines = f.readlines()

for line in candidates_lines:
  user_id, *candidates_ids = line.strip().split()
  candidates_seq[user_id] = candidates_ids
  random.shuffle(candidates_seq[user_id])

# candidates_50_per_user_valid.txt
valid_candidates_seq = {}

with open("../../beauty_review_results/candidates_50_per_user_valid.txt", "r") as f:
  valid_candidates_lines = f.readlines()

for line in valid_candidates_lines:
  user_id, *candidates_ids = line.strip().split()
  valid_candidates_seq[user_id] = candidates_ids
  random.shuffle(valid_candidates_seq[user_id])

# candidates_50_per_user_test_valid.txt
test_valid_candidates_seq = {}

with open("../../beauty_review_results/candidates_50_per_user_test_valid.txt", "r") as f:
  test_valid_candidates_lines = f.readlines()

for line in test_valid_candidates_lines:
  user_id, *candidates_ids = line.strip().split()
  test_valid_candidates_seq[user_id] = candidates_ids
  random.shuffle(test_valid_candidates_seq[user_id])

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

# Replace product IDs with product names (candidates)
test_candidates_mapping = {}
valid_candidates_mapping = {}
test_valid_candidates_mapping = {}

# test
for user_id, product_ids in candidates_seq.items():
    # Map product IDs to product names
    product_names = [product_mapping.get(product_id, "Unknown") for product_id in product_ids]
    test_candidates_mapping[user_id] = product_names

# valid
for user_id, product_ids in valid_candidates_seq.items():
    # Map product IDs to product names
    product_names = [product_mapping.get(product_id, "Unknown") for product_id in product_ids]
    valid_candidates_mapping[user_id] = product_names

# test + valid
for user_id, product_ids in test_valid_candidates_seq.items():
    # Map product IDs to product names
    product_names = [product_mapping.get(product_id, "Unknown") for product_id in product_ids]
    test_valid_candidates_mapping[user_id] = product_names

print("후보군 mapping complete!")

# Replace product IDs with product names (top 10)
test_top10_mapping = {}
valid_top10_mapping = {}
test_valid_top10_mapping = {}

# test
for user_id, product_ids in top_10_candidates_per_user.items():
    # Map product IDs to product names
    product_names = [product_mapping.get(product_id, "Unknown") for product_id in product_ids]
    test_top10_mapping[user_id] = product_names

# valid
for user_id, product_ids in top_10_candidates_per_user_valid.items():
    # Map product IDs to product names
    product_names = [product_mapping.get(product_id, "Unknown") for product_id in product_ids]
    valid_top10_mapping[user_id] = product_names

# test + valid
for user_id, product_ids in top_10_candidates_per_user_test_valid.items():
    # Map product IDs to product names
    product_names = [product_mapping.get(product_id, "Unknown") for product_id in product_ids]
    test_valid_top10_mapping[user_id] = product_names

print("top10 mapping complete!")

# 데이터 구조 초기화
training_data4 = {"data": []}

# create json
for uid in user_item_name_seq.keys():
  if uid in test_valid_candidates_mapping:
    training_data4['data'].append({
        "instruction": "Recommend 10 other items based on user's history from the candidate list.",
        "input": {
          "user_id": uid,
          "interaction": user_item_name_seq[uid],
          "candidates": test_valid_candidates_mapping[uid]
        },
        "output": {
          "recommended": test_valid_top10_mapping[uid]
        }
    })

# json 저장
with open("fineTuningDT/fine_tuning_data.json", "w", encoding='utf-8') as json_file:
  json.dump(training_data4, json_file, ensure_ascii=False, indent=4)

print("JSON 파일이 성공적으로 생성되었습니다. (try 4)\n")

data4 = load_dataset("json", data_files="fineTuningDT/fine_tuning_data.json", field="data")
data4.push_to_hub("namejun12000/PALR_finetuning20")

# 데이터 구조 초기화 (inf1 test)
inference_data = {"data": []}

# create json
for uid in user_item_name_seq.keys():
  if uid in test_candidates_mapping and uid in user_preference_test:
    inference_data['data'].append({
        "instruction": "Recommend 10 other items based on user's history from the candidate list.",
        "input": {
          "user_id": uid,
          "preference": user_preference_test[uid],
          "interaction": user_item_name_seq[uid],
          "candidates": test_candidates_mapping[uid]
        },
        "output": {
          "recommended": test_top10_mapping[uid]
        }
    })

# json 저장
with open("fineTuningDT/inference_data.json", "w", encoding='utf-8') as json_file:
  json.dump(inference_data, json_file, ensure_ascii=False, indent=4)

print("JSON 파일이 성공적으로 생성되었습니다. (inference [test])\n")

data_inf1 = load_dataset("json", data_files="fineTuningDT/inference_data.json", field="data")
data_inf1.push_to_hub("namejun12000/PALR_inference1")

# 데이터 구조 초기화 (inf2 valid)
inference_data2 = {"data": []}

# create json
for uid in user_item_name_seq.keys():
  if uid in valid_candidates_mapping and uid in user_preference_valid:
    inference_data2['data'].append({
        "instruction": "Recommend 10 other items based on user's history from the candidate list.",
        "input": {
          "user_id": uid,
          "preference": user_preference_valid[uid],
          "interaction": user_item_name_seq[uid],
          "candidates": valid_candidates_mapping[uid]
        },
        "output": {
          "recommended": valid_top10_mapping[uid]
        }
    })

# json 저장
with open("fineTuningDT/inference_data2.json", "w", encoding='utf-8') as json_file:
  json.dump(inference_data2, json_file, ensure_ascii=False, indent=4)

print("JSON 파일이 성공적으로 생성되었습니다. (Inference [valid])\n")

data_inf2 = load_dataset("json", data_files="fineTuningDT/inference_data2.json", field="data")
data_inf2.push_to_hub("namejun12000/PALR_inference2")

# 시드 고정
random.seed(42)
np.random.seed(42)

# Custom Dataset into hf
code_dataset4 = "namejun12000/PALR_finetuning20"
code_dataset_inf1 = "namejun12000/PALR_inference1"
code_dataset_inf2 = "namejun12000/PALR_inference2"

# fine-tuning (20%)
dataset_20_try4 = load_dataset(code_dataset4, split="train[:20%]") # 20% 
dataset_80_try4 = load_dataset(code_dataset4, split="train[20%:]") # 80%

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
dataset_dict4.push_to_hub("namejun12000/PALR_finetuning20")
dataset_dict_inf1.push_to_hub("namejun12000/PALR_inference1")
dataset_dict_inf2.push_to_hub("namejun12000/PALR_inference2")

print("Upload jsons in HuggingFace are completed!\n")