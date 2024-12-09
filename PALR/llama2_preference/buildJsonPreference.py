import random
import numpy as np
import pandas as pd
import json
import huggingface_hub
from datasets import load_dataset

huggingface_hub.login()

# 시드 고정
random.seed(42)
np.random.seed(42)

# 1. product.txt 읽어서 매핑 생성
with open('../../data/product.txt', 'r', encoding='utf-8') as f:
    product_lines = f.readlines()

asin_to_name = {}
for line in product_lines:
    parts = line.strip().split(' ', 1)  # asin_id, product_name 분리
    if len(parts) == 2:
        asin_id = parts[0]  # 첫 번째 부분: asin_id
        product_name = parts[1]  # 두 번째 부분: product_name
        asin_to_name[asin_id] = product_name  # 딕셔너리에 저장

# 2. product_category.txt 읽어서 매핑 생성
with open('../../data/product_category.txt', 'r', encoding='utf-8') as f:
  productC_lines = f.readlines()

asin_to_category = {}
for line in productC_lines:
  parts = line.strip().split(' ', 1) # asin_id, product_category 분리
  if len(parts) == 2:
    asin_id = parts[0] # asin_id
    product_category = parts[1] # product category
    asin_to_category[asin_id] = product_category # dict 저장

# user_item_sequences.txt
user_item_seq = {}

with open("../../data/user_item_sequences.txt", "r") as f:
  user_item_lines = f.readlines()

for line in user_item_lines:
  user_id, *product_ids = line.strip().split()
  user_item_seq[user_id] = product_ids

# 사용자별 product_name 시퀀스 생성
user_product_seq = {}

for user_id, product_ids in user_item_seq.items():
    # 각 asin_id를 product_name으로 매핑
    product_names = [asin_to_name.get(asin_id, f"Unknown({asin_id})") for asin_id in product_ids]
    user_product_seq[user_id] = product_names

# # 결과 확인
# for user_id, product_names in user_product_seq.items():
#     print(f"{user_id}: {', '.join(product_names)}")
    
# # 사용자별 제품 이름과 카테고리를 결합한 형식 생성
# user_item_with_categories = {}

# for user_id, product_ids in user_item_seq.items():
#     product_details = []
#     for asin_id in product_ids:
#         product_name = asin_to_name.get(asin_id, f"Unknown({asin_id})")
#         categories = asin_to_category.get(asin_id, "Unknown Category").split(", ")
#         formatted_categories = ", ".join(categories)  # 카테고리 결합
#         product_details.append(f'{product_name}: {formatted_categories}')
    
    # user_item_with_categories[user_id] = "; ".join(product_details)

# # 결과 출력
# for user_id, product_line in user_item_with_categories.items():
#     print(f"{user_id}: {product_line}")

# JSON 데이터 생성 (test)
prefdata = {"data": []}

for uid, product_ids in user_item_seq.items():
    # 마지막 상호작용 제외
    product_ids_without_last = product_ids[:-1] if len(product_ids) > 1 else product_ids

    # # interaction_category 생성
    # interaction_category = "; ".join(
    #     [f"{asin_to_name.get(asin_id, f'Unknown({asin_id})')}: {asin_to_category.get(asin_id, 'Unknown Category')}" for asin_id in product_ids_without_last]
    # )
    interaction = [asin_to_name.get(asin_id, f'Unknown({asin_id})') for asin_id in product_ids_without_last]
    category = [asin_to_category.get(asin_id, "Unknown Category") for asin_id in product_ids_without_last]


    # JSON 데이터 추가
    prefdata['data'].append({
        "instruction": "Your task is to find two categories to summarize user's preference based on history interactions and their categories.",
        "input": {
            "user_id": uid,
            "interaction": interaction,
            "category": category
        }
    })

# json 저장
with open("prefDT/preference_data.json", "w", encoding='utf-8') as json_file:
  json.dump(prefdata, json_file, ensure_ascii=False, indent=4)

print("JSON 파일이 성공적으로 생성되었습니다.\n")

# JSON 데이터 생성 (valid)
prefdata2 = {"data": []}

for uid, product_ids in user_item_seq.items():
    # 마지막 상호작용 제외
    product_ids_without_last = product_ids[:-2] if len(product_ids) > 1 else product_ids

    # # interaction_category 생성
    # interaction_category = "; ".join(
    #     [f"{asin_to_name.get(asin_id, f'Unknown({asin_id})')}: {asin_to_category.get(asin_id, 'Unknown Category')}" for asin_id in product_ids_without_last]
    # )
    interaction = [asin_to_name.get(asin_id, f'Unknown({asin_id})') for asin_id in product_ids_without_last]
    category = [asin_to_category.get(asin_id, "Unknown Category") for asin_id in product_ids_without_last]


    # JSON 데이터 추가
    prefdata2['data'].append({
        "instruction": "Your task is to find two categories to summarize user's preference based on history interactions and their categories.",
        "input": {
            "user_id": uid,
            "interaction": interaction,
            "category": category
        }
    })

# json 저장
with open("prefDT/preference_data2.json", "w", encoding='utf-8') as json_file:
  json.dump(prefdata2, json_file, ensure_ascii=False, indent=4)

print("JSON 파일이 성공적으로 생성되었습니다.\n")

pref1 = load_dataset("json", data_files="prefDT/preference_data.json", field="data")
pref1.push_to_hub("namejun12000/PALR_preference")

pref2 = load_dataset("json", data_files="prefDT/preference_data2.json", field="data")
pref2.push_to_hub("namejun12000/PALR_preference2")
print("Upload jsons in HuggingFace are completed!\n")

