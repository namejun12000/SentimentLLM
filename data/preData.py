import pandas as pd
import gzip
import re
import emoji
from sklearn.preprocessing import LabelEncoder
import random  # Random 시드 고정
import numpy as np  # NumPy 시드 고정
import html

# 시드 고정
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# file path
file1 = "reviews_Beauty_5.json.gz"
file2 = 'meta_Beauty.json.gz'

def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)

# get dataframe
def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

# 텍스트 정리 함수
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # 특수 문자 제거
    text = re.sub(r'\s+', ' ', text)  # 연속된 공백을 하나로
    return text.strip()  # 앞뒤 공백 제거

# 이모지 제거 함수
def remove_emoji(text):
    return emoji.replace_emoji(text, replace='') if isinstance(text, str) and text else ''

# Function to create a natural language summary for Llama model compatibility
def format_natural_language_summary(categories_str):
    try:
        categories_list = eval(categories_str)
        main_category = categories_list[0][1] if len(categories_list[0]) > 1 else None
        sub_category = categories_list[0][2] if len(categories_list[0]) > 2 else None
        # Format with natural language
        if sub_category:
            return f"{main_category} - {sub_category}"
        else:
            return f"{main_category}"
    except:
        return None
    
# load and save full beauty dataset
df= getDF(file1)
df2 = getDF(file2)

# 중복 데이터만 필터링하여 확인 (user_id, parent_asin, timestamp 열 기준)
duplicates = df[df.duplicated(subset=['reviewerID', 'asin', 'unixReviewTime'], keep=False)]
print(f"Check Duplicates: {duplicates}")

# 필요한 열만 선택
df = df[['reviewerID', 'asin', 'unixReviewTime', 'overall', 'reviewText']]

# 데이터 타입 변환 및 기본 전처리
df['reviewText'] = df['reviewText'].astype(str)
df2['title'] = df2['title'].astype(str)
df2['categories'] = df2['categories'].astype(str)


# HTML 엔티티 처리 후 추가 전처리 (특수문자 제거, 이모지 제거)
df2['title'] = df2['title'].apply(lambda x: clean_text(remove_emoji(html.unescape(x))))

# Apply the function to df2['categories'] and store the result in a new column 'second_category'
df2['second_category'] = df2['categories'].apply(format_natural_language_summary)

# 필요한 열만 선택 (meta)
df2 = df2[['asin', 'title', 'second_category']]

# 결측값 처리
df2['title'] = df2['title'].fillna('Unknown Title')
df2['second_category'] = df2['second_category'].fillna('Unknown Category')

missing_values = df2[['title', 'second_category']].isnull().sum()
print(f"Meta dataset missing values: {missing_values}")

# 사용자별 상호작용 정렬
df = df.sort_values(by=['reviewerID', 'unixReviewTime'], ascending=True)

## 사용자별 최신 20개 상호작용만 유지
# df = df.groupby('reviewerID').tail(20).reset_index(drop=True)

# 텍스트 전처리
df['reviewText'] = df['reviewText'].apply(lambda x: clean_text(remove_emoji(x)))

# 열 이름 변경
df = df.rename(columns={
    'reviewerID': 'user_id', 
    'asin': 'parent_asin', 
    'unixReviewTime': 'timestamp', 
    'overall': 'rating',
    'reviewText': 'text'
})

df2 = df2.rename(columns={
    'asin': 'parent_asin',
    'title': 'title',
    'second_category': 'second_category'
})


# 빈 문자열을 NaN으로 변환 후 결측값 처리 (chained assignment 경고를 방지하기 위해 loc 사용)
df.loc[df['text'] == '', 'text'] = float('NaN')
df.loc[df['text'] == ' ', 'text'] = float('NaN')
df['text'] = df['text'].fillna('product')

# review + meta csv
df = pd.merge(df, df2[['parent_asin','title', 'second_category']], on='parent_asin', how='left')

# 레이블 인코딩
le_user = LabelEncoder()
le_item = LabelEncoder()
df['user_id'] = le_user.fit_transform(df['user_id']) + 1
df['parent_asin_id'] = le_item.fit_transform(df['parent_asin']) + 1

# 데이터 저장
df.to_csv("beauty.csv", encoding='utf-8', index=False)
df2.to_csv('meta.csv', encoding='utf-8', index=False)

# SASRec 용 텍스트 생성
with open('beauty_review.txt', 'w') as f:
    for _, row in df.iterrows():
        f.write(f"{row['user_id']} {row['parent_asin_id']}\n")

# 사용자-아이템 상호작용 시퀀스 생성
user_item_seq = df.groupby('user_id')['parent_asin_id'].apply(list).reset_index()
user_item_seq.columns = ['user_id', 'item_seq']

with open("user_item_sequences.txt", 'w') as f:
    for _, row in user_item_seq.iterrows():
        item_sequence_str = ' '.join(map(str, row['item_seq']))
        f.write(f"{row['user_id']} {item_sequence_str}\n")

# 사용자별 20개 초과 상호작용을 갖는 사용자 확인
users_with_more_than_20_items = user_item_seq[user_item_seq['item_seq'].str.len() > 20]
print(f"Number of users with more than 20 items: {len(users_with_more_than_20_items)}")

# 감성 분석 모델 입력용 텍스트 생성
with open('only_review.txt', 'w') as f:
    for _, row in df.iterrows():
        f.write(f"{row['user_id']} {row['parent_asin_id']} {row['text']}\n")

# product name txt
with open('product.txt', 'w') as f:
    for _, row in df.iterrows():
        f.write(f"{row['parent_asin_id']} {row['parent_asin']} ({row['title']})\n")

# product cateory txt
with open('product_category.txt', 'w') as f:
    for _, row in df.iterrows():
        f.write(f"{row['parent_asin_id']} {row['second_category']}\n")

print(f"Finished Data preprocessing!")

