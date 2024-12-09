import torch 
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# 긴 리뷰를 512 토큰 이하로 분할하는 함수
def split_into_chunks(review_text, tokenizer, max_length=500):
    inputs = tokenizer(review_text, return_tensors='pt', truncation=False)
    input_ids = inputs['input_ids'][0]

    # 입력 토큰을 512개 이하로 분할
    chunks = []
    for i in range(0, len(input_ids), max_length):
        chunk_ids = input_ids[i:i+max_length]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk_text)
    
    return chunks

# 감성 분석 실행 함수
def run_sentiment_analysis(input_file, output_file):
    # Set device to MPS if available
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    # import model and tokenizer
    model_id = "SamLowe/roberta-base-go_emotions"  # RoBERTa model fine-tuned for emotions
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.to(device)

    # sentiment pipeline 설정
    sentiment_pipeline = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        top_k=None,  # Get scores for all emotion labels
        truncation=True, 
        device=0 if torch.cuda.is_available() else -1
    )

    # 감성 레이블을 5개 강도 범주로 매핑하는 딕셔너리
    emotion_strength_mapping = {
        "admiration": 2, "approval": 2, "joy": 2, "love": 2, "pride": 2, "relief": 2, "gratitude": 2,
        "curiosity": 1, "excitement": 1, "desire": 1, "amusement": 1, "caring": 1, "optimism": 1,
        "neutral": 0, "realization": 0, "surprise": 0,
        "confusion": -1, "disappointment": -1, "embarrassment": -1,
        "grief": -2, "anger": -2, "annoyance": -2, "disapproval": -2, "fear": -2, "nervousness": -2, "remorse": -2, "sadness": -2, "disgust": -2
    }

    # 감성 강도에 따른 레이블 매핑
    strength_label_mapping = {
        2: "Strong Positive",
        1: "Positive",
        0: "Neutral",
        -1: "Negative",
        -2: "Strong Negative"
    }

    # Load data
    with open(input_file, 'r') as file:
        lines = file.readlines()

    line_count = len(lines)
    print(f"length of text: {line_count}\n")

    # 감정 분석 결과를 저장할 리스트
    sentiment_results = []

    # 각 줄에서 user_id, item_id, review_text를 추출하는 부분에 예외 처리 추가
    for idx, line in enumerate(lines, start=1):
        try:
            # 각 줄을 세 개의 값으로 분리
            parts = line.strip().split(' ', 2)
            user_id, item_id, review_text = parts

            # 리뷰를 512 토큰 이하로 분할
            chunks = split_into_chunks(review_text, tokenizer)

            all_sentiments = []
            
            # 각 분할된 부분에 대해 감정 분석 수행
            for chunk in chunks:
                sentiments = sentiment_pipeline(chunk)  
                all_sentiments.append(sentiments[0])  # 모든 부분의 감정 분석 결과 저장

            final_sentiment = max(all_sentiments, key=lambda s: max([x['score'] for x in s]))
            dominant_emotion = final_sentiment[0]['label']
            dominant_score = final_sentiment[0]['score']
            
            # 감성 레이블을 감성 강도로 변환
            emotion_strength = emotion_strength_mapping.get(dominant_emotion, -100)
            # 감성 강도에 따른 범주 레이블
            strength_label = strength_label_mapping.get(emotion_strength, "Unknown")

            sentiment_results.append({
                'user_id': user_id,
                'item_id': item_id,
                'dominant_emotion': dominant_emotion,
                # 'sentiment_score': dominant_score,
                'emotion_strength': emotion_strength,
                'strength_label': strength_label
            })

            print(f"진행 상황: {idx}/{line_count} 라인 처리 완료")

        except Exception as e:
            print(f"Error processing line {idx}: {e}")

    # 감정 분석 결과를 파일로 저장
    with open(output_file, 'w') as f:
        for result in sentiment_results:
            f.write(f"{result['user_id']} {result['item_id']} {result['dominant_emotion']} {result['emotion_strength']} {result['strength_label']}\n")

    print("Sentiment analysis completed and saved with emotion strength.")
    
    return sentiment_results

