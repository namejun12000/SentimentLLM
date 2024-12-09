import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorWithPadding
import torch
from torch.utils.data import Dataset
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from transformers import get_linear_schedule_with_warmup, EarlyStoppingCallback
from torch.optim import AdamW

# 최대 토큰 길이 설정 
max_input_length = 2048 # 최대 토큰 길이 확인 후 설정

class FinetuningDataset(Dataset):
    def __init__(self, data, tokenizer, max_input_length=max_input_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        
        # 입력과 출력을 하나의 텍스트로 결합
        full_text = (
            f"[INST] {entry['instruction']}\n"
            f"interaction history: {', '.join(entry['input']['interaction'])}\n"
            f"sentiment label: {', '.join(entry['input']['sentiments'])}\n"
            f"candidate list: {', '.join(entry['input']['candidates'])} [/INST]\n"
            f"Recommended Products: {', '.join(entry['output']['recommended'])}"
        )

        # 텍스트를 토큰화
        inputs = self.tokenizer(
            full_text, return_tensors="pt", padding="max_length",
            max_length=self.max_input_length, truncation=True
        )

        # `labels` 설정
        labels = inputs["input_ids"].clone()
        labels[inputs["attention_mask"] == 0] = -100  # 패딩 토큰은 -100으로 마스킹

        return { 
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": labels.squeeze()
        }


# 각 데이터셋에서 마지막 상호작용을 제외하고 입력 데이터 준비
def preprocess_data(example):
    interaction_history = example["input"]["interaction"]
    setiment_history = example["input"]["sentiments"]
    # 마지막 아이템을 제외한 상호작용 시퀀스를 train_items로 설정
    train_items = interaction_history[:-2]  # 마지막 상호작용 제외
    sentiment_items = setiment_history[:-2]
    example["input"]["interaction"] = train_items
    example["input"]["sentiments"] = sentiment_items
    return example

# huggingface-cli login 터미널에 입력

# Clear GPU cache
torch.cuda.empty_cache()

if os.path.exists('./results_try1'):
    os.system('rm -rf ./results_try1')  # 결과 폴더 삭제
if os.path.exists('./logs_try1'):
    os.system('rm -rf ./logs_try1')    # 로그 폴더 삭제
if os.path.exists('./fine_tuned_model_try1'):
    os.system('rm -rf ./fine_tuned_model_try1')

# cuda 사용 가능 여부 확인
if torch.cuda.is_available():
  print("CUDA is available")
  num_device = torch.cuda.device_count()
  print(f"Number of CUDA devices: {num_device}")
  
  for i in range(num_device):
    print(f"Device {i}: {torch.cuda.get_device_name(i)}")
else:
  print("CUDA is not available.")

# Load dataset
base_model = "meta-llama/Llama-2-7b-hf"
code_dataset = "namejun12000/AW_finetuning_5core_split1_all_final_valid_include"

# 20% 데이터셋 로드
dataset = load_dataset(code_dataset, split="train_20")

# 5%를 검증 데이터로 분리
eval_size = int(0.05 * len(dataset))
train_size = len(dataset) - eval_size

# 데이터 분할
train_dataset = dataset.select(range(train_size))
eval_dataset = dataset.select(range(train_size, len(dataset)))

# 데이터 전처리
train_dataset = train_dataset.map(preprocess_data)
eval_dataset = eval_dataset.map(preprocess_data)


# ##############
# # 전체 데이터셋의 3% 크기 계산
# sample_size = int(0.03 * len(dataset))

# # 3% 데이터셋 로드 (샘플링)
# train_dataset = dataset.select(range(sample_size))

# eval_dataset = load_dataset(code_dataset, split="train_80").shuffle(seed=42).select(range(5))  # 450개 샘플링; 10% 정도

# # 데이터 전처리
# train_dataset = train_dataset.map(preprocess_data)
# eval_dataset = eval_dataset.map(preprocess_data)
# ###############

print(train_dataset[3])
print(eval_dataset[3])

# 샘플링된 데이터 길이 확인
sampled_length = len(train_dataset)
eval_length = len(eval_dataset)
print(f"train dataset length (20%): {sampled_length}")
print(f"eval dataset length: {eval_length}")
print("모델 로드...")
# Load model (Llama2-7b)
model = AutoModelForCausalLM.from_pretrained(base_model, 
                                             device_map={"": 0}, 
                                             load_in_8bit=False)
model.config.use_cache = False
model.config.pretraining_tp = 1
model.to("cuda")

# LoRA configuration for memory efficiency
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.2,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# def initialize_weights(module):
#     if hasattr(module, 'reset_parameters'):
#         module.reset_parameters()

# model.apply(initialize_weights)  # 모델의 모든 가중치를 초기화

print("토크나이저 로드...")
# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'

# # 최대 토큰 길이 확인
# max_input_length = 0
# max_input_index = -1  # 가장 긴 input의 인덱스를 저장할 변수

# print("최대 토큰 길이 확인\n")

# # 모든 샘플에 대해 실제 길이 확인
# for idx, entry in enumerate(dataset):
#     # 입력과 출력을 하나의 텍스트로 결합
#     full_text = (
#         f"[INST] {entry['instruction']}\n"
#         f"interaction history: {', '.join(entry['input']['interaction'])}\n"
#         f"sentiment label: {', '.join(entry['input']['sentiments'])}\n"
#         f"candidate list: {', '.join(entry['input']['candidates'])} [/INST]\n"
#         f"Recommended Products: {', '.join(entry['output']['recommended'])}"
#     )

#     # 텍스트를 토큰화 (패딩과 트렁케이션 없이)
#     tokens = tokenizer(full_text, return_tensors="pt", padding=False, truncation=False).input_ids

#     # 각 샘플의 토큰 길이 확인
#     full_text_length = tokens.shape[1]
    
#     # 최대 길이 갱신 및 인덱스 저장
#     if full_text_length > max_input_length:
#         max_input_length = full_text_length
#         max_input_index = idx  # 가장 긴 토큰 길이를 가진 인덱스를 저장

# # 결과 출력
# print("최대 input 토큰 길이:", max_input_length)
# print("최대 input 토큰 길이를 가진 인덱스:", max_input_index)

print("토큰화...")
tokenized_dataset = FinetuningDataset(train_dataset, tokenizer)
# 검증 데이터셋도 동일한 방식으로 토큰화
eval_dataset = FinetuningDataset(eval_dataset, tokenizer)
print("토큰화 완료")

# 샘플 1 확인
sample = tokenized_dataset[0]
print(f"Input IDs length: {len(sample['input_ids'])}")
print(f"Labels length: {len(sample['labels'])}")
print("Input text:", sample['input_ids'])
print("output text:", sample['labels'])
print("Decoded input text:", tokenizer.decode(sample['input_ids'], skip_special_tokens=True))

print("data 패딩 추가")
# Data Collator 설정 (패딩 추가)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True) # FinetuningDataset Func에서 패딩을 이미 처리해서 필요하지는 않음
print("data_collator 완료")

# Training arguments 설정
training_args = TrainingArguments(
    output_dir="./results_try1",
    per_device_train_batch_size=2,  # 메모리 문제를 해결하기 위해 배치 크기를 줄임
    gradient_accumulation_steps=4,
    num_train_epochs=5,
    learning_rate = 1e-5, # lora 적용시 학습률을 높게 설정
    logging_dir="./logs_try1",
    logging_steps=200,  # 매 스텝마다 로그를 출력하도록 설정
    fp16=True,  # Mixed Precision 사용
    weight_decay=0.02, # 과적합 방지
    max_grad_norm=1.0,
    save_strategy="steps",       # save_strategy를 epoch로 설정
    # save_strategy="no",
    eval_strategy="steps",  # 평가 전략 설정
    eval_steps = 200,
    save_steps = 2000,
    save_total_limit=1,  # 최신 3개의 체크포인트만 유지
    load_best_model_at_end=True,
    report_to="tensorboard"  # 기본 콘솔 출력으로 설정 (wandb나 tensorboard를 사용하지 않는 경우)
)

# 옵티마이저와 스케줄러 설정
optimizer = AdamW(model.parameters(),
                  lr=training_args.learning_rate,
                  weight_decay=training_args.weight_decay)


num_train_steps = training_args.num_train_epochs * len(tokenized_dataset) // training_args.gradient_accumulation_steps

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * num_train_steps), # 전체 학습 단계의 10%로 설정하여 학습 초반 급격한 변화 완화
    num_training_steps=num_train_steps
)

# Early Stopping 설정
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=5, # 성능이 개선되지 않는 평가 주기의 횟수
    early_stopping_threshold=0.0005 # 개선으로 간주할 최소 성능 차이 
)


# SFTTrainer 초기화
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    optimizers=(optimizer, scheduler),
    callbacks=[early_stopping_callback],
    max_seq_length=max_input_length
)

# 파인튜닝 시작
print("Fine-tuning...\n")
trainer.train()
print("Fine-tuning completed.")
trainer.model.save_pretrained('./fine_tuned_model_try1')
print("Saved model!")

# print("Evaluating on sample of 80 dataset...\n")
# eval_results = trainer.predict(eval_dataset)

# print("Evaluation results:", eval_results.metrics)

# rm -rf ./results/*  # 체크포인트 파일 삭제
# rm -rf ./logs/*     # 로그 파일 삭제