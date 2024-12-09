import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
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


# 모델을 GPU로 이동
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# 데이터셋 로드
dataset = load_dataset("namejun12000/PALR_preference")['train']  # 필요한 split 로드 (test)
dataset2 = load_dataset("namejun12000/PALR_preference2")['train']  # 필요한 split 로드 (valid)

# sampled_dataset = dataset.select(range(0,5))

def gen_preference1(model, dataset, tokenizer):
    results = []

    for example in tqdm(dataset, desc="Generate User Preference", unit="user"):
        user_id = example["input"]["user_id"]
        interaction = example["input"]["interaction"]
        category = example["input"]["category"]
        instruction = example["instruction"]

        interaction_str = ", ".join(interaction)
        category_str = ", ".join(category)

        input_text = (
            f"{instruction}\n"
            f"History Products:\n{interaction_str}\n"
            f"Categories:\n{category_str}\n"
            "Output the top two categories in this format:\n"
            "'Category_1', 'Category_2'.\nOutput: "
        )

        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                temperature=0.2,
                top_k=3,
                top_p=0.8,
                repetition_penalty=1.2
            )
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # print(f"Generated text: {generated_text}")  # 디버깅용

            # Output 뒤의 값만 추출
            generated_text_after_output = re.search(r"Output:\s*(.*)", generated_text, re.DOTALL)
            if generated_text_after_output:
                output_content = generated_text_after_output.group(1).strip()
                match = re.findall(r"'(.*?)'", output_content)
                if match and len(match) >= 2:
                    selected_categories = match[:2]
                else:
                    selected_categories = ["Unknown", "Unknown"]
                    print(f"Warning: Unexpected output format: {output_content}")
            else:
                selected_categories = ["Unknown", "Unknown"]
                print(f"Warning: 'Output:' not found in text: {generated_text}")

            results.append({
                "user_id": user_id,
                "categories": selected_categories,
                "output_text": f"The user is interested in '{selected_categories[0]}' and '{selected_categories[1]}' products."
            })

    return results



preferences = gen_preference1(model, dataset, tokenizer)

# 결과를 텍스트 파일로 저장
output_file = "user_preferences.txt"
with open(output_file, "w", encoding="utf-8") as f:
    for preference in preferences:
          # user_id와 output_text를 합쳐서 저장
        f.write(f"{preference['user_id']} {preference['output_text']}\n")

print("Test Prefenece Generated!!!\n")

preferences2 = gen_preference1(model, dataset2, tokenizer)
# 결과를 텍스트 파일로 저장 (Valid)
output_file1 = "user_preferences2.txt"
with open(output_file1, "w", encoding="utf-8") as f:
    for preference in preferences2:
        f.write(f"{preference['user_id']} {preference["output_text"]}\n")  # output_text만 저장

print("Valid Prefenece Generated!!!\n")
