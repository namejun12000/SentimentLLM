# PALR & SetimentLLM (Propose Model)

## Environment
Python 3.10  
Cuda 12.6  
PyTorch 2.5 Version 24.09  

## How to execute it
### PALR
Datasets/preData.py -> sas_goeR.py -> PALR/llama2_preference/buildJsonPreference.py -> PALR/llama2_preference/llmPreference.py -> PALR/llama2_finetuning/buildJsonFinetuning.py -> PALR/llama2_finetuning/llmTrainerPALR.py -> PALR/llama2_finetuning/llmInference.py

### SentimentLLM
Datasets/preData.py -> sas_goeR.py -> buildJson.py -> llama2_finetuning20_new/llmTrainerTry1.py -> llama2_finetuning20_new/llmInferenceTry1.py

## 1. Datasets
The **data** repository contains two *json.gz* files (Original Amazon Beauty Dataset), *preData.py* (for data preprocessing), and five *.txt* files (preprocessed data).

## 2. SASRec
The retrieval layer for PALR & SentimentLLM.
We execute *sas_goeR.py* to create *beauty_review_results*, which includes SASRec results and the top 50 candidates ([test, valid, test+valid]).

## 3. SentimentLLM (Propose Model)

### 3.1 goeRobert 
This repository contains *goeRoberta.py* and *sentiment_strength_results.txt*.
*goeRoberta.py* extracts sentiment labels from user reviews and creates sentiment label sequences for each user-item pair in *sentiment_strength_results.txt*.
We also execute *sas_goeR.py* to create *sentiment_strength_results.txt* (by including inline comments for the SASRec part).

### 3.2 llama2_finetuning20_new
Before fine-tuning and inference, we execute *buildJson.py* to create prompt templates for fine-tuning and inference.
This repository contains an LLama2-7b for fine-tuning (20% of user samples) and inference. The model's test results can be found in *inference_results*.

## 4. PALR
This repository contains *llama2_finetuning* and *llama2_preference*.
The *llama2_preference* repository creates user profile summaries using the LLama2-7b. Before creating user profile summaries, we execute *buildJsonPreference.py* to generate prompt templates.
The *llama2_finetuning20* repository contains an LLM for fine-tuning (20% of user samples) and inference. Before executing fine-tuning and inference, we execute *buildJsonFinetuning.py* to generate prompt templates for fine-tuning and inference.
The model's test results are available in *inference_results*.

