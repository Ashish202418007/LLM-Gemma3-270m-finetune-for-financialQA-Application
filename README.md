# Model Name

Gemma 3 270M FinanceQA for financial news question-answering. Fine-tuned on news and financial Q&A datasets using the Unsloth framework with LoRA PEFT (rank 16, alpha 32) and transformer reinforcement learning, it delivers financial news and is ideal for finance-focused applications.
---

## Model Overview

**Model Type:** [Gemma 3 270M]  
**Framework:** [Unsloth]  
**Package:** [transformer reinforcement learning]
**Task:** [Financial Question answering]  
**Language(s):** [English]

**Abstract:**  
The primary use case of the Gemma 3 270M FinanceQA model is efficient and accurate question answering and insight generation specifically tailored for financial news. It excels in processing financial text data, delivering structured, verifiable responses in resource-constrained environments like on-device or lightweight servers.

---

## Intended Use

**Primary Use Cases:**  
Extracting answers from news articles and financial documents.  
Assisting with research and analysis in financial and news domains.

**Out-of-Scope Use Cases / Limitations:**  
Not intended for medical, legal, or other specialized domains outside news and finance.  
Accuracy may degrade on highly technical financial texts outside the dataset’s scope.

**Ethical Considerations / Biases:**  
The model may reflect biases present in both NewsQA and financial datasets.  
Users should verify critical information independently.



## How to Use:
**using hugging face transformers**

```Bash
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Ashish-kar-007/Gemma3-270m-financeQA-finetuned-merged-16bit")
model = AutoModelForCausalLM.from_pretrained("Ashish-kar-007/Gemma3-270m-financeQA-finetuned-merged-16bit")
messages = [
    {"role": "user", "content": "Who are you?"},
]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=40)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))
```
**using sagemaker**

```bash
import json
import sagemaker
import boto3
from sagemaker.huggingface import HuggingFaceModel, get_huggingface_llm_image_uri

try:
    role = sagemaker.get_execution_role()
except ValueError:
    iam = boto3.client('iam')
    role = iam.get_role(RoleName='sagemaker_execution_role')['Role']['Arn']

# Hub Model configuration. https://huggingface.co/models
hub = {
    'HF_MODEL_ID':'Ashish-kar-007/Gemma3-270m-financeQA-finetuned-merged-16bit',
    'SM_NUM_GPUS': json.dumps(1)
}



# create Hugging Face Model Class
huggingface_model = HuggingFaceModel(
    image_uri=get_huggingface_llm_image_uri("huggingface",version="3.2.3"),
    env=hub,
    role=role, 
)

# deploy model to SageMaker Inference
predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type="ml.g5.2xlarge",
    container_startup_health_check_timeout=300,
  )
  
# send request
predictor.predict({
    "inputs": "Hi, what can you help me with?",
})

```
---
## Training Data

Whole dataset can be found here:
[https://www.kaggle.com/datasets/ashish202418007/gemma3-270m-financial-qa-finetuning-dataset]

**Datasets Used:**  
**Citation:**
1. **NewsQA** – A dataset of question-answer pairs from CNN news articles.  
   
```bash
   @inproceedings{trischler2017newsqa,
     title={NewsQA: A Machine Comprehension Dataset},
     author={Trischler, Adam and Wang, Tong and Yuan, Xingdi and Harris, Justin and Sordoni, Alessandro and Bachman, Philip and Suleman, Kaheer},
     booktitle={Proceedings of the 2nd Workshop on Representation Learning for NLP},
     year={2017}
   }
```
2. **Financial Q&A 10k** – A Kaggle dataset of financial question-answer pairs.
Source: Financial Q&A 10k on Kaggle

```bash 
@misc{saeedian2023financialqa,
  title={Financial Q\&A 10k Dataset},
  author={Yousef Saeedian},
  year={2023},
  howpublished={\url{https://www.kaggle.com/datasets/yousefsaeedian/financial-q-and-a-10k}}
}
```


