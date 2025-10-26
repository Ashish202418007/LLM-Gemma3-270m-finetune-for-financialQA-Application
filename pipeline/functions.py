import re, string , json
from collections import Counter
from typing import List, Dict
import json
import torch
import mlflow
from typing import List, Dict, Optional, Tuple
from datasets import Dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from tqdm import tqdm
from config import TrainingConfig
from transformers import TrainerCallback
from peft import PeftModel
from difflib import SequenceMatcher



def load_jsonl(file_path: str) -> List[Dict]:
    """Load JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data



# ============================================================================
# Callbacks
# ============================================================================

class EarlyStoppingCallback(TrainerCallback):
    """Custom early stopping callback with configurable patience"""
    
    def __init__(self, patience: int = 5, threshold: float = 0.0):
        self.patience = patience
        self.threshold = threshold
        self.best_metric = None
        self.counter = 0
        
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        
        metric_value = metrics.get(args.metric_for_best_model)
        if metric_value is None:
            return
        
        # Adjust for greater_is_better
        if args.greater_is_better:
            metric_value = -metric_value
            
        if self.best_metric is None:
            self.best_metric = metric_value
        elif metric_value < self.best_metric - self.threshold:
            self.best_metric = metric_value
            self.counter = 0
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            print(f"\nEarly stopping triggered after {self.counter} evaluations without improvement")
            control.should_training_stop = True
            
        return control


class QAEvaluationCallback(TrainerCallback):
    """Callback to run QA-specific evaluation (EM and F1) during training"""
    
    def __init__(self, eval_dataset, tokenizer, config: TrainingConfig):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.config = config
        self.best_f1 = 0.0
        
    def on_evaluate(self, args, state, control, model, metrics=None, **kwargs):
        """Run QA evaluation and log EM/F1 scores"""
        if metrics is None or not hasattr(self.eval_dataset, 'raw_data'):
            return
        
        print("\n" + "="*60)
        print("Running QA Evaluation (EM/F1)...")
        print("="*60)
        
        # Sample a subset for faster evaluation during training
        max_eval_samples = max(100, len(self.eval_dataset.raw_data))
        eval_samples = self.eval_dataset.raw_data[:max_eval_samples]
        
        predictions = []
        references = []
        
        model.eval()
        with torch.no_grad():
            for example in tqdm(eval_samples, desc="Evaluating QA"):
                context = example['context']
                question = example['question']
                ground_truths = example['answers']
                
                # Generate answer
                pred = self._generate_answer(model, context, question)
                
                predictions.append(pred)
                references.append(ground_truths)
        
        # Compute metrics
        qa_metrics = compute_metrics(predictions, references)
        
        # Log to MLflow
        if mlflow.active_run():
            mlflow.log_metric("qa_exact_match", qa_metrics['exact_match'], step=state.global_step)
            mlflow.log_metric("qa_f1_score", qa_metrics['f1'], step=state.global_step)
        
        # Update best F1
        if qa_metrics['f1'] > self.best_f1:
            self.best_f1 = qa_metrics['f1']
        
        print(f"QA Metrics - EM: {qa_metrics['exact_match']:.2f}%, F1: {qa_metrics['f1']:.2f}%")
        print(f"Best F1 so far: {self.best_f1:.2f}%")
        print("="*60 + "\n")
        
        return control
    
    def _generate_answer(self, model, context: str, question: str) -> str:
        """Generate answer for a question"""
        messages = [
            {'role': 'system', 'content': self.config.system_prompt},
            {'role': 'user', 'content': f"Context:\n{context}\n\nQuestion: {question}"}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        ).removeprefix('<bos>')
        
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
        ).to(model.device)
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=self.config.final_eval_max_new_tokens,
            temperature=self.config.final_eval_temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k
        )


        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=False)

        split_marker = "<start_of_turn>model"
        answer = full_output.split(split_marker, 1)[-1].strip()
        answer = answer.replace("<end_of_turn>", "").strip()
        
        return answer


# ============================================================================
# Evaluiation Metrics
# ============================================================================


def normalize_answer(s: str) -> str:
    """Normalize text: lowercase, remove articles, punctuation (keep $ and .), extra whitespace, normalize numbers."""
    
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation) - set(['$', '.'])
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    def normalize_numbers(text):
        # Remove commas from numbers
        return re.sub(r'(\d),(\d)', r'\1\2', text)
    
    return white_space_fix(remove_articles(remove_punc(lower(normalize_numbers(s)))))


def f1_score(prediction: str, ground_truth: str) -> float:
    """Calculate token-level F1 score."""
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    
    if not pred_tokens or not truth_tokens:
        return int(pred_tokens == truth_tokens)
    
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
    
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    
    return f1


def soft_f1_score(prediction: str, ground_truth: str) -> float:
    """Soft F1 using SequenceMatcher for long paraphrased answers."""
    pred_norm = normalize_answer(prediction)
    gt_norm = normalize_answer(ground_truth)
    
    if not pred_norm or not gt_norm:
        return int(pred_norm == gt_norm)
    
    return SequenceMatcher(None, pred_norm, gt_norm).ratio()


def exact_match_score(prediction: str, ground_truth: str) -> float:
    """Calculate exact match score."""
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction: str, ground_truths: List[str]) -> float:
    """Compute max score for a prediction over multiple ground truths."""
    scores = [metric_fn(prediction, gt) for gt in ground_truths]
    return max(scores) if scores else 0.0


def compute_metrics(predictions: List[str], references: List[List[str]]) -> Dict[str, float]:
    """Compute EM, token F1, soft F1, and combined score."""
    
    em_total = f1_total = soft_f1_total = 0
    total = len(predictions)
    
    for pred, gts in zip(predictions, references):
        em_total += metric_max_over_ground_truths(exact_match_score, pred, gts)
        f1_total += metric_max_over_ground_truths(f1_score, pred, gts)
        soft_f1_total += metric_max_over_ground_truths(soft_f1_score, pred, gts)
    
    em_avg = em_total / total if total > 0 else 0
    f1_avg = f1_total / total if total > 0 else 0
    soft_f1_avg = soft_f1_total / total if total > 0 else 0
    
    combined_score = 0.7 * f1_avg + 0.3 * em_avg  # original combined formula
    
    return {
        'exact_match': em_avg,
        'f1': f1_avg,
        'soft_f1': soft_f1_avg,
        'total': total,
        'combined_score': combined_score
    }


# ============================================================================
# Dataset Preparation
# ============================================================================

class QADataset:
    """Wrapper to keep raw data alongside tokenized data"""
    def __init__(self, tokenized_dataset, raw_data):
        self.tokenized_dataset = tokenized_dataset
        self.raw_data = raw_data
    
    def __len__(self):
        return len(self.tokenized_dataset)
    
    def __getitem__(self, idx):
        return self.tokenized_dataset[idx]


def prepare_dataset(
    data: List[Dict],
    tokenizer,
    config: TrainingConfig,
    keep_raw: bool = False
) -> Dataset:
    """Prepare dataset with optional raw data retention"""
    
    # Global flag for printing first example only
    first_example_printed = {"done": False}
    
    def convert_to_chatml(example):
        """Convert to chat format"""
        if "answers" in example:
            if isinstance(example["answers"], list):
                answer_text = example["answers"][0]
            else:
                answer_text = example["answers"]
        else:
            answer_text = "No answer provided"

        return {
            "conversations": [
                {"role": "system", "content": config.system_prompt},
                {"role": "user", "content": f"Context:\n{example['context']}\n\nQuestion: {example['question']}"},
                {"role": "assistant", "content": answer_text},
            ]
        }
    
    def formatting_prompts_func(examples):
        """Format conversations into text"""
        convos = examples["conversations"]
        texts = []
        
        for idx, convo in enumerate(convos):
            formatted_text = tokenizer.apply_chat_template(
                convo,
                tokenize=False,
                add_generation_prompt=False
            ).removeprefix("<bos>")
            texts.append(formatted_text)
            
            # Print only first example, only once
            if idx == 0 and not first_example_printed["done"]:
                print("\n" + "="*70)
                print("TRAINING FORMAT (First Example)")
                print("="*70)
                print(formatted_text)
                print("="*70)
                
                # Detect chat tokens
                special_tokens = {
                    "Gemma": ["<start_of_turn>", "<end_of_turn>"],
                    "Llama3": ["<|start_header_id|>", "<|end_header_id|>"],
                    "Qwen": ["<|im_start|>", "<|im_end|>"],
                    "Mistral": ["[INST]", "[/INST]"]
                }
                
                for model_type, tokens in special_tokens.items():
                    if any(token in formatted_text for token in tokens):
                        print(f"✓ Detected {model_type} format: {tokens}")
                        break
                
                first_example_printed["done"] = True
        
        return {"text": texts}
    
    def tokenize_fn(examples):
        """Tokenize the text"""
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=config.max_seq_length,
        )
        return tokenized
    
    # Convert to dataset
    dataset = Dataset.from_list(data)
    dataset = dataset.map(convert_to_chatml)
    dataset = dataset.map(formatting_prompts_func, batched=True)
    dataset = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
    
    if keep_raw:
        dataset.raw_data = data
    
    return dataset


def prepare_train_eval_datasets(
    config: TrainingConfig,
    tokenizer
) -> Tuple[Dataset, Optional[Dataset]]:
    """Prepare training and evaluation datasets with snippet printing (raw + tokenized)"""
    
    print(f"\nLoading training data from: {config.train_data_path}")
    train_data = load_jsonl(config.train_data_path)
    print(f"Loaded {len(train_data)} training samples")
    
    # Load or split evaluation data
    if config.eval_data_path:
        print(f"Loading evaluation data from: {config.eval_data_path}")
        eval_data = load_jsonl(config.eval_data_path)
        print(f"Loaded {len(eval_data)} evaluation samples")
    elif config.eval_split_ratio > 0:
        print(f"Splitting dataset: {1-config.eval_split_ratio:.1%} train, {config.eval_split_ratio:.1%} eval")
        split_idx = int(len(train_data) * (1 - config.eval_split_ratio))
        eval_data = train_data[split_idx:]
        train_data = train_data[:split_idx]
    else:
        eval_data = None
    
    # Prepare datasets
    print("\nPreparing training dataset...")
    train_dataset = prepare_dataset(train_data, tokenizer, config, keep_raw=True)
    
    eval_dataset = None
    if eval_data:
        print("Preparing evaluation dataset...")
        eval_dataset = prepare_dataset(eval_data, tokenizer, config, keep_raw=True)
    
    
    print(f"Training samples: {len(train_dataset)}")
    if eval_dataset:
        print(f"Evaluation samples: {len(eval_dataset)}")
    
    return train_dataset, eval_dataset



# ============================================================================
# Model Setup
# ============================================================================

def setup_model_and_tokenizer(config: TrainingConfig):
    """Initialize model and tokenizer with LoRA"""
    
    print(f"Loading model: {config.model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        load_in_4bit=config.load_in_4bit,
        load_in_8bit=config.load_in_8bit,
        full_finetuning=config.full_finetuning
    )
    
    print("Applying LoRA...")
    if not isinstance(model, PeftModel):
        model = FastLanguageModel.get_peft_model(
            model,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias=config.lora_bias,
            target_modules=config.lora_target_modules,
            random_state=config.seed
        )
    else:
        print("LoRA adapters already added. Skipping adapter creation.")
    
    print("Setting up chat template...")
    tokenizer = get_chat_template(
        tokenizer,
        chat_template=config.chat_template
    )
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


# ============================================================================
# Final Evaluation
# ============================================================================

# def run_final_evaluation(
#     model,
#     tokenizer,
#     config: TrainingConfig,
#     eval_data_path: str = None
# ) -> Dict:
#     """Run comprehensive evaluation on test/validation set"""
    
#     # Determine which data to use
#     if eval_data_path is None:
#         eval_data_path = config.final_eval_data_path or config.eval_data_path
    
#     if eval_data_path is None:
#         print("No evaluation data path specified for final evaluation")
#         return {}
    
#     print("\n" + "="*70)
#     print("RUNNING FINAL EVALUATION")
#     print("="*70)
#     print(f"Evaluation data: {eval_data_path}")
    
#     # Load evaluation data
#     eval_data = load_jsonl(eval_data_path)
    
#     if config.final_eval_max_samples:
#         eval_data = eval_data[:config.final_eval_max_samples]
    
#     print(f"Evaluating on {len(eval_data)} samples...")
    
#     predictions = []
#     references = []
#     detailed_results = []
#     start_time = time.time()
    

#     model.eval()
#     with torch.no_grad():
#         for example in tqdm(eval_data, desc="Final Evaluation"):
#             context = example['context']
#             question = example['question']
#             ground_truths = example['answers']
            
#             # Create prompt
#             messages = [
#                 {'role': 'system', 'content': config.system_prompt},
#                 {'role': 'user', 'content': f"Context:\n{context}\n\nQuestion: {question}"}
#             ]
            
#             text = tokenizer.apply_chat_template(
#                 messages,
#                 tokenize=False,
#                 add_generation_prompt=True,
#             ).removeprefix('<bos>')
            
#             # Generate answer
#             inputs = tokenizer(
#                 text,
#                 return_tensors="pt",
#                 truncation=True,
#                 max_length=config.max_seq_length
#             ).to(model.device)
            
#             outputs = model.generate(
#                 **inputs,
#                 max_new_tokens=config.final_eval_max_new_tokens,
#                 temperature=config.final_eval_temperature,
#                 top_p = config.top_p,
#                 top_k = config.top_k,
#                 do_sample=config.final_eval_temperature > 0,
#                 pad_token_id=tokenizer.pad_token_id,
#             )
            
#             full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
#             pred = full_output[len(text):].strip()
#             pred = pred.split('\n')[0].strip()
            
#             predictions.append(pred)
#             references.append(ground_truths)
            
#             detailed_results.append({
#                 'id': example['key'],
#                 'question': question,
#                 'prediction': pred,
#                 'ground_truths': ground_truths,
#                 'em': metric_max_over_ground_truths(exact_match_score, pred, ground_truths),
#                 'f1': metric_max_over_ground_truths(f1_score, pred, ground_truths)
#             })
    
#     elapsed_time = time.time() - start_time
    
#     # Compute metrics
#     metrics = compute_metrics(predictions, references)
#     metrics['samples_per_second'] = len(eval_data) / elapsed_time
#     metrics['total_time_seconds'] = elapsed_time
#     metrics['avg_time_per_sample'] = elapsed_time / len(eval_data)
    
#     # Log to MLflow
#     if mlflow.active_run():
#         mlflow.log_metric("final_exact_match", metrics['exact_match'])
#         mlflow.log_metric("final_f1_score", metrics['f1'])
#         mlflow.log_metric("final_total_samples", metrics['total'])
#         mlflow.log_metric("final_samples_per_second", metrics['samples_per_second'])
        
#         # Log score distributions
#         em_scores = [r['em'] for r in detailed_results]
#         f1_scores = [r['f1'] for r in detailed_results]
#         mlflow.log_metric("final_em_std", float(np.std(em_scores)))
#         mlflow.log_metric("final_f1_std", float(np.std(f1_scores)))
        
#         # Save detailed results
#         results_file = Path(config.output_dir) / "final_evaluation_results.json"
#         results_file.parent.mkdir(parents=True, exist_ok=True)
        
#         output_data = {
#             'model_name': config.model_name,
#             'eval_data_path': eval_data_path,
#             'timestamp': datetime.now().isoformat(),
#             'metrics': metrics,
#             'config': {
#                 'max_new_tokens': config.final_eval_max_new_tokens,
#                 'temperature': config.final_eval_temperature,
#                 'max_seq_length': config.max_seq_length,
#             },
#             'predictions': detailed_results
#         }
        
#         with open(results_file, 'w', encoding='utf-8') as f:
#             json.dump(output_data, f, indent=2, ensure_ascii=False)
        
#         mlflow.log_artifact(str(results_file))
#         print(f"\nDetailed results saved to: {results_file}")
    
#     # Print results
#     print("\n" + "="*70)
#     print("FINAL EVALUATION RESULTS")
#     print("="*70)
#     print(f"Exact Match (EM): {metrics['exact_match']:.2f}%")
#     print(f"F1 Score: {metrics['f1']:.2f}%")
#     print(f"Total Samples: {metrics['total']}")
#     print(f"Speed: {metrics['samples_per_second']:.2f} samples/sec")
#     print(f"Avg Time/Sample: {metrics['avg_time_per_sample']:.3f} seconds")
#     print("="*70 + "\n")
    
#     return metrics


# def prepare_dataset(
#     data: List[Dict],
#     tokenizer,
#     config: TrainingConfig,
#     keep_raw: bool = False
# ) -> Dataset:
#     """Prepare dataset with optional raw data retention"""
    
#     def convert_to_chatml(example):
#         """Convert to chat format"""
#         return {
#             "conversations": [
#                 {
#                     "role": "system", 
#                     "content": config.system_prompt
#                 },
#                 {
#                     "role": "user", 
#                     "content": f"Context:\n{example['context']}\n\nQuestion: {example['question']}"
#                 },
#                 {
#                     "role": "assistant", 
#                     "content": example["answers"][0]
#                 },
#             ]
#         }
    
#     def formatting_prompts_func(examples):
#         """Format conversations into text"""
#         convos = examples["conversations"]
#         texts = [
#             tokenizer.apply_chat_template(
#                 convo,
#                 tokenize=False,
#                 add_generation_prompt=False
#             ).removeprefix("<bos>")
#             for convo in convos
#         ]
#         return {"text": texts}
    
#     def tokenize_fn(examples):
#         """Tokenize the text"""
#         return tokenizer(
#             examples["text"],
#             truncation=True,
#             padding=False,
#             max_length=config.max_seq_length,
#         )
    
#     # Convert to dataset
#     dataset = Dataset.from_list(data)
    
#     # Apply conversions
#     dataset = dataset.map(convert_to_chatml)
#     dataset = dataset.map(formatting_prompts_func, batched=True)
#     dataset = dataset.map(
#         tokenize_fn, 
#         batched=True, 
#         remove_columns=dataset.column_names
#     )
    
#     if keep_raw:
#         dataset.raw_data = data
    
#     return dataset


# def prepare_train_eval_datasets(
#     config: TrainingConfig,
#     tokenizer
# ) -> Tuple[Dataset, Optional[Dataset]]:
#     """Prepare training and evaluation datasets"""
    
#     print(f"\nLoading training data from: {config.train_data_path}")
#     train_data = load_jsonl(config.train_data_path)
#     print(f"Loaded {len(train_data)} training samples")
    
#     # Load or split evaluation data
#     if config.eval_data_path:
#         print(f"Loading evaluation data from: {config.eval_data_path}")
#         eval_data = load_jsonl(config.eval_data_path)
#         print(f"Loaded {len(eval_data)} evaluation samples")
#     elif config.eval_split_ratio > 0:
#         print(f"Splitting dataset: {1-config.eval_split_ratio:.1%} train, {config.eval_split_ratio:.1%} eval")
#         split_idx = int(len(train_data) * (1 - config.eval_split_ratio))
#         eval_data = train_data[split_idx:]
#         train_data = train_data[:split_idx]
#     else:
#         eval_data = None
    
#     # Prepare datasets
#     print("Preparing training dataset...")
#     train_dataset = prepare_dataset(train_data, tokenizer, config, keep_raw=False)
    
#     eval_dataset = None
#     if eval_data:
#         print("Preparing evaluation dataset...")
#         eval_dataset = prepare_dataset(eval_data, tokenizer, config, keep_raw=True)
    
#     print(f"Training samples: {len(train_dataset)}")
#     if eval_dataset:
#         print(f"Evaluation samples: {len(eval_dataset)}")
    
#     return train_dataset, eval_dataset
