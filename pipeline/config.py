from typing import List, Optional
from dataclasses import dataclass


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class TrainingConfig:
    """Configuration for training pipeline"""
    # Model settings
    model_name: str = "./output/NewsQA_finetune_gemma3-270m/final_model"
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    full_finetuning: bool = False
    
    # LoRA settings
    lora_r: int = 64
    lora_alpha: int = 32
    lora_dropout: float = 0.00
    lora_bias: str = "none"
    lora_target_modules: List[str] = None
    
    # Data settings
    train_data_path: str = "./data/Financial_QA_train.jsonl"
    eval_data_path: str = None 
    eval_split_ratio: float = 0.00  # Used only if eval_data_path is None
    max_seq_length: int = 1024
    chat_template: str = "gemma-3"
    system_prompt: str = '''Answer the question based on the context below. Be concise and extract the answer directly from the context.'''
    
    # Training settings
    output_dir: str = "output"
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    num_train_epochs: int = 3
    max_steps: int = -1  # -1 means use num_train_epochs
    learning_rate: float = 2.1e-5
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    optim: str = "adamw_8bit"
    
    # Evaluation settings during training
    if eval_split_ratio > 0.0 or eval_data_path is not None:
        eval_strategy: str = "steps"
        eval_steps: int = 50
        metric_for_best_model: str = "eval_loss"
        load_best_model_at_end: bool = True

    save_strategy: str = "steps"
    save_steps: int = 50
    save_total_limit: int = 3


    greater_is_better: bool = False
    
    # Early stopping
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.001
    
    # Generation settings
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 64

    # Logging
    logging_steps: int = 10
    report_to: str = "mlflow"
    
    # MLflow settings
    mlflow_tracking_uri: str = "mlruns"
    mlflow_experiment_name: str = "llm_finetuning"
    mlflow_registered_model_name: Optional[str] = None
    
    # Misc
    seed: int = 42
    fp16: bool = False
    bf16: bool = True
    
    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj", 
                "up_proj", "down_proj"
            ]

