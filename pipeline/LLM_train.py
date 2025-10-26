"""
End-to-End LLM Fine-tuning Pipeline 
"""

import torch
import mlflow
from mlflow import transformers
from pathlib import Path
from typing import List, Optional
from unsloth.chat_templates import train_on_responses_only
from trl import SFTTrainer, SFTConfig
from datetime import datetime
from functions import (setup_model_and_tokenizer,
                      prepare_train_eval_datasets,
                      EarlyStoppingCallback, QAEvaluationCallback)
from config import TrainingConfig 
import argparse
from dataclasses import fields, MISSING
from transformers import DataCollatorForLanguageModeling


def train_model(config: TrainingConfig):
    """End-to-end training pipeline with evaluation"""
    
    # Setup MLflow
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.mlflow_experiment_name)
    
    with mlflow.start_run():
        # Log configuration
        mlflow.log_params({
            "model_name": config.model_name,
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "learning_rate": config.learning_rate,
            "batch_size": config.per_device_train_batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "num_train_epochs": config.num_train_epochs,
            "lr_scheduler_type": config.lr_scheduler_type,
            "eval_split_ratio": config.eval_split_ratio,
            "max_seq_length": config.max_seq_length,
            "early_stopping_patience": config.early_stopping_patience,
        })
        
        model, tokenizer = setup_model_and_tokenizer(config)
        train_dataset, eval_dataset = prepare_train_eval_datasets(config, tokenizer)


        if torch.cuda.is_available():
            gpu_stats = torch.cuda.get_device_properties(0)
            start_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
            max_memory = round(gpu_stats.total_memory / 1024**3, 3)
            print(f"\nGPU: {gpu_stats.name}")
            print(f"Max memory: {max_memory} GB")
            print(f"Reserved memory: {start_memory} GB")
        
        # training arguments
        training_args = SFTConfig(
            dataset_text_field=None,
            output_dir=config.output_dir,
            
            # Batch sizes
            per_device_train_batch_size=config.per_device_train_batch_size,
            per_device_eval_batch_size=config.per_device_eval_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            
            # Training steps
            num_train_epochs=config.num_train_epochs,
            max_steps=config.max_steps,
            
            # Optimizer settings
            learning_rate=config.learning_rate,
            warmup_ratio=config.warmup_ratio,
            weight_decay=config.weight_decay,
            optim=config.optim,
            
            # Learning rate scheduler
            lr_scheduler_type=config.lr_scheduler_type,
            
            # Evaluation
            eval_strategy=config.eval_strategy if eval_dataset else "no",
            eval_steps=config.eval_steps if eval_dataset else None,
            
            # Saving
            save_strategy=config.save_strategy,
            save_steps=config.save_steps,
            save_total_limit=config.save_total_limit,
            load_best_model_at_end=config.load_best_model_at_end if eval_dataset else False,
            metric_for_best_model=config.metric_for_best_model if eval_dataset else None,
            greater_is_better=config.greater_is_better,
            
            # Logging
            logging_steps=config.logging_steps,
            report_to=config.report_to,
            
            # Misc
            seed=config.seed,
            fp16=config.fp16,
            bf16=config.bf16,

            # Gradient clipping
            max_grad_norm=1.0
        )
        data_collator = DataCollatorForLanguageModeling(
                        tokenizer=tokenizer,
                        mlm=False, 
)
        # Setup trainer
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=None,
            args=training_args,
            data_collator=data_collator
        )
        
        # Configure response-only training
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<start_of_turn>user\n",
            response_part="<start_of_turn>model\n",
        )
        
        # callbacks
        if eval_dataset:
            print(f"\n✓ Train: {len(train_dataset)} samples")
            print(f"✓ Eval: {len(eval_dataset)} samples")
            # Early stopping
            if config.early_stopping_patience > 0:
                early_stopping = EarlyStoppingCallback(
                    patience=config.early_stopping_patience,
                    threshold=config.early_stopping_threshold
                )
                trainer.add_callback(early_stopping)
            
            # QA evaluation callback
            qa_callback = QAEvaluationCallback(eval_dataset, tokenizer, config)
            trainer.add_callback(qa_callback)
        
        # Train
        print("\n" + "="*70)
        print("STARTING TRAINING")
        print("="*70 + "\n")
        
        trainer_stats = trainer.train()
        
        # Log training stats
        mlflow.log_metrics({
            "train_runtime": trainer_stats.metrics['train_runtime'],
            "train_samples_per_second": trainer_stats.metrics.get('train_samples_per_second', 0),
            "train_loss": trainer_stats.metrics.get('train_loss', 0),
        })
        
        # Print training completion stats
        print("\n" + "="*70)
        print("TRAINING COMPLETED")
        print("="*70)
        print(f"Runtime: {trainer_stats.metrics['train_runtime']:.2f} seconds "
              f"({trainer_stats.metrics['train_runtime']/60:.2f} minutes)")
        
        if torch.cuda.is_available():
            used_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
            memory_for_training = round(used_memory - start_memory, 3)
            print(f"Peak memory: {used_memory} GB")
            print(f"Memory for training: {memory_for_training} GB")
            
            mlflow.log_metrics({
                "peak_memory_gb": used_memory,
                "training_memory_gb": memory_for_training,
            })
        
        # Save final model
        final_model_path = Path(config.output_dir) / "final_model"
        print(f"\nSaving final model to: {final_model_path}")
        trainer.save_model(str(final_model_path))
        
        print(f"\n✓ Training completed successfully!")
        print(f"✓ MLflow experiment: {config.mlflow_experiment_name}")
        print(f"✓ View results: mlflow ui --backend-store-uri {config.mlflow_tracking_uri}")

        # try:
        #     print("\nLogging model to MLflow...")
        #     transformers.log_model(
        #         transformers_model=trainer.model,  
        #         artifact_path="finetuned_model",
        #         task="text-generation", 
        #         registered_model_name=config.mlflow_registered_model_name
        #                             if hasattr(config, "mlflow_registered_model_name")
        #                             else None,
        #     )

        #     print("Model logged successfully.")

        # except Exception as e:
        #     print(f"Failed to log model: {e}")


        # if hasattr(config, "mlflow_registered_model_name") and config.mlflow_registered_model_name:
        #     try:
        #         run_id = mlflow.active_run().info.run_id
        #         model_uri = f"runs:/{run_id}/model"
        #         registered_model = mlflow.register_model(
        #             model_uri=model_uri,
        #             name=config.mlflow_registered_model_name
        #         )
        #         print(f"✓ Model registered as version {registered_model.version}")
        #     except Exception as e:
        #         print(f"Failed to register model version: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End LLM Fine-tuning Pipeline")

    for f in fields(TrainingConfig):
        arg_name = f"--{f.name}"
        arg_kwargs = {}
        if f.type == List[str]:
            arg_kwargs["type"] = str
            arg_kwargs["nargs"] = "+"

        elif f.type == bool:
            arg_kwargs["type"] = lambda x: str(x).lower() in ["true", "1", "yes"]
        else:
            arg_kwargs["type"] = f.type if f.type not in [Optional[str], Optional[int], Optional[float]] else str

        if f.default is not None and f.default != MISSING:
            arg_kwargs["default"] = f.default

        parser.add_argument(arg_name, **arg_kwargs)


    args = parser.parse_args()
    cli_config = TrainingConfig(**vars(args))
    train_model(cli_config)
