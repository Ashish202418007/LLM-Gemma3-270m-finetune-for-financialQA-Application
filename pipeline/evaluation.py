"""
End-to-End LLM Evaluation Pipeline
"""


import json
from typing import List, Dict
from pathlib import Path
import torch
from unsloth import FastLanguageModel
from tqdm import tqdm
import time
from datetime import datetime
import argparse
import mlflow
from functions import load_jsonl, f1_score, metric_max_over_ground_truths , compute_metrics, exact_match_score


class QAModelEvaluator:
    """Simple wrapper for evaluating any model on NewsQA with MLflow tracking."""
    
    def __init__(self, 
                 model_path: str,
                 device: str = "auto",
                 torch_dtype = torch.bfloat16,
                 max_new_tokens: int = 64,
                 temperature: float = 0.1,
                 use_mlflow: bool = True):
        """
        Initialize model for evaluation.
        
        Args:
            model_path: Path or HuggingFace model ID
            device: Device to use ("cuda", "cpu", or "auto")
            torch_dtype: Data type for model weights
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (lower = more deterministic)
            use_mlflow: Whether to use MLflow tracking
        """
        print(f"Loading model from: {model_path}")
        
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.use_mlflow = use_mlflow
        self.device = device
        self.system_prompt = "Answer the question based on the context below. Be concise and extract the answer directly from the context."
        

        # Load model
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=4096,
            dtype=torch_dtype,
            load_in_4bit=True,   
        )
        self.device = self.model.device
        self.model.eval()
        
        print(f"Model loaded on: {self.device}")
    

    def generate_answer(self, context: str, question: str) -> str:
            """Generate answer using chat template (Unsloth accelerated)."""

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
            ]

            # Build chat template
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=0.95,
                    top_k=64,
                    do_sample=self.temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                )

            gen_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
            answer = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            return answer.split("\n")[0].strip()

    
    def evaluate(self, 
                 data_path: str,
                 max_samples: int = None,
                 output_file: str = None,
                 experiment_name: str = "NewsQA-Evaluation",
                 run_name: str = None,
                 tags: Dict = None) -> Dict:
        """
        Evaluate model on NewsQA data with MLflow tracking.
        
        Args:
            data_path: Path to JSONL file
            max_samples: Maximum samples to evaluate (None for all)
            output_file: Path to save detailed results
            experiment_name: MLflow experiment name
            run_name: MLflow run name (auto-generated if None)
            tags: Additional tags for MLflow run
        
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"\nLoading data from: {data_path}")
        data = load_jsonl(data_path)
        
        if max_samples:
            data = data[:max_samples]
        
        print(f"Evaluating on {len(data)} samples...")
        if self.use_mlflow:
            mlflow.set_experiment(experiment_name)
            
            # Generate run name if not provided
            if run_name is None:
                model_name = self.model_path.split('/')[-1]
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                run_name = f"{model_name}_{timestamp}"
        
        # Start MLflow run
        if self.use_mlflow:
            with mlflow.start_run(run_name=run_name):
                return self._run_evaluation(data, data_path, output_file, tags)
        else:
            return self._run_evaluation(data, data_path, output_file, tags)
    
    def _run_evaluation(self, data: List[Dict], data_path: str, 
                       output_file: str = None, tags: Dict = None) -> Dict:
        """Internal method to run evaluation."""
        
        # Log parameters to MLflow
        if self.use_mlflow:
            mlflow.log_param("model_path", self.model_path)
            mlflow.log_param("data_path", data_path)
            mlflow.log_param("num_samples", len(data))
            mlflow.log_param("max_new_tokens", self.max_new_tokens)
            mlflow.log_param("temperature", self.temperature)
            mlflow.log_param("device", str(self.device))
            
            # Log additional tags
            if tags:
                mlflow.set_tags(tags)
            
            # Auto-tags
            mlflow.set_tag("dataset", "NewsQA")
            mlflow.set_tag("task", "Question Answering")
        
        predictions = []
        references = []
        detailed_results = []
        
        start_time = time.time()
        
        # Evaluation loop
        for example in tqdm(data, desc="Evaluating"):
            context = example['context']
            question = example['question']
            ground_truths = example['answers'] 
            

            pred = self.generate_answer(context, question)
            
            predictions.append(pred)
            references.append(ground_truths)
            
            detailed_results.append({
                'question': question,
                'prediction': pred,
                'ground_truths': ground_truths,
                'em': metric_max_over_ground_truths(exact_match_score, pred, ground_truths),
                'f1': metric_max_over_ground_truths(f1_score, pred, ground_truths)
            })
        
        elapsed_time = time.time() - start_time
        
        # Compute overall metrics
        metrics = compute_metrics(predictions, references)
        metrics['samples_per_second'] = len(data) / elapsed_time
        metrics['total_time_seconds'] = elapsed_time
        metrics['avg_time_per_sample'] = elapsed_time / len(data)
        
        # Log metrics 
        if self.use_mlflow:
            mlflow.log_metric("exact_match", metrics['exact_match'])
            mlflow.log_metric("f1_score", metrics['f1'])
            mlflow.log_metric("total_samples", metrics['total'])
            mlflow.log_metric("samples_per_second", metrics['samples_per_second'])
            mlflow.log_metric("total_time_seconds", metrics['total_time_seconds'])
            mlflow.log_metric("avg_time_per_sample", metrics['avg_time_per_sample'])
            
            # Log distribution of scores
            em_scores = [r['em'] for r in detailed_results]
            f1_scores = [r['f1'] for r in detailed_results]
            if len(em_scores) > 0:
                em_std = float(torch.tensor(em_scores, dtype=torch.float32).std())
            else:
                em_std = 0.0
            mlflow.log_metric("em_std", em_std)
            if len(f1_scores) > 0:
                f1_std = float(torch.tensor(f1_scores, dtype=torch.float32).std())
            else:
                f1_std = 0.0
            mlflow.log_metric("f1_std", f1_std)
        
        # Save detailed results
        if output_file:
            output_data = {
                'model_path': self.model_path,
                'data_path': data_path,
                'metrics': metrics,
                'config': {
                    'max_new_tokens': self.max_new_tokens,
                    'temperature': self.temperature
                },
                'predictions': detailed_results
            }
            
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"\nDetailed results saved to: {output_file}")
            
            # Log results file as artifact in MLflow
            if self.use_mlflow:
                mlflow.log_artifact(output_file)
        
        # Log sample predictions to MLflow
        if self.use_mlflow:
            sample_predictions_file = "sample_predictions.json"
            with open(sample_predictions_file, 'w', encoding='utf-8') as f:
                json.dump(detailed_results[:10], f, indent=2, ensure_ascii=False)
            mlflow.log_artifact(sample_predictions_file)
            Path(sample_predictions_file).unlink()  # Clean up temp file
        
        return metrics


# ============================================================================
# Main Evaluation Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate model on dataset with MLflow tracking')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model or HuggingFace model ID')
    parser.add_argument('--data_path', type=str, default='./data/newsQA_validation.jsonl',
                       help='Path to NewsQA JSONL file')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum samples to evaluate')
    parser.add_argument('--output_file', type=str, default='./results/evaluation_results.json',
                       help='Path to save results')
    parser.add_argument('--max_new_tokens', type=int, default=64,
                       help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.1,
                       help='Generation temperature')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda, cpu, or auto)')
    parser.add_argument('--no_mlflow', action='store_true',
                       help='Disable MLflow tracking')
    parser.add_argument('--experiment_name', type=str, default='NewsQA-Evaluation',
                       help='MLflow experiment name')
    parser.add_argument('--run_name', type=str, default=None,
                       help='MLflow run name (auto-generated if not provided)')
    parser.add_argument('--mlflow_tracking_uri', type=str, default=None,
                       help='MLflow tracking URI (default: local mlruns folder)')
    
    args = parser.parse_args()
    
    # Setup MLflow tracking URI if provided
    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        print(f"MLflow tracking URI: {args.mlflow_tracking_uri}")
    else:
        print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    
    # Initialize evaluator
    evaluator = QAModelEvaluator(
        model_path=args.model_path,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        use_mlflow=not args.no_mlflow
    )
    
    # Prepare tags
    tags = {
        "model_family": args.model_path.split('/')[0] if '/' in args.model_path else "local",
        "evaluation_script": "evaluate_newsqa.py"
    }
    
    # Run evaluation
    metrics = evaluator.evaluate(
        data_path=args.data_path,
        max_samples=args.max_samples,
        output_file=args.output_file,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        tags=tags
    )
    
    # Print results
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    print(f"Model: {args.model_path}")
    print(f"Data: {args.data_path}")
    print(f"Samples: {metrics['total']}")
    print(f"Exact Match (EM): {metrics['exact_match']:.2f}")
    print(f"F1 Score: {metrics['f1']:.2f}")
    print(f"Speed: {metrics['samples_per_second']:.2f} samples/sec")
    print(f"Avg Time/Sample: {metrics['avg_time_per_sample']:.3f} seconds")
    print(f"Total Time: {metrics['total_time_seconds']:.2f} seconds")
    print("="*60)
    
    if not args.no_mlflow:
        print(f"\n✓ Results logged to MLflow experiment: {args.experiment_name}")
        print(f"  View results: mlflow ui")


if __name__ == "__main__":
    main()