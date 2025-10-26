# push_lora_argparse.py

import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import HfApi, login

def main(args):
    login(token=args.hf_token)
    model = AutoModelForCausalLM.from_pretrained(args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = PeftModel.from_pretrained(model, args.lora_checkpoint)


    model.save_pretrained_merged(
        args.local_save_path,
        tokenizer,
        save_method="merged_16bit"
    )
    print(f"Merged model saved locally at: {args.local_save_path}")

    model.push_to_hub_merged(
        args.hf_repo_name,
        tokenizer,
        save_method="merged_16bit",
        token=args.hf_token
    )
    print(f"Merged model pushed to Hugging Face at: https://huggingface.co/{args.hf_repo_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push Unsloth LoRA model to Hugging Face")

    parser.add_argument("--base_model", type=str, required=True,
                        help="Base model name (e.g., gemma3-270m)")
    parser.add_argument("--lora_checkpoint", type=str, required=True,
                        help="Path to Unsloth LoRA checkpoint folder")
    parser.add_argument("--local_save_path", type=str, default="merged_model_16bit",
                        help="Local folder to save merged model")
    parser.add_argument("--hf_repo_name", type=str, required=True,
                        help="Hugging Face repo name (username/model-name)")
    parser.add_argument("--hf_token", type=str, required=True,
                        help="Hugging Face API token")

    args = parser.parse_args()
    main(args)
