# need !huggingface-cli login and llama.cpp

from unsloth import FastLanguageModel
import argparse
from huggingface_hub import create_repo

def pushtoHF(model_path_or_name:str, huggingface_username:str, model_name:str):

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_path_or_name,
        trust_remote_code=True
        )
    
  
    print("\nPushing merged 16-bit model to the Hub...")
    model.push_to_hub_merged(f"{huggingface_username}/{model_name}-merged-16bit", tokenizer, save_method="merged_16bit")
    print(f"Successfully pushed merged 16-bit model to {huggingface_username}/{model_name}-merged-16bit")


    print("\nPushing GGUF quantized model to the Hub...")
    model.push_to_hub_gguf(f"{huggingface_username}/{model_name}-gguf", tokenizer, quantization_method="q4_k_m")
    print(f"Successfully pushed GGUF model to {huggingface_username}/{model_name}-gguf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push Pushto model to Hugging Face Hub")
    parser.add_argument("model_path_or_name", type=str, help="Path or name of the Pushto model")
    parser.add_argument("huggingface_username", type=str, help="Hugging Face directory (username or organization)")
    parser.add_argument("model_name", type=str, help="Name for the model on Hugging Face Hub")
    parser.add_argument("--skip_merged", action="store_true")
    parser.add_argument("--skip_gguf", action="store_true")

    args = parser.parse_args()

    pushtoHF(args.model_path_or_name, args.huggingface_username, args.model_name)
    