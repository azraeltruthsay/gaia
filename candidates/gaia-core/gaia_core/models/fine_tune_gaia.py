import argparse
import json
import logging
import os
import sys
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

# Add the project root to the Python path to allow for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from gaia_core.config import Config, get_config

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("GAIA.FineTuner")

def run_fine_tuning(config: Config, dataset_path: str, base_model_name: str, new_model_name: str):
    """
    Fine-tunes a Hugging Face model using the provided dataset.

    Args:
        config: GAIA's configuration object.
        dataset_path: Path to the .jsonl training data file.
        base_model_name: The name of the base model from Hugging Face Hub.
        new_model_name: The name to save the new fine-tuned model under.
    """
    # --- 1. Load and Prepare the Dataset ---
    logger.info(f"Loading dataset from: {dataset_path}")
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset file not found at {dataset_path}")
        return

    # The 'datasets' library can load a .jsonl file directly
    try:
        dataset = load_dataset('json', data_files=dataset_path)
        logger.info("Dataset loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    # --- 2. Load Tokenizer and Model ---
    # Note: This uses a standard Hugging Face model, not a GGUF from the ModelPool,
    # as the Trainer API requires a transformers.PreTrainedModel.
    logger.info(f"Loading base model and tokenizer for '{base_model_name}'...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        model = AutoModelForCausalLM.from_pretrained(base_model_name)
        
        # Add a padding token if it doesn't exist.
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))

    except Exception as e:
        logger.error(f"Failed to load model or tokenizer '{base_model_name}': {e}")
        logger.error("Please ensure you have internet access to download the model and are logged in via `huggingface-cli login` if it's a private model.")
        return

    def tokenize_function(examples):
        # Assumes your JSONL has a "text" field.
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    logger.info("Tokenizing dataset...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # --- 3. Set Up Training ---
    output_dir = os.path.join(config.LORA_ADAPTERS_DIR, new_model_name)
    logger.info(f"Fine-tuned model will be saved to: {output_dir}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,  # Keep it short for a demo
        per_device_train_batch_size=1, # Keep it small to avoid memory issues
        learning_rate=5e-5,
        logging_dir=os.path.join(output_dir, 'logs'),
        logging_steps=10,
        save_steps=50,
        report_to="none" # Can be "tensorboard", "wandb" etc.
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
    )

    # --- 4. Run Training ---
    logger.info("Starting fine-tuning process...")
    trainer.train()
    logger.info("Fine-tuning complete.")

    # --- 5. Save Final Model ---
    final_path = os.path.join(output_dir, "final")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info(f"Final model and tokenizer saved to {final_path}")
    logger.warning("IMPORTANT: The saved model is in Hugging Face format. It must be converted to GGUF format to be used with GAIA's ModelPool.")

def main():
    parser = argparse.ArgumentParser(description="GAIA Model Fine-Tuning Script")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the training dataset (.jsonl file).")
    parser.add_argument("--base-model", type=str, default="gpt2", help="Name of the Hugging Face base model to fine-tune (e.g., 'gpt2', 'meta-llama/Llama-2-7b-chat-hf').")
    parser.add_argument("--output-name", type=str, required=True, help="Name for the new fine-tuned model directory.")
    
    args = parser.parse_args()

    try:
        config = Config()
        run_fine_tuning(
            config=config,
            dataset_path=args.dataset,
            base_model_name=args.base_model,
            new_model_name=args.output_name
        )
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    main()
