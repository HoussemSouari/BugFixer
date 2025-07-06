import os
import torch
import logging
from torch.utils.data import Dataset, DataLoader
from transformers import TrainingArguments, Trainer
from .architecture import MultiTaskCodeT5
from datasets import load_from_disk
from src.utils.logger import setup_logger
from src.utils.config import load_config
import numpy as np

# Set up logger
setup_logger()
logger = logging.getLogger(__name__)

class JavaRefinementDataset(Dataset):
    """Custom dataset for Java bug fixing"""
    def __init__(self, dataset, split="train"):
        self.split = split
        self.data = dataset[split]
        logger.info(f"Loaded {len(self.data)} {split} examples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"]),
            "attention_mask": torch.tensor(item["attention_mask"]),
            "labels": torch.tensor(item["labels"]),
            "error_label": torch.tensor(item["error_label"])
        }

class MultiTaskTrainer(Trainer):
    """Custom trainer for multi-task learning"""
    def compute_loss(self, model, inputs, return_outputs=False):
        # Separate classification labels
        error_labels = inputs.pop("error_label")
        
        # Forward pass
        outputs = model(**inputs)
        
        # Calculate losses
        gen_loss = outputs["loss_gen"]
        class_loss = torch.nn.functional.cross_entropy(
            outputs["logits_class"], 
            error_labels
        )
        
        # Weighted loss (70% generation, 30% classification)
        loss = 0.7 * gen_loss + 0.3 * class_loss
        
        return (loss, outputs) if return_outputs else loss

def main():
    # Load configuration
    config = load_config("config/training.yaml")
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Load tokenized dataset
    logger.info(f"Loading dataset from {config.dataset_path}")
    tokenized_dataset = load_from_disk(config.dataset_path)
    
    # Create datasets
    train_dataset = JavaRefinementDataset(tokenized_dataset, "train")
    val_dataset = JavaRefinementDataset(tokenized_dataset, "validation")
    
    # Initialize model
    logger.info("Initializing model...")
    model = MultiTaskCodeT5(config.model_name)
    
    # Log model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        logging_dir=f"{config.output_dir}/logs",
        logging_steps=config.logging_steps,
        evaluation_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=3,
        fp16=config.fp16,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        report_to="none",  # Disable external services
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        prediction_loss_only=True,
        dataloader_num_workers=4,
        ddp_find_unused_parameters=False,
    )
    
    # Initialize trainer
    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    # Start training
    logger.info("Starting training...")
    train_result = trainer.train()
    
    # Save final model
    final_model_path = f"{config.output_dir}/final_model"
    trainer.save_model(final_model_path)
    logger.info(f"Saved final model to {final_model_path}")
    
    # Log training metrics
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    # Evaluate on validation set
    logger.info("Evaluating on validation set...")
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)
    
    # Save best model
    best_model_path = f"{config.output_dir}/best_model"
    trainer.save_model(best_model_path)
    logger.info(f"Saved best model to {best_model_path}")
    
    logger.info("Training complete!")

if __name__ == "__main__":
    main()