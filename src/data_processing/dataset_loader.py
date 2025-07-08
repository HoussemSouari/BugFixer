from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
import logging
import os

class DatasetLoader:
    """Class to handle loading and saving datasets"""
    
    def __init__(self, dataset_path: str, tokenizer_name: str = "Salesforce/codet5-base"):
        self.dataset_path = dataset_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.logger = logging.getLogger(__name__)
    
    def load_raw_dataset(self) -> DatasetDict:
        """Load raw dataset from the specified path"""
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset path {self.dataset_path} does not exist.")
        
        self.logger.info(f"Loading dataset from {self.dataset_path}")
        return load_dataset("json", data_files=self.dataset_path)
    
    def save_dataset(self, dataset: DatasetDict, output_dir: str):
        """Save the processed dataset to the specified directory"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        self.logger.info(f"Saving dataset to {output_dir}")
        dataset.save_to_disk(output_dir)

    def tokenize_dataset(self, dataset: DatasetDict, max_length: int = 256) -> DatasetDict:
        """Tokenize the dataset using the loaded tokenizer"""
        def tokenize_function(examples):

            """Tokenization function for the dataset"""
            inputs = self.tokenizer(
                examples["buggy"],
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            targets = self.tokenizer(
                examples["fixed"],
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            return {
                "input_ids": inputs["input_ids"].squeeze().tolist(),
                "attention_mask": inputs["attention_mask"].squeeze().tolist(),
                "labels": targets["input_ids"].squeeze().tolist(),
            }
        
        self.logger.info("Tokenizing dataset")
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        return tokenized_dataset