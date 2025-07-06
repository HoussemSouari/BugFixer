from transformers import AutoTokenizer
from tqdm import tqdm

class TokenizationUtils:
    def __init__(self, model_name="Salesforce/codet5-base-java", max_length=256):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        
        # Add Java-specific tokens
        self.tokenizer.add_tokens(["<generics>", "<annotation>"])
    
    def tokenize_dataset(self, dataset):
        """Tokenize entire dataset"""
        tokenized = {
            "train": [],
            "validation": [],
            "test": []
        }
        
        for split in ["train", "validation", "test"]:
            print(f"Tokenizing {split} split...")
            for example in tqdm(dataset[split]):
                try:
                    tokenized_example = self._tokenize_example(example)
                    tokenized[split].append(tokenized_example)
                except Exception as e:
                    print(f"Skipping tokenization: {str(e)}")
        
        return tokenized
    
    def _tokenize_example(self, example):
        """Tokenize single example"""
        # Tokenize inputs
        inputs = self.tokenizer(
            example["buggy"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize targets
        targets = self.tokenizer(
            example["fixed"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Convert error type to label
        error_types = ["SYNTAX", "LOGICAL", "RUNTIME", "OTHER"]
        error_label = error_types.index(example["error_type"])
        
        return {
            "input_ids": inputs["input_ids"].squeeze().tolist(),
            "attention_mask": inputs["attention_mask"].squeeze().tolist(),
            "labels": targets["input_ids"].squeeze().tolist(),
            "error_label": error_label
        }