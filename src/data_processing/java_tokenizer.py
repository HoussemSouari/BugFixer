from transformers import AutoTokenizer
import os 

class JavaTokenizer:
    def __init__(self, model_name="Salesforce/codet5-base", max_length=256):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        
        # Add Java-specific tokens
        self.tokenizer.add_tokens(["<generics>", "<annotation>"])
    
    def tokenize(self, code_snippet):
        """Tokenize a single Java code snippet"""
        return self.tokenizer(
            code_snippet,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
    
    def tokenize_dataset(self, dataset):
        """Tokenize an entire dataset"""
        tokenized = {
            "train": [],
            "validation": [],
            "test": []
        }
        
        for split in ["train", "validation", "test"]:
            print(f"Tokenizing {split} split...")
            for example in dataset[split]:
                try:
                    tokenized_example = self._tokenize_example(example)
                    tokenized[split].append(tokenized_example)
                except Exception as e:
                    print(f"Skipping tokenization: {str(e)}")
        
        return tokenized
    
    def decode(self, token_ids):
        """Decode token IDs back to Java code"""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
    
    def save_tokenizer(self, save_path):
        """Save the tokenizer to a specified path"""
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"Tokenizer saved to {save_path}")


    def load_tokenizer(self, load_path):
        """Load a tokenizer from a specified path"""
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Tokenizer path {load_path} does not exist.")
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        print(f"Tokenizer loaded from {load_path}")


