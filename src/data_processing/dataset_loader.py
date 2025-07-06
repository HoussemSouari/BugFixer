from datasets import load_dataset

import os
import json

class DatasetLoader:
    def __init__(self, cache_dir="./data"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def load_raw_dataset(self):
        """Load raw dataset from Hugging Face"""
        return load_dataset(
            "code_x_glue_cc_code_refinement", 
            "java",
            cache_dir=os.path.join(self.cache_dir, "raw")
        )
    
    def save_processed(self, dataset, output_dir):
        """Save processed dataset to disk"""
        os.makedirs(output_dir, exist_ok=True)
        
        for split in ["train", "validation", "test"]:
            with open(os.path.join(output_dir, f"{split}.json"), "w") as f:
                json.dump(dataset[split], f, indent=2)
    
    def load_processed(self, input_dir):
        """Load processed dataset from disk"""
        dataset = {}
        
        for split in ["train", "validation", "test"]:
            with open(os.path.join(input_dir, f"{split}.json"), "r") as f:
                dataset[split] = json.load(f)
        
        return dataset