from tqdm import tqdm
from .java_preprocessor import JavaPreprocessor
from .bug_type_classifier import BugTypeClassifier

class DatasetFormatter:
    def __init__(self):
        self.preprocessor = JavaPreprocessor()
        self.classifier = BugTypeClassifier()
    
    def preprocess_dataset(self, dataset, sample_fraction=1.0):
        """
        Preprocess entire dataset:
        1. Clean code
        2. Classify bug types
        3. Filter invalid examples
        """
        processed = {
            "train": [],
            "validation": [],
            "test": []
        }
        
        for split in ["train", "validation", "test"]:
            split_data = dataset[split]
            num_samples = int(len(split_data) * sample_fraction)
            
            print(f"Processing {split} split ({num_samples} samples)...")
            for i in tqdm(range(num_samples)):
                example = split_data[i]
                try:
                    processed_example = self._process_example(example)
                    processed[split].append(processed_example)
                except Exception as e:
                    print(f"Skipping example {i}: {str(e)}")
        
        return processed
    
    def _process_example(self, example):
        """Process single example"""
        # Clean code
        buggy_clean = self.preprocessor.clean_code(example["buggy"])
        fixed_clean = self.preprocessor.clean_code(example["fixed"])
        
        # Classify bug type
        error_type = self.classifier.classify(buggy_clean, fixed_clean)
        
        return {
            "id": example["id"],
            "buggy": buggy_clean,
            "fixed": fixed_clean,
            "error_type": error_type
        }
    
