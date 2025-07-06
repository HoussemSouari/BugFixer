import torch
from transformers import AutoTokenizer
from ..data_processing.java_preprocessor import JavaPreprocessor
from .architecture import MultiTaskCodeT5
import logging

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class BugFixer:
    def __init__(self, model_path, device="cpu"):
        """
        Initialize the bug fixing model
        
        Args:
            model_path: Path to trained model weights
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.device = device
        self.preprocessor = JavaPreprocessor()
        self.tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base-java")
        self.error_types = ["Syntax", "Logical", "Runtime", "Other"]
        
        # Initialize model
        self.model = MultiTaskCodeT5()
        try:
            state_dict = torch.load(model_path, map_location=device)
            self.model.load_state_dict(state_dict)
            self.model.to(device)
            self.model.eval()
            logger.info(f"Model loaded successfully from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise RuntimeError(f"Failed to load model from {model_path}")
    
    def preprocess_code(self, code: str) -> str:
        """
        Preprocess Java code using the same pipeline as training
        
        Args:
            code: Raw Java code string
            
        Returns:
            Cleaned Java code string
        """
        try:
            return self.preprocessor.clean_code(code)
        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            return code  # Return original if preprocessing fails
    
    def predict(self, buggy_code: str) -> dict:
        """
        Predict bug type and fixed code
        
        Args:
            buggy_code: Java code with potential bug
            
        Returns:
            Dictionary with:
            - error_type: Predicted bug category
            - confidence: Classification confidence
            - fixed_code: Generated fixed code
            - refactoring_suggestions: List of suggestions
        """
        try:
            # Preprocess input code
            cleaned_code = self.preprocess_code(buggy_code)
            logger.debug(f"Preprocessed code: {cleaned_code}")
            
            # Tokenize input
            inputs = self.tokenizer(
                cleaned_code,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=256
            ).to(self.device)
            
            # Run model inference
            with torch.no_grad():
                # Forward pass for classification
                outputs = self.model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask
                )
                
                # Get classification results
                class_logits = outputs["logits_class"]
                class_probs = torch.softmax(class_logits, dim=-1)
                class_pred = torch.argmax(class_logits, dim=-1).item()
                confidence = class_probs[0][class_pred].item()
                
                # Generate fixed code
                generated_ids = self.model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_length=256,
                    num_beams=5,
                    early_stopping=True
                )
                fixed_code = self.tokenizer.decode(
                    generated_ids[0], 
                    skip_special_tokens=True
                )
                
                # Get refactoring suggestions
                suggestions = self.get_refactoring_suggestions(
                    self.error_types[class_pred]
                )
            
            return {
                "error_type": self.error_types[class_pred],
                "confidence": confidence,
                "fixed_code": fixed_code,
                "refactoring_suggestions": suggestions,
                "class_probs": class_probs[0].cpu().numpy().tolist()
            }
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return {
                "error": str(e),
                "error_type": "Unknown",
                "fixed_code": buggy_code,
                "refactoring_suggestions": ["Failed to process code"]
            }
    
    def get_refactoring_suggestions(self, error_type: str) -> list:
        """
        Generate context-aware refactoring suggestions
        
        Args:
            error_type: Predicted bug category
            
        Returns:
            List of refactoring suggestions
        """
        suggestions = {
            "Syntax": [
                "Check for missing semicolons at line ends",
                "Verify bracket balancing (curly and round)",
                "Ensure proper import statements are present",
                "Check variable declaration syntax",
                "Validate method signature correctness"
            ],
            "Logical": [
                "Review conditional statements and operators",
                "Check loop termination conditions",
                "Validate algorithm implementation logic",
                "Verify variable initialization before use",
                "Test edge cases and boundary conditions"
            ],
            "Runtime": [
                "Add null checks before method calls",
                "Implement try-catch blocks for exception handling",
                "Validate input parameters before processing",
                "Check resource closing in finally blocks",
                "Verify array index boundaries before access"
            ],
            "Other": [
                "Consider extracting repeated code into methods",
                "Evaluate potential performance optimizations",
                "Review for code smells and anti-patterns",
                "Simplify complex conditional expressions",
                "Add documentation comments for clarity"
            ]
        }
        return suggestions.get(error_type, [
            "Review code for potential improvements"
        ])
    
    def batch_predict(self, code_list: list) -> list:
        """
        Process multiple code snippets in a batch
        
        Args:
            code_list: List of Java code strings
            
        Returns:
            List of prediction results
        """
        results = []
        for code in code_list:
            results.append(self.predict(code))
        return results


if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Java Bug Fixer Inference")
    parser.add_argument("--model", type=str, required=True, 
                        help="Path to trained model")
    parser.add_argument("--input", type=str, 
                        help="Input Java file or code string")
    parser.add_argument("--output", type=str, default="output.json",
                        help="Output file for results")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to use (cpu/cuda)")
    args = parser.parse_args()
    
    # Initialize bug fixer
    fixer = BugFixer(args.model, device=args.device)
    
    # Handle input
    if args.input.endswith(".java"):
        with open(args.input, "r") as f:
            code = f.read()
        results = fixer.predict(code)
    else:
        results = fixer.predict(args.input)
    
    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {args.output}")
    print("\nPrediction Summary:")
    print(f"Error Type: {results['error_type']} ({results['confidence']:.2%} confidence)")
    print("\nFixed Code:")
    print(results['fixed_code'])
    print("\nSuggestions:")
    for i, suggestion in enumerate(results['refactoring_suggestions']):
        print(f"{i+1}. {suggestion}")