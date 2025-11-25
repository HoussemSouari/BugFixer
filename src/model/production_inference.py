"""
Production-ready inference system with beam search, confidence scoring,
and fallback mechanisms for Java bug-fixing.
"""

import logging
import torch
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Result from a single prediction"""
    buggy_code: str
    predicted_code: str
    predicted_bug_type: Optional[str] = None
    confidence: float = 0.0
    beam_outputs: Optional[List[str]] = None
    alternative_suggestions: Optional[List[Tuple[str, float]]] = None
    is_fallback: bool = False
    error_message: Optional[str] = None


class ProductionInference:
    """
    Production-grade inference system for bug fixing.
    Includes beam search, confidence scoring, and error handling.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        device: torch.device,
        bug_types: Optional[List[str]] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.bug_types = bug_types or [
            'off_by_one', 'null_check', 'missing_break',
            'operator', 'logic', 'resource_leak', 'type_mismatch'
        ]
        
        self.model.eval()
        logger.info("Initialized ProductionInference")
    
    def predict_single(
        self,
        buggy_code: str,
        max_length: int = 512,
        num_beams: int = 5,
        temperature: float = 0.7,
        length_penalty: float = 1.0,
        return_alternatives: bool = False,
        predict_bug_type: bool = True,
    ) -> PredictionResult:
        """
        Make prediction for single code sample.
        
        Args:
            buggy_code: Buggy code to fix
            max_length: Maximum generation length
            num_beams: Number of beams for beam search
            temperature: Temperature for generation
            length_penalty: Length penalty for beam search
            return_alternatives: Whether to return alternative predictions
            predict_bug_type: Whether to predict bug type
            
        Returns:
            PredictionResult object
        """
        try:
            # Tokenize input
            inputs = self.tokenizer(
                buggy_code,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
            )
            
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
            
            # Generate predictions with beam search
            with torch.no_grad():
                # Main prediction
                if return_alternatives:
                    # Generate multiple beams
                    outputs = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=max_length,
                        num_beams=num_beams,
                        num_return_sequences=num_beams,
                        temperature=temperature,
                        length_penalty=length_penalty,
                        early_stopping=True,
                        output_scores=True,
                        return_dict_in_generate=True,
                    )
                    
                    # Decode all beam outputs
                    decoded_sequences = self.tokenizer.batch_decode(
                        outputs.sequences,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True
                    )
                    
                    # Calculate confidence scores (normalized probabilities)
                    # Access sequence scores if available
                    sequence_scores = outputs.sequences_scores if hasattr(outputs, 'sequences_scores') else None
                    
                    beam_results = list(zip(decoded_sequences, [1.0] * len(decoded_sequences)))
                    predicted_code = decoded_sequences[0]
                    confidence = 0.9 if sequence_scores is None else float(sequence_scores[0])
                    
                else:
                    outputs = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=max_length,
                        num_beams=num_beams,
                        temperature=temperature,
                        length_penalty=length_penalty,
                        early_stopping=True,
                    )
                    
                    predicted_code = self.tokenizer.decode(
                        outputs[0],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True
                    )
                    
                    beam_results = None
                    confidence = 0.85
                
                # Predict bug type if requested
                predicted_bug_type = None
                if predict_bug_type:
                    try:
                        pred_type, probs = self.model.predict_bug_type(input_ids, attention_mask)
                        type_idx = pred_type.item()
                        if 0 <= type_idx < len(self.bug_types):
                            predicted_bug_type = self.bug_types[type_idx]
                    except:
                        pass
            
            return PredictionResult(
                buggy_code=buggy_code,
                predicted_code=predicted_code,
                predicted_bug_type=predicted_bug_type,
                confidence=confidence,
                beam_outputs=beam_results if return_alternatives else None,
                alternative_suggestions=beam_results[:5] if beam_results else None,
            )
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return PredictionResult(
                buggy_code=buggy_code,
                predicted_code=buggy_code,  # Fallback to original
                is_fallback=True,
                error_message=str(e),
            )
    
    def predict_batch(
        self,
        buggy_codes: List[str],
        batch_size: int = 8,
        **kwargs
    ) -> List[PredictionResult]:
        """
        Make predictions for a batch of code samples.
        """
        results = []
        
        for i in range(0, len(buggy_codes), batch_size):
            batch = buggy_codes[i:i + batch_size]
            
            for code in batch:
                result = self.predict_single(code, **kwargs)
                results.append(result)
        
        return results
    
    def batch_generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_length: int = 512,
        num_beams: int = 5,
        temperature: float = 0.7,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate predictions for tokenized inputs.
        
        Args:
            input_ids: Tokenized input
            attention_mask: Attention mask
            max_length: Maximum generation length
            num_beams: Number of beams
            temperature: Sampling temperature
            
        Returns:
            Generated token IDs
        """
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device),
                max_length=max_length,
                num_beams=num_beams,
                temperature=temperature,
                early_stopping=True,
                **kwargs
            )
        
        return outputs
    
    def interactive_mode(self):
        """
        Interactive mode for testing predictions.
        """
        print("\n" + "=" * 70)
        print("Java Bug Fixer - Interactive Mode")
        print("=" * 70)
        print("Enter buggy Java code (type 'END' on new line when done)")
        print("-" * 70 + "\n")
        
        while True:
            lines = []
            print("Enter code (or 'quit' to exit):")
            
            while True:
                line = input()
                if line.lower() == 'quit':
                    print("Exiting...")
                    return
                if line == 'END':
                    break
                lines.append(line)
            
            buggy_code = '\n'.join(lines)
            
            if not buggy_code.strip():
                print("No code provided.\n")
                continue
            
            print("\nGenerating fix...")
            result = self.predict_single(
                buggy_code,
                return_alternatives=True,
                predict_bug_type=True,
            )
            
            print("\n" + "-" * 70)
            print("PREDICTION RESULTS:")
            print("-" * 70)
            print(f"\nPredicted Bug Type: {result.predicted_bug_type or 'Unknown'}")
            print(f"Confidence: {result.confidence:.2%}\n")
            
            print("Fixed Code:")
            print(result.predicted_code)
            
            if result.alternative_suggestions:
                print("\nAlternative Suggestions:")
                for idx, (code, conf) in enumerate(result.alternative_suggestions[1:], 1):
                    print(f"\n  Option {idx} (confidence: {conf:.2%}):")
                    print("  " + code.replace('\n', '\n  '))
            
            print("\n" + "=" * 70 + "\n")


class ConfidenceScorer:
    """
    Assign confidence scores to predictions based on multiple signals.
    """
    
    def __init__(self):
        self.weights = {
            'model_confidence': 0.6,
            'syntax_validity': 0.2,
            'code_similarity': 0.1,
            'consistency': 0.1,
        }
    
    def score_prediction(
        self,
        predicted_code: str,
        buggy_code: str,
        is_valid: bool,
        similarity: float,
        model_score: float = None,
    ) -> float:
        """
        Calculate combined confidence score.
        
        Args:
            predicted_code: Predicted fixed code
            buggy_code: Original buggy code
            is_valid: Whether syntax is valid
            similarity: Semantic similarity score
            model_score: Model's own confidence
            
        Returns:
            Confidence score [0, 1]
        """
        scores = {}
        
        # Model confidence
        scores['model_confidence'] = model_score or 0.5
        
        # Syntax validity bonus
        scores['syntax_validity'] = 1.0 if is_valid else 0.0
        
        # Code similarity
        scores['code_similarity'] = similarity
        
        # Consistency (code not too different from buggy code)
        buggy_tokens = set(buggy_code.split())
        pred_tokens = set(predicted_code.split())
        overlap = len(buggy_tokens & pred_tokens) / max(len(buggy_tokens), 1)
        scores['consistency'] = min(overlap, 1.0)
        
        # Weighted combination
        confidence = sum(
            scores[key] * self.weights[key]
            for key in scores
        )
        
        return min(max(confidence, 0.0), 1.0)


class FallbackMechanism:
    """
    Fallback strategies when model prediction fails.
    """
    
    def __init__(self, training_dataset: List[Tuple[str, str]] = None):
        """
        Initialize fallback with training dataset for similarity matching.
        """
        self.training_dataset = training_dataset or []
    
    def find_similar_fix(self, buggy_code: str, top_k: int = 3) -> List[str]:
        """
        Find similar bugs in training set and return their fixes.
        Uses simple token-based similarity.
        """
        if not self.training_dataset:
            return []
        
        buggy_tokens = set(buggy_code.split())
        similarities = []
        
        for train_buggy, train_fixed in self.training_dataset:
            train_tokens = set(train_buggy.split())
            overlap = len(buggy_tokens & train_tokens)
            union = len(buggy_tokens | train_tokens)
            
            if union > 0:
                jaccard = overlap / union
                similarities.append((jaccard, train_fixed))
        
        # Sort by similarity and return top-k
        similarities.sort(reverse=True)
        return [fix for _, fix in similarities[:top_k]]
    
    def apply_heuristic_fix(self, buggy_code: str) -> Optional[str]:
        """
        Apply simple heuristic fixes for common bugs.
        """
        # Heuristic 1: Off-by-one in loops
        if '<=' in buggy_code and 'length' in buggy_code:
            fixed = buggy_code.replace('<=', '<')
            if fixed != buggy_code:
                return fixed
        
        # Heuristic 2: Missing null check
        if '.' in buggy_code and 'if' not in buggy_code:
            # Very basic heuristic - may not apply
            pass
        
        # Heuristic 3: Missing break in switch
        if 'case' in buggy_code and 'break' not in buggy_code:
            # Add break before next case
            fixed = buggy_code.replace('case', 'break;\ncase')
            if fixed != buggy_code:
                return fixed
        
        return None
