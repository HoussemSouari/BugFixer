"""
Comprehensive evaluation metrics for Java bug-fixing model.
Includes BLEU, CodeBLEU, exact match, syntax validity, and per-bug-type accuracy.
"""

import re
import json
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

import torch
import numpy as np
from tqdm import tqdm

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import single_meteor_score
    import nltk
    nltk.download('wordnet', quiet=True)
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False

try:
    from rouge_score import rouge_scorer
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False

try:
    from codebleu import calc_code_bleu
    HAS_CODEBLEU = True
except ImportError:
    HAS_CODEBLEU = False

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Single evaluation result for a code pair"""
    buggy_code: str
    fixed_code: str
    predicted_code: str
    bug_type: str
    
    exact_match: float
    bleu: float
    meteor: float
    rouge_l: float
    code_bleu: float
    
    syntax_valid: bool
    syntax_error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'exact_match': self.exact_match,
            'bleu': self.bleu,
            'meteor': self.meteor,
            'rouge_l': self.rouge_l,
            'code_bleu': self.code_bleu,
            'syntax_valid': self.syntax_valid,
            'bug_type': self.bug_type,
        }


class JavaSyntaxValidator:
    """Validate Java syntax"""
    
    @staticmethod
    def is_valid(code: str) -> Tuple[bool, Optional[str]]:
        """Basic syntax validation for Java"""
        if not code.strip():
            return False, "Empty code"
        
        # Count braces
        open_braces = code.count('{')
        close_braces = code.count('}')
        if open_braces != close_braces:
            return False, f"Mismatched braces: {open_braces} vs {close_braces}"
        
        # Count brackets
        open_brackets = code.count('[')
        close_brackets = code.count(']')
        if open_brackets != close_brackets:
            return False, f"Mismatched brackets: {open_brackets} vs {close_brackets}"
        
        # Count parentheses
        open_parens = code.count('(')
        close_parens = code.count(')')
        if open_parens != close_parens:
            return False, f"Mismatched parentheses: {open_parens} vs {close_parens}"
        
        return True, None


class BugFixEvaluator:
    """
    Comprehensive evaluator for bug-fixing model.
    """
    
    def __init__(self, tokenizer, bug_types: List[str] = None):
        self.tokenizer = tokenizer
        self.bug_types = bug_types or [
            'off_by_one', 'null_check', 'missing_break',
            'operator', 'logic', 'resource_leak', 'type_mismatch'
        ]
        
        # Initialize metrics
        self.rouge_scorer = None
        if HAS_ROUGE:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        
        logger.info(f"Initialized evaluator with {len(self.bug_types)} bug types")
    
    def compute_exact_match(self, prediction: str, reference: str) -> float:
        """Exact match score (0 or 1)"""
        # Normalize whitespace
        pred_clean = ' '.join(prediction.split())
        ref_clean = ' '.join(reference.split())
        
        return 1.0 if pred_clean == ref_clean else 0.0
    
    def compute_bleu(self, prediction: str, reference: str, max_n: int = 4) -> float:
        """Compute BLEU score"""
        if not HAS_NLTK:
            logger.warning("NLTK not available, skipping BLEU")
            return 0.0
        
        try:
            # Tokenize
            pred_tokens = prediction.split()
            ref_tokens = reference.split()
            
            if not pred_tokens or not ref_tokens:
                return 0.0
            
            # Calculate BLEU with smoothing
            smoothing_function = SmoothingFunction().method1
            weights = tuple([1.0 / max_n] * max_n)
            
            bleu_score = sentence_bleu(
                [ref_tokens],
                pred_tokens,
                weights=weights,
                smoothing_function=smoothing_function
            )
            
            return float(bleu_score)
        except Exception as e:
            logger.debug(f"BLEU calculation error: {e}")
            return 0.0
    
    def compute_meteor(self, prediction: str, reference: str) -> float:
        """Compute METEOR score"""
        if not HAS_NLTK:
            return 0.0
        
        try:
            pred_tokens = prediction.split()
            ref_tokens = reference.split()
            
            if not pred_tokens or not ref_tokens:
                return 0.0
            
            meteor_score = single_meteor_score(
                reference.split(),
                prediction.split()
            )
            
            return float(meteor_score)
        except Exception as e:
            logger.debug(f"METEOR calculation error: {e}")
            return 0.0
    
    def compute_rouge_l(self, prediction: str, reference: str) -> float:
        """Compute ROUGE-L score"""
        if not HAS_ROUGE:
            logger.warning("ROUGE not available, skipping ROUGE")
            return 0.0
        
        try:
            scores = self.rouge_scorer.score(reference, prediction)
            return float(scores['rougeL'].fmeasure)
        except Exception as e:
            logger.debug(f"ROUGE calculation error: {e}")
            return 0.0
    
    def compute_code_bleu_simplified(self, prediction: str, reference: str) -> float:
        """
        Simplified CodeBLEU that uses BLEU + syntactic similarity.
        Full CodeBLEU requires parse trees which is complex.
        """
        if not HAS_NLTK:
            return 0.0
        
        # Standard BLEU
        bleu = self.compute_bleu(prediction, reference)
        
        # Token-level match (syntactic similarity)
        pred_tokens = set(prediction.split())
        ref_tokens = set(reference.split())
        
        if len(ref_tokens) > 0:
            syntactic_match = len(pred_tokens & ref_tokens) / len(ref_tokens)
        else:
            syntactic_match = 0.0
        
        # Weighted combination
        code_bleu = 0.7 * bleu + 0.3 * syntactic_match
        return float(code_bleu)
    
    def evaluate_single(
        self,
        buggy_code: str,
        fixed_code: str,
        predicted_code: str,
        bug_type: str = 'unknown',
    ) -> EvaluationResult:
        """
        Evaluate a single prediction.
        """
        # Syntax validation
        syntax_valid, syntax_error = JavaSyntaxValidator.is_valid(predicted_code)
        
        # Normalize for comparison
        pred_normalized = ' '.join(predicted_code.split())
        fixed_normalized = ' '.join(fixed_code.split())
        
        # Compute metrics
        exact_match = self.compute_exact_match(pred_normalized, fixed_normalized)
        bleu = self.compute_bleu(pred_normalized, fixed_normalized)
        meteor = self.compute_meteor(pred_normalized, fixed_normalized)
        rouge_l = self.compute_rouge_l(pred_normalized, fixed_normalized)
        code_bleu = self.compute_code_bleu_simplified(pred_normalized, fixed_normalized)
        
        return EvaluationResult(
            buggy_code=buggy_code,
            fixed_code=fixed_code,
            predicted_code=predicted_code,
            bug_type=bug_type,
            exact_match=exact_match,
            bleu=bleu,
            meteor=meteor,
            rouge_l=rouge_l,
            code_bleu=code_bleu,
            syntax_valid=syntax_valid,
            syntax_error=syntax_error,
        )
    
    def evaluate_batch(
        self,
        buggy_codes: List[str],
        fixed_codes: List[str],
        predicted_codes: List[str],
        bug_types: List[str] = None,
    ) -> List[EvaluationResult]:
        """Evaluate a batch of predictions"""
        if bug_types is None:
            bug_types = ['unknown'] * len(buggy_codes)
        
        results = []
        for buggy, fixed, pred, bug_type in zip(
            buggy_codes, fixed_codes, predicted_codes, bug_types
        ):
            result = self.evaluate_single(buggy, fixed, pred, bug_type)
            results.append(result)
        
        return results
    
    def compute_aggregate_metrics(
        self,
        results: List[EvaluationResult],
    ) -> Dict[str, Any]:
        """Compute aggregate metrics across all results"""
        if not results:
            return {}
        
        metrics = {
            'num_samples': len(results),
            'exact_match': np.mean([r.exact_match for r in results]),
            'bleu': np.mean([r.bleu for r in results]),
            'meteor': np.mean([r.meteor for r in results]),
            'rouge_l': np.mean([r.rouge_l for r in results]),
            'code_bleu': np.mean([r.code_bleu for r in results]),
            'syntax_valid_rate': np.mean([float(r.syntax_valid) for r in results]),
        }
        
        # Per bug-type metrics
        metrics['per_bug_type'] = {}
        for bug_type in self.bug_types:
            type_results = [r for r in results if r.bug_type == bug_type]
            if type_results:
                metrics['per_bug_type'][bug_type] = {
                    'count': len(type_results),
                    'exact_match': np.mean([r.exact_match for r in type_results]),
                    'bleu': np.mean([r.bleu for r in type_results]),
                    'code_bleu': np.mean([r.code_bleu for r in type_results]),
                }
        
        return metrics
    
    def generate_report(self, metrics: Dict[str, Any]) -> str:
        """Generate formatted evaluation report"""
        report = "\n" + "=" * 70 + "\n"
        report += "                    EVALUATION REPORT\n"
        report += "=" * 70 + "\n"
        
        report += f"\nTotal Samples: {metrics.get('num_samples', 0)}\n"
        report += "-" * 70 + "\n"
        
        report += "\nOverall Metrics:\n"
        report += f"  Exact Match:      {metrics.get('exact_match', 0):.4f}\n"
        report += f"  BLEU-4:           {metrics.get('bleu', 0):.4f}\n"
        report += f"  CodeBLEU:         {metrics.get('code_bleu', 0):.4f}\n"
        report += f"  ROUGE-L:          {metrics.get('rouge_l', 0):.4f}\n"
        report += f"  METEOR:           {metrics.get('meteor', 0):.4f}\n"
        report += f"  Syntax Valid:     {metrics.get('syntax_valid_rate', 0):.4f}\n"
        
        # Per bug-type metrics
        if metrics.get('per_bug_type'):
            report += "\n" + "-" * 70 + "\n"
            report += "Per Bug-Type Metrics:\n"
            for bug_type, stats in metrics['per_bug_type'].items():
                report += f"\n  {bug_type.upper()}:\n"
                report += f"    Count:        {stats.get('count', 0)}\n"
                report += f"    Exact Match:  {stats.get('exact_match', 0):.4f}\n"
                report += f"    BLEU:         {stats.get('bleu', 0):.4f}\n"
                report += f"    CodeBLEU:     {stats.get('code_bleu', 0):.4f}\n"
        
        report += "\n" + "=" * 70 + "\n"
        return report
    
    def save_results(self, results: List[EvaluationResult], output_path: str):
        """Save detailed results to JSON"""
        data = [r.to_dict() for r in results]
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(results)} evaluation results to {output_path}")


import os
