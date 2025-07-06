import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import nltk
import json
import os

# Download NLTK data (only needed once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class ClassificationMetrics:
    """Compute classification metrics for bug type prediction"""
    @staticmethod
    def compute(y_true, y_pred, labels):
        """
        Compute classification metrics
        
        Args:
            y_true: List of true labels
            y_pred: List of predicted labels
            labels: List of class names
            
        Returns:
            dict: Classification metrics
        """
        results = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
            "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
            "f1": f1_score(y_true, y_pred, average='weighted', zero_division=0),
            "confusion_matrix": ClassificationMetrics._confusion_matrix(y_true, y_pred, labels)
        }
        
        # Per-class metrics
        class_report = {}
        for i, label in enumerate(labels):
            class_report[label] = {
                "precision": precision_score(y_true, y_pred, labels=[i], average=None, zero_division=0)[0],
                "recall": recall_score(y_true, y_pred, labels=[i], average=None, zero_division=0)[0],
                "f1": f1_score(y_true, y_pred, labels=[i], average=None, zero_division=0)[0],
                "support": np.sum(np.array(y_true) == i)
            }
        results["class_report"] = class_report
        
        return results
    
    @staticmethod
    def _confusion_matrix(y_true, y_pred, labels):
        """Generate confusion matrix with labels"""
        cm = confusion_matrix(y_true, y_pred)
        return {
            "matrix": cm.tolist(),
            "labels": labels
        }

class GenerationMetrics:
    """Compute code generation metrics for bug fixing"""
    @staticmethod
    def compute(references, predictions, lang="java"):
        """
        Compute generation metrics
        
        Args:
            references: List of reference code strings
            predictions: List of predicted code strings
            lang: Programming language for CodeBLEU
            
        Returns:
            dict: Generation metrics
        """
        # Tokenize code for BLEU calculation
        ref_tokens = [nltk.word_tokenize(ref) for ref in references]
        pred_tokens = [nltk.word_tokenize(pred) for pred in predictions]
        
        # Compute metrics
        return {
            "exact_match": GenerationMetrics._exact_match(references, predictions),
            "bleu": GenerationMetrics._bleu_score(ref_tokens, pred_tokens),
            "rouge": GenerationMetrics._rouge_score(references, predictions),
            "codebleu": GenerationMetrics._codebleu_score(references, predictions, lang)
        }
    
    @staticmethod
    def _exact_match(references, predictions):
        """Compute exact match percentage"""
        return sum(1 for ref, pred in zip(references, predictions) if ref == pred) / len(references)
    
    @staticmethod
    def _bleu_score(ref_tokens, pred_tokens):
        """Compute BLEU-4 score"""
        # Prepare references in BLEU format
        refs = [[ref] for ref in ref_tokens]
        
        # Compute BLEU with smoothing
        smoothie = SmoothingFunction().method4
        return corpus_bleu(refs, pred_tokens, smoothing_function=smoothie)
    
    @staticmethod
    def _rouge_score(references, predictions):
        """Compute ROUGE scores"""
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = {"rouge1": 0, "rouge2": 0, "rougeL": 0}
        
        for ref, pred in zip(references, predictions):
            score = scorer.score(ref, pred)
            for key in scores:
                scores[key] += score[key].fmeasure
        
        # Average scores
        for key in scores:
            scores[key] /= len(references)
        
        return scores
    
    @staticmethod
    def _codebleu_score(references, predictions, lang):
        """Compute CodeBLEU score if available, else return None"""
        try:
            from codebleu import calc_codebleu
            return calc_codebleu(
                references=[[ref] for ref in references],
                predictions=predictions,
                lang=lang,
                weights=(0.25, 0.25, 0.25, 0.25),
            )
        except ImportError:
            return {"codebleu": None, "error": "codebleu package not installed"}

class ResultSaver:
    """Save evaluation results to files"""
    @staticmethod
    def save_results(results, output_dir):
        """Save evaluation results to JSON file"""
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "evaluation_results.json")
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        with open(output_path, "w") as f:
            json.dump(results, f, indent=4, default=convert_types)
        
        print(f"Results saved to {output_path}")
        return output_path
    
    @staticmethod
    def save_failures(failures, output_dir, max_examples=50):
        """Save failure cases to JSON file"""
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "failure_cases.json")
        
        with open(output_path, "w") as f:
            json.dump(failures[:max_examples], f, indent=4)
        
        print(f"Failure cases saved to {output_path}")
        return output_path