import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import json
import os

class ResultAnalyzer:
    """Analyze and visualize evaluation results"""
    
    @staticmethod
    def analyze_classification(metrics, output_dir=None):
        """
        Analyze classification metrics and generate visualizations
        
        Args:
            metrics: Classification metrics dictionary
            output_dir: Directory to save visualizations
            
        Returns:
            dict: Analysis summary
        """
        # Print basic metrics
        print("\nClassification Results:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        
        # Print per-class metrics
        print("\nPer-class Metrics:")
        for cls, scores in metrics["class_report"].items():
            print(f"{cls}:")
            print(f"  Precision: {scores['precision']:.4f}")
            print(f"  Recall: {scores['recall']:.4f}")
            print(f"  F1: {scores['f1']:.4f}")
            print(f"  Support: {scores['support']}")
        
        # Confusion matrix visualization
        cm = np.array(metrics["confusion_matrix"]["matrix"])
        labels = metrics["confusion_matrix"]["labels"]
        
        plt.figure(figsize=(10, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap=plt.cm.Blues, values_format="d")
        plt.title("Confusion Matrix")
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            cm_path = os.path.join(output_dir, "confusion_matrix.png")
            plt.savefig(cm_path, bbox_inches="tight")
            print(f"Confusion matrix saved to {cm_path}")
        
        plt.show()
        
        return {
            "summary": {
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"]
            },
            "class_report": metrics["class_report"]
        }
    
    @staticmethod
    def analyze_generation(metrics, output_dir=None):
        """
        Analyze generation metrics and generate visualizations
        
        Args:
            metrics: Generation metrics dictionary
            output_dir: Directory to save visualizations
            
        Returns:
            dict: Analysis summary
        """
        print("\nCode Generation Results:")
        print(f"Exact Match: {metrics['exact_match']:.4f}")
        print(f"BLEU-4: {metrics['bleu']:.4f}")
        
        print("\nROUGE Scores:")
        for key, value in metrics["rouge"].items():
            print(f"{key}: {value:.4f}")
        
        print("\nCodeBLEU Components:")
        if "codebleu" in metrics and isinstance(metrics["codebleu"], dict):
            codebleu = metrics["codebleu"]
            print(f"CodeBLEU: {codebleu.get('codebleu', 'N/A'):.4f}")
            print(f"  N-gram Match: {codebleu.get('ngram_match_score', 'N/A'):.4f}")
            print(f"  Weighted N-gram: {codebleu.get('weighted_ngram_match_score', 'N/A'):.4f}")
            print(f"  Syntax Match: {codebleu.get('syntax_match_score', 'N/A'):.4f}")
            print(f"  Dataflow Match: {codebleu.get('dataflow_match_score', 'N/A'):.4f}")
        else:
            print("CodeBLEU not computed")
        
        # Visualization: Metric comparison
        gen_metrics = {
            "Exact Match": metrics["exact_match"],
            "BLEU-4": metrics["bleu"],
            "ROUGE-1": metrics["rouge"]["rouge1"],
            "ROUGE-2": metrics["rouge"]["rouge2"],
            "ROUGE-L": metrics["rouge"]["rougeL"],
        }
        
        if "codebleu" in metrics and isinstance(metrics["codebleu"], dict):
            gen_metrics["CodeBLEU"] = metrics["codebleu"].get("codebleu", 0)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(gen_metrics.keys()), y=list(gen_metrics.values()), palette="viridis")
        plt.title("Code Generation Metrics")
        plt.ylabel("Score")
        plt.ylim(0, 1)
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            metrics_path = os.path.join(output_dir, "generation_metrics.png")
            plt.savefig(metrics_path, bbox_inches="tight")
            print(f"Metrics visualization saved to {metrics_path}")
        
        plt.show()
        
        return {
            "summary": gen_metrics
        }
    
    @staticmethod
    def analyze_failures(failures, error_types, output_dir=None, max_examples=5):
        """
        Analyze failure cases and generate report
        
        Args:
            failures: List of failure cases
            error_types: List of error type names
            output_dir: Directory to save report
            max_examples: Max examples to print
            
        Returns:
            dict: Failure analysis summary
        """
        if not failures:
            print("\nNo failure cases to analyze")
            return {}
        
        # Categorize failures
        failure_types = {
            "classification": [],
            "generation": [],
            "both": []
        }
        
        for failure in failures:
            if failure["type"] == "classification":
                failure_types["classification"].append(failure)
            elif failure["type"] == "generation":
                failure_types["generation"].append(failure)
            else:
                failure_types["both"].append(failure)
        
        # Print summary
        print("\nFailure Analysis:")
        print(f"Total Failures: {len(failures)}")
        print(f"Classification Errors: {len(failure_types['classification'])}")
        print(f"Generation Errors: {len(failure_types['generation'])}")
        print(f"Both Errors: {len(failure_types['both'])}")
        
        # Print examples
        print("\nSample Failure Cases:")
        for i, failure in enumerate(failures[:max_examples]):
            print(f"\nFailure #{i+1}: {failure['type'].upper()} ERROR")
            
            if failure["type"] == "classification":
                print(f"Buggy Code: {failure['buggy_code']}")
                print(f"True Type: {error_types[failure['true_label']]}")
                print(f"Predicted Type: {error_types[failure['pred_label']]}")
            
            elif failure["type"] == "generation":
                print(f"Buggy Code: {failure['buggy_code']}")
                print(f"Reference Fixed: {failure['reference_fixed']}")
                print(f"Predicted Fixed: {failure['predicted_fixed']}")
            
            else:  # Both
                print(f"Buggy Code: {failure['buggy_code']}")
                print(f"True Type: {error_types[failure['true_label']]}")
                print(f"Predicted Type: {error_types[failure['pred_label']]}")
                print(f"Reference Fixed: {failure['reference_fixed']}")
                print(f"Predicted Fixed: {failure['predicted_fixed']}")
        
        return {
            "counts": {
                "total": len(failures),
                "classification": len(failure_types["classification"]),
                "generation": len(failure_types["generation"]),
                "both": len(failure_types["both"])
            }
        }
    
    @staticmethod
    def full_analysis(results_path, output_dir, error_types):
        """
        Perform full analysis of evaluation results
        
        Args:
            results_path: Path to evaluation results JSON
            output_dir: Directory to save analysis
            error_types: List of error type names
        """
        # Load results
        with open(results_path, "r") as f:
            results = json.load(f)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Classification analysis
        cls_metrics = results["classification"]
        cls_analysis = ResultAnalyzer.analyze_classification(cls_metrics, output_dir)
        
        # Generation analysis
        gen_metrics = results["generation"]
        gen_analysis = ResultAnalyzer.analyze_generation(gen_metrics, output_dir)
        
        # Failure analysis
        failures = results.get("failures", [])
        failure_analysis = ResultAnalyzer.analyze_failures(
            failures, error_types, output_dir
        )
        
        # Save analysis summary
        analysis_summary = {
            "classification": cls_analysis,
            "generation": gen_analysis,
            "failures": failure_analysis
        }
        
        summary_path = os.path.join(output_dir, "analysis_summary.json")
        with open(summary_path, "w") as f:
            json.dump(analysis_summary, f, indent=4)
        
        print(f"Analysis summary saved to {summary_path}")
        return analysis_summary