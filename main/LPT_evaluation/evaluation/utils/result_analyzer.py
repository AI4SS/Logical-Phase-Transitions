"""
Result Analysis and Visualization Tools

Provides comprehensive analysis of FOLIO evaluation results including:
- Performance analysis by complexity
- Error analysis and patterns
- Comparison between different models/configurations
- Visualization of results
"""

import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
from dataclasses import dataclass

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib/seaborn not available. Plotting functions will be disabled.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelComparison:
    """Comparison between different models"""
    model_names: List[str]
    accuracies: List[float]
    accuracy_by_label: Dict[str, List[float]]
    accuracy_by_complexity: Dict[str, List[float]]
    response_times: List[float]


class ResultAnalyzer:
    """Comprehensive analysis of evaluation results"""
    
    def __init__(self, results_dir: str = "results/evaluation"):
        self.results_dir = Path(results_dir)
        if not self.results_dir.exists():
            raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    def load_results(self, results_file: str) -> Tuple[List[Dict], Dict]:
        """Load results and summary from JSON files, 支持 results_modified.json"""
        results_path = self.results_dir / results_file
        # Load detailed results
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)

        # Load corresponding summary - handle both old and new format
        if 'results_modified.json' in results_file:
            summary_file = results_file.replace('results_modified.json', 'summary_modified.json')
            summary_path = self.results_dir / summary_file
            if not summary_path.exists():
                # fallback: 尝试 summary.json
                summary_file = results_file.replace('results_modified.json', 'summary.json')
                summary_path = self.results_dir / summary_file
        elif 'results.json' in results_file:
            summary_file = results_file.replace('results.json', 'summary.json')
            summary_path = self.results_dir / summary_file
        else:
            summary_file = results_file.replace('_results.json', '_summary.json')
            summary_path = self.results_dir / summary_file

        summary = {}
        if summary_path.exists():
            with open(summary_path, 'r', encoding='utf-8') as f:
                summary = json.load(f)

        return results, summary
    
    def analyze_complexity_performance(self, results: List[Dict], num_bins: int = 5) -> Dict[str, Any]:
        """Analyze performance across different complexity levels"""

        # Filter results with complexity scores
        complexity_results = [r for r in results if r.get('complexity_score') is not None]

        if not complexity_results:
            return {"error": "No complexity data available"}

        # Create complexity bins
        complexity_scores = [r['complexity_score'] for r in complexity_results]
        min_complexity = min(complexity_scores)
        max_complexity = max(complexity_scores)

        # Define bins
        bins = np.linspace(min_complexity, max_complexity, num_bins + 1)
        bin_labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins)-1)]

        analysis = {
            "num_bins": num_bins,
            "complexity_range": {"min": min_complexity, "max": max_complexity},
            "bins": bin_labels,
            "bin_edges": bins.tolist(),
            "performance_by_bin": {},
            "parse_error_by_bin": {},
            "correlation": {}
        }

        # Analyze performance in each bin
        for i, label in enumerate(bin_labels):
            bin_results = [r for r in complexity_results
                          if bins[i] <= r['complexity_score'] < bins[i+1]]

            total_in_bin = len(bin_results)

            if total_in_bin:
                correct = sum(1 for r in bin_results if r.get('is_correct', False))
                accuracy = correct / total_in_bin
                avg_complexity = np.mean([r['complexity_score'] for r in bin_results])

                parse_errors = sum(1 for r in bin_results if r.get('error'))
                parse_error_rate = parse_errors / total_in_bin if total_in_bin else 0.0

                analysis["performance_by_bin"][label] = {
                    "count": total_in_bin,
                    "accuracy": accuracy,
                    "avg_complexity": avg_complexity
                }

                analysis["parse_error_by_bin"][label] = {
                    "total": total_in_bin,
                    "parse_errors": parse_errors,
                    "error_rate": parse_error_rate
                }
            else:
                analysis["parse_error_by_bin"][label] = {
                    "total": 0,
                    "parse_errors": 0,
                    "error_rate": 0.0
                }

        # Calculate correlation between complexity and performance
        complexities = [r['complexity_score'] for r in complexity_results]
        accuracies = [1 if r.get('is_correct', False) else 0 for r in complexity_results]

        if len(complexities) > 1:
            correlation = np.corrcoef(complexities, accuracies)[0, 1]
            analysis["correlation"]["complexity_accuracy"] = correlation

        return analysis
    
    def analyze_error_patterns(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze common error patterns"""
        
        incorrect_results = [r for r in results if not r.get('is_correct', False)]
        error_results = [r for r in results if r.get('error')]
        
        analysis = {
            "total_errors": len(incorrect_results),
            "parse_errors": len(error_results),
            "prediction_errors": len(incorrect_results) - len(error_results),
            "error_by_label": {},
            "error_by_complexity": {},
            "common_error_types": {}
        }
        
        # Errors by ground truth label
        for label in ['True', 'False', 'Uncertain']:
            label_results = [r for r in results if r.get('ground_truth') == label]
            label_errors = [r for r in label_results if not r.get('is_correct', False)]
            
            if label_results:
                error_rate = len(label_errors) / len(label_results)
                analysis["error_by_label"][label] = {
                    "total": len(label_results),
                    "errors": len(label_errors),
                    "error_rate": error_rate
                }
        
        # Errors by complexity
        complexity_results = [r for r in incorrect_results if r.get('complexity_score') is not None]
        if complexity_results:
            complexities = [r['complexity_score'] for r in complexity_results]
            analysis["error_by_complexity"] = {
                "avg_complexity": np.mean(complexities),
                "median_complexity": np.median(complexities),
                "complexity_range": {"min": min(complexities), "max": max(complexities)}
            }
        
        # Common parse error types
        parse_error_types = {}
        for result in error_results:
            error_msg = result.get('error', '')
            if 'parse' in error_msg.lower():
                error_type = 'Parse Error'
            elif 'json' in error_msg.lower():
                error_type = 'JSON Error'
            elif 'timeout' in error_msg.lower():
                error_type = 'Timeout Error'
            else:
                error_type = 'Other Error'
            
            parse_error_types[error_type] = parse_error_types.get(error_type, 0) + 1
        
        analysis["common_error_types"] = parse_error_types
        
        return analysis
    
    def compare_models(self, result_files: List[str]) -> ModelComparison:
        """Compare performance across multiple models"""
        
        model_names = []
        accuracies = []
        accuracy_by_label = {"True": [], "False": [], "Uncertain": []}
        accuracy_by_complexity = {}
        response_times = []
        
        for file in result_files:
            try:
                results, summary = self.load_results(file)
                
                # Extract model name from filename
                model_name = file.replace('_results.json', '').split('_')[-2]  # Rough extraction
                model_names.append(model_name)
                
                # Overall accuracy
                accuracies.append(summary.get('accuracy', 0.0))
                
                # Accuracy by label
                for label in accuracy_by_label:
                    label_acc = summary.get('accuracy_by_label', {}).get(label, 0.0)
                    accuracy_by_label[label].append(label_acc)
                
                # Accuracy by complexity
                for complexity_level, acc in summary.get('accuracy_by_complexity', {}).items():
                    if complexity_level not in accuracy_by_complexity:
                        accuracy_by_complexity[complexity_level] = []
                    accuracy_by_complexity[complexity_level].append(acc)
                
                # Response time
                response_times.append(summary.get('avg_response_time', 0.0))
                
            except Exception as e:
                logger.warning(f"Failed to load results from {file}: {e}")
        
        return ModelComparison(
            model_names=model_names,
            accuracies=accuracies,
            accuracy_by_label=accuracy_by_label,
            accuracy_by_complexity=accuracy_by_complexity,
            response_times=response_times
        )
    
    def generate_detailed_report(self, results_file: str, output_file: Optional[str] = None) -> str:
        """Generate comprehensive analysis report"""

        results, summary = self.load_results(results_file)

        # Perform analyses with different bin configurations
        complexity_analysis_5bins = self.analyze_complexity_performance(results, num_bins=5)
        complexity_analysis_10bins = self.analyze_complexity_performance(results, num_bins=10)
        error_analysis = self.analyze_error_patterns(results)

        # Generate report
        report_lines = []
        report_lines.append("FOLIO EVALUATION DETAILED ANALYSIS REPORT")
        report_lines.append("=" * 60)
        report_lines.append("")

        # Basic summary
        report_lines.append("BASIC SUMMARY")
        report_lines.append("-" * 20)
        report_lines.append(f"Total Examples: {summary.get('total_examples', 0)}")
        report_lines.append(f"Overall Accuracy: {summary.get('accuracy', 0):.3f}")
        report_lines.append(f"Average Response Time: {summary.get('avg_response_time', 0):.2f}s")
        report_lines.append("")

        # Performance by label
        report_lines.append("PERFORMANCE BY LABEL")
        report_lines.append("-" * 25)
        for label, acc in summary.get('accuracy_by_label', {}).items():
            report_lines.append(f"{label:10}: {acc:.3f}")
        report_lines.append("")

        # Complexity analysis with 5 bins
        if complexity_analysis_5bins.get("performance_by_bin"):
            report_lines.append("PERFORMANCE BY COMPLEXITY (5 BINS)")
            report_lines.append("-" * 35)
            for bin_label, data in complexity_analysis_5bins["performance_by_bin"].items():
                report_lines.append(f"{bin_label:15}: {data['accuracy']:.3f}")

            parse_error_bins = complexity_analysis_5bins.get("parse_error_by_bin", {})
            if parse_error_bins and any(info.get("parse_errors", 0) for info in parse_error_bins.values()):
                report_lines.append("")
                report_lines.append("PARSE ERROR PERFORMANCE BY COMPLEXITY (5 BINS)")
                report_lines.append("-" * 45)
                for bin_label in complexity_analysis_5bins.get("bins", []):
                    data = parse_error_bins.get(bin_label)
                    if not data:
                        continue
                    report_lines.append(
                        f"{bin_label:15}: {data['error_rate']:.3f} ({data['parse_errors']}/{data['total']})"
                    )

            if "complexity_accuracy" in complexity_analysis_5bins.get("correlation", {}):
                corr = complexity_analysis_5bins["correlation"]["complexity_accuracy"]
                report_lines.append(f"\nComplexity-Accuracy Correlation: {corr:.3f}")
            report_lines.append("")

        # Complexity analysis with 10 bins
        if complexity_analysis_10bins.get("performance_by_bin"):
            report_lines.append("PERFORMANCE BY COMPLEXITY (10 BINS)")
            report_lines.append("-" * 36)
            for bin_label, data in complexity_analysis_10bins["performance_by_bin"].items():
                report_lines.append(f"{bin_label:15}: {data['accuracy']:.3f}")

            parse_error_bins = complexity_analysis_10bins.get("parse_error_by_bin", {})
            if parse_error_bins and any(info.get("parse_errors", 0) for info in parse_error_bins.values()):
                report_lines.append("")
                report_lines.append("PARSE ERROR PERFORMANCE BY COMPLEXITY (10 BINS)")
                report_lines.append("-" * 46)
                for bin_label in complexity_analysis_10bins.get("bins", []):
                    data = parse_error_bins.get(bin_label)
                    if not data:
                        continue
                    report_lines.append(
                        f"{bin_label:15}: {data['error_rate']:.3f} ({data['parse_errors']}/{data['total']})"
                    )

            report_lines.append("")

        # Error analysis
        report_lines.append("ERROR ANALYSIS")
        report_lines.append("-" * 15)
        report_lines.append(f"Total Incorrect: {error_analysis['total_errors']}")
        report_lines.append(f"Parse Errors: {error_analysis['parse_errors']}")
        report_lines.append(f"Prediction Errors: {error_analysis['prediction_errors']}")

        if error_analysis.get("error_by_label"):
            report_lines.append("\nError Rates by Label:")
            for label, data in error_analysis["error_by_label"].items():
                report_lines.append(f"  {label}: {data['error_rate']:.3f} ({data['errors']}/{data['total']})")

        if error_analysis.get("common_error_types"):
            report_lines.append("\nCommon Error Types:")
            for error_type, count in error_analysis["common_error_types"].items():
                report_lines.append(f"  {error_type}: {count}")

        report_lines.append("")

        # Compile report
        report_text = "\n".join(report_lines)

        # Save to file if specified
        if output_file:
            output_path = self.results_dir / output_file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            logger.info(f"Detailed report saved to {output_path}")

        return report_text
    
    def plot_complexity_performance(self, results_file: str, save_path: Optional[str] = None):
        """Plot performance vs complexity"""
        if not PLOTTING_AVAILABLE:
            logger.warning("Plotting not available. Install matplotlib and seaborn.")
            return
        
        results, _ = self.load_results(results_file)
        complexity_analysis = self.analyze_complexity_performance(results)
        
        if not complexity_analysis.get("performance_by_bin"):
            logger.warning("No complexity data available for plotting")
            return
        
        # Prepare data
        bins = list(complexity_analysis["performance_by_bin"].keys())
        accuracies = [complexity_analysis["performance_by_bin"][bin]["accuracy"] for bin in bins]
        counts = [complexity_analysis["performance_by_bin"][bin]["count"] for bin in bins]
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Accuracy by complexity
        ax1.bar(bins, accuracies, alpha=0.7, color='skyblue')
        ax1.set_title('Accuracy by Complexity Level')
        ax1.set_xlabel('Complexity Range')
        ax1.set_ylabel('Accuracy')
        ax1.tick_params(axis='x', rotation=45)
        
        # Sample count by complexity
        ax2.bar(bins, counts, alpha=0.7, color='lightcoral')
        ax2.set_title('Sample Count by Complexity Level')
        ax2.set_xlabel('Complexity Range')
        ax2.set_ylabel('Number of Examples')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            save_path = self.results_dir / save_path
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Complexity performance plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_model_comparison(self, result_files: List[str], save_path: Optional[str] = None):
        """Plot comparison between multiple models"""
        if not PLOTTING_AVAILABLE:
            logger.warning("Plotting not available. Install matplotlib and seaborn.")
            return
        
        comparison = self.compare_models(result_files)
        
        # Create comparison plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # Overall accuracy comparison
        ax1.bar(comparison.model_names, comparison.accuracies, alpha=0.7, color='skyblue')
        ax1.set_title('Overall Accuracy Comparison')
        ax1.set_ylabel('Accuracy')
        ax1.tick_params(axis='x', rotation=45)
        
        # Accuracy by label
        x = np.arange(len(comparison.model_names))
        width = 0.25
        for i, (label, accs) in enumerate(comparison.accuracy_by_label.items()):
            ax2.bar(x + i*width, accs, width, label=label, alpha=0.7)
        
        ax2.set_title('Accuracy by Label')
        ax2.set_xlabel('Models')
        ax2.set_ylabel('Accuracy')
        ax2.set_xticks(x + width)
        ax2.set_xticklabels(comparison.model_names, rotation=45)
        ax2.legend()
        
        # Response time comparison
        ax3.bar(comparison.model_names, comparison.response_times, alpha=0.7, color='lightgreen')
        ax3.set_title('Response Time Comparison')
        ax3.set_ylabel('Average Response Time (s)')
        ax3.tick_params(axis='x', rotation=45)
        
        # Accuracy vs Response Time scatter
        ax4.scatter(comparison.response_times, comparison.accuracies, alpha=0.7, s=100)
        for i, name in enumerate(comparison.model_names):
            ax4.annotate(name, (comparison.response_times[i], comparison.accuracies[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        ax4.set_xlabel('Response Time (s)')
        ax4.set_ylabel('Accuracy')
        ax4.set_title('Accuracy vs Response Time')
        
        plt.tight_layout()
        
        if save_path:
            save_path = self.results_dir / save_path
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Model comparison plot saved to {save_path}")
        else:
            plt.show()
    
    def export_to_csv(self, results_file: str, output_file: str):
        """Export results to CSV for further analysis"""
        results, _ = self.load_results(results_file)
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Save to CSV
        output_path = self.results_dir / output_file
        df.to_csv(output_path, index=False)
        logger.info(f"Results exported to CSV: {output_path}")
        
        return df


if __name__ == "__main__":
    # Example usage
    analyzer = ResultAnalyzer("results/evaluation")
    
    # List available result files
    result_files = list(analyzer.results_dir.glob("*_results.json"))
    if result_files:
        print(f"Found {len(result_files)} result files:")
        for f in result_files:
            print(f"  - {f.name}")
        
        # Analyze the first file
        first_file = result_files[0].name
        print(f"\nAnalyzing {first_file}...")
        
        # Generate detailed report
        report = analyzer.generate_detailed_report(first_file)
        print(report)
        
        # Export to CSV
        analyzer.export_to_csv(first_file, first_file.replace('.json', '.csv'))
        
    else:
        print("No result files found in the directory.")
