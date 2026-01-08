#!/usr/bin/env python3
"""
Logical Reasoning Evaluation Framework - Command Line Interface

Main entry point for running evaluations, analyzing results, and managing configurations.
Supports ProverQA and NSA-LR datasets.
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Optional, List
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from evaluation.evaluator import FOLIOEvaluator
from evaluation.config_manager import ConfigManager, FullConfig, get_preset_configs
from evaluation.utils.result_analyzer import ResultAnalyzer
from evaluation.models.llm_interface import LLMInterface


def setup_logging(level: str = "INFO", log_file: str | None = None):
    """Setup logging configuration with optional rotating file handler.

    Parameters:
        level: log level string (DEBUG/INFO/...)
        log_file: if provided, write logs to this file with rotation (5MB * 5 backups)
    """
    root = logging.getLogger()
    # Avoid duplicate handlers if re-invoked
    if not root.handlers:
        root.setLevel(getattr(logging, level.upper()))
    else:
        # Update level if already configured
        root.setLevel(getattr(logging, level.upper()))

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Ensure a console handler exists (stream to stderr)
    has_console = any(isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler) for h in root.handlers)
    if not has_console:
        ch = logging.StreamHandler()
        ch.setLevel(getattr(logging, level.upper()))
        ch.setFormatter(formatter)
        root.addHandler(ch)

    if log_file:
        try:
            from pathlib import Path
            log_path = Path(log_file)
            if log_path.parent and not log_path.parent.exists():
                log_path.parent.mkdir(parents=True, exist_ok=True)
            fh = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=5, encoding='utf-8')
            fh.setLevel(getattr(logging, level.upper()))
            fh.setFormatter(formatter)
            root.addHandler(fh)
            root.debug(f"File logging enabled -> {log_file}")
        except Exception as e:
            root.error(f"Failed to set up file logging at {log_file}: {e}")


def cmd_evaluate(args):
    """Run evaluation command"""
    print("🚀 Starting Logical Reasoning Evaluation...")
    
    config_manager = ConfigManager()
    
    # Load configuration
    if args.config:
        config = config_manager.load_config(args.config)
    elif args.preset:
        preset_configs = get_preset_configs()
        if args.preset not in preset_configs:
            print(f"❌ Unknown preset: {args.preset}")
            print(f"Available presets: {list(preset_configs.keys())}")
            return
        config = FullConfig.from_dict(preset_configs[args.preset])
    else:
        # Use default config if available
        available_configs = config_manager.list_configs()
        if available_configs:
            config = config_manager.load_config(available_configs[0])
            print(f"📝 Using default config: {available_configs[0]}")
        else:
            print("❌ No configuration found. Please specify --config or --preset")
            return
    
    # Override config with command line arguments
    if args.max_examples:
        config.evaluation.max_examples = args.max_examples
    if args.start_idx:
        config.evaluation.start_idx = args.start_idx
    if args.end_idx:
        config.evaluation.end_idx = args.end_idx
    if args.output_dir:
        config.evaluation.output_dir = args.output_dir
    if args.template:
        config.evaluation.prompt_template = args.template
    if args.temperature is not None:
        config.model.temperature = args.temperature
    if args.max_tokens:
        config.model.max_tokens = args.max_tokens
    
    # Validate configuration
    issues = config_manager.validate_config(config)
    if issues:
        print("❌ Configuration validation failed:")
        for issue in issues:
            print(f"   - {issue}")
        return
    
    print(f"🤖 Model: {config.model.provider} - {config.model.model_name}")
    print(f"📊 Data: {config.evaluation.data_path}")
    print(f"📝 Template: {config.evaluation.prompt_template}")
    if config.evaluation.max_examples:
        print(f"🔢 Max examples: {config.evaluation.max_examples}")
    
    # Create evaluator configuration
    evaluator_config = {
        'provider': config.model.provider,
        'model_name': config.model.model_name,
        'provider_config': config.model.provider_config or {
            'base_url': config.model.base_url,
            'api_key': config.model.api_key,
            'temperature': config.model.temperature,
            'max_tokens': config.model.max_tokens
        },
        'data_path': config.evaluation.data_path,
        'prompt_template': config.evaluation.prompt_template,
        'use_chat_format': config.evaluation.use_chat_format,
        'include_complexity': config.evaluation.include_complexity,
        'output_dir': config.evaluation.output_dir
    }
    
    try:
        # Initialize evaluator
        evaluator = FOLIOEvaluator(evaluator_config)
        
        # Run evaluation
        results, summary = evaluator.evaluate(
            max_examples=config.evaluation.max_examples,
            start_idx=config.evaluation.start_idx,
            end_idx=config.evaluation.end_idx,
            filter_by_complexity=config.evaluation.filter_by_complexity,
            filter_by_label=config.evaluation.filter_by_label
        )
        
        print(f"\n✅ Evaluation completed!")
        print(f"📈 Overall Accuracy: {summary.accuracy:.3f}")
        print(f"📊 Total Examples: {summary.total_examples}")
        print(f"⏱️  Average Response Time: {summary.avg_response_time:.2f}s")
        if summary.error_count > 0:
            print(f"⚠️  Errors: {summary.error_count}")
        
        print(f"💾 Results saved to: {config.evaluation.output_dir}")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()


def transform_complexity_scores(data):
    """Apply sqrt transformation to complexity_score in all data entries"""
    import math
    transformed_count = 0
    for item in data:
        if 'complexity_score' in item:
            original_score = item['complexity_score']
            item['complexity_score'] = math.sqrt(original_score)
            transformed_count += 1
    return transformed_count


def cmd_analyze(args):
    """Run analysis command"""
    print("📊 Analyzing Results...")
    from pathlib import Path
    import json
    import math

    analyzer = ResultAnalyzer(args.results_dir)

    if args.file:
        # Analyze specific file
        print(f"📁 Analyzing: {args.file}")
        file_path = Path(args.file)

        # Automatically apply sqrt transformation to complexity_score
        try:
            with open(file_path, 'r', encoding='utf-8', errors='surrogatepass') as f:
                data = json.load(f)

            # Apply transform_complexity
            if isinstance(data, list) and data and 'complexity_score' in data[0]:
                transformed = transform_complexity_scores(data)
                print(f"🔄 Applied sqrt transformation to {transformed} records")

                # Save transformed data
                with open(file_path, 'w', encoding='utf-8', errors='surrogatepass') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                print(f"💾 Transformed results saved to file")
        except Exception as e:
            print(f"⚠️ complexity_score transformation failed or skipped: {e}")

        try:
            # Handle results_modified.json filename
            if 'results_modified.json' in args.file:
                output_file = args.file.replace('results_modified.json', 'analysis_modified.txt')
                csv_file = args.file.replace('results_modified.json', 'results_modified.csv')
                plot_file = args.file.replace('results_modified.json', 'complexity_plot_modified.png')

                # Auto-correct predicted_answer to match ground_truth
                import json
                from pathlib import Path
                file_path = Path(args.file)
                with open(file_path, 'r', encoding='utf-8', errors='surrogatepass') as f:
                    data = json.load(f)

                # Matching rules
                def match(gt, pred):
                    gt_map = {'True': ['A', 'True', 'true'], 'False': ['B', 'False', 'false'], 'Uncertain': ['C', 'Uncertain', 'uncertain']}
                    pred_norm = str(pred).strip()
                    for k, v in gt_map.items():
                        if gt == k and (pred_norm in v):
                            return True
                    # Support direct True/False/Uncertain labels
                    if gt == pred_norm:
                        return True
                    return False


                # Calculate overall accuracy and performance by label
                total = len(data)
                correct_count = 0
                label_stats = {'True': {'total': 0, 'correct': 0}, 'False': {'total': 0, 'correct': 0}, 'Uncertain': {'total': 0, 'correct': 0}}
                for item in data:
                    gt = item.get('ground_truth', '').strip()
                    pred = item.get('predicted_answer', '').strip()
                    if match(gt, pred):
                        item['is_correct'] = True
                        item['error'] = None
                        correct_count += 1
                        label_stats[gt]['correct'] += 1
                    else:
                        item['is_correct'] = False
                    if gt in label_stats:
                        label_stats[gt]['total'] += 1

                accuracy = correct_count / total if total > 0 else 0.0
                accuracy_by_label = {k: (label_stats[k]['correct'] / label_stats[k]['total'] if label_stats[k]['total'] > 0 else 0.0) for k in label_stats}

                # Save corrected file (overwrite original)
                with open(file_path, 'w', encoding='utf-8', errors='surrogatepass') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                print(f"Auto-corrected is_correct field, accuracy: {accuracy:.3f}")

                # Update summary_modified.json file
                summary_file = args.file.replace('results_modified.json', 'summary_modified.json')
                import os
                summary_data = {
                    'total_examples': total,
                    'accuracy': accuracy,
                    'accuracy_by_label': accuracy_by_label
                }
                with open(summary_file, 'w', encoding='utf-8', errors='surrogatepass') as f:
                    json.dump(summary_data, f, ensure_ascii=False, indent=2)
                print(f"Updated summary_modified.json with latest accuracy and label-wise performance")

            elif 'results.json' in args.file:
                output_file = args.file.replace('results.json', 'analysis.txt')
                csv_file = args.file.replace('results.json', 'results.csv')
                plot_file = args.file.replace('results.json', 'complexity_plot.png')
            else:
                output_file = args.file.replace('_results.json', '_analysis.txt')
                csv_file = args.file.replace('_results.json', '_results.csv')
                plot_file = args.file.replace('_results.json', '_complexity_plot.png')


            # Original analysis report
            report = analyzer.generate_detailed_report(
                args.file,
                output_file=output_file
            )
            if not args.quiet:
                print(report)

            # Generate error-filtered analysis report
            import json
            from pathlib import Path

            file_path = Path(args.file)
            try:
                with open(file_path, 'r', encoding='utf-8', errors='surrogatepass') as f:
                    data = json.load(f)
            except json.JSONDecodeError as err:
                print(f"❌ Failed to parse result file, skipping error-filtered report: {err}")
                data = None

            filtered_data = None
            removed_count = 0

            if isinstance(data, list):
                valid_items = [item for item in data if isinstance(item, dict)]
                filtered_data = [
                    item for item in valid_items
                    if not item.get('error')
                ]
                removed_count = len(valid_items) - len(filtered_data)
            elif isinstance(data, dict):
                results_list = data.get('results')
                if isinstance(results_list, list):
                    valid_items = [item for item in results_list if isinstance(item, dict)]
                    filtered_results = [
                        item for item in valid_items
                        if not item.get('error')
                    ]
                    removed_count = len(valid_items) - len(filtered_results)
                    filtered_data = {**data, 'results': filtered_results}
                else:
                    filtered_data = data
            elif data is not None:
                print("⚠️ Unknown result file format, skipping error-filtered report.")

            if filtered_data is not None:
                if isinstance(filtered_data, list) and not filtered_data:
                    print("⚠️ All entries contain errors, not generating error-filtered report.")
                elif isinstance(filtered_data, dict) and not filtered_data.get('results', filtered_data):
                    print("⚠️ All entries contain errors, not generating error-filtered report.")
                else:
                    tmp_results_name = f"{file_path.stem}_woError_results.json"
                    tmp_summary_name = f"{file_path.stem}_woError_summary.json"
                    tmp_file_wo_error = file_path.parent / tmp_results_name
                    tmp_summary_file = file_path.parent / tmp_summary_name
                    try:
                        with open(tmp_file_wo_error, 'w', encoding='utf-8', errors='surrogatepass') as f:
                            json.dump(filtered_data, f, ensure_ascii=False, indent=2)

                        output_path = Path(output_file)
                        output_file_wo_error = str(
                            output_path.with_name(output_path.stem + 'woError' + output_path.suffix)
                        )

                        records_for_summary = (
                            filtered_data
                            if isinstance(filtered_data, list)
                            else filtered_data.get('results', [])
                        )

                        total_examples = len(records_for_summary)
                        correct = sum(
                            1 for item in records_for_summary
                            if isinstance(item, dict) and item.get('is_correct')
                        )
                        accuracy = (correct / total_examples) if total_examples else 0.0

                        labels = ['True', 'False', 'Uncertain']
                        accuracy_by_label = {}
                        for label in labels:
                            label_items = [
                                item for item in records_for_summary
                                if isinstance(item, dict) and item.get('ground_truth') == label
                            ]
                            if label_items:
                                label_correct = sum(1 for item in label_items if item.get('is_correct'))
                                accuracy_by_label[label] = label_correct / len(label_items)

                        response_times = [
                            item.get('response_time') for item in records_for_summary
                            if isinstance(item, dict) and isinstance(item.get('response_time'), (int, float))
                        ]
                        avg_response_time = (
                            sum(response_times) / len(response_times)
                            if response_times else 0.0
                        )

                        summary_payload = {
                            'total_examples': total_examples,
                            'accuracy': accuracy,
                            'accuracy_by_label': accuracy_by_label,
                            'avg_response_time': avg_response_time,
                            'error_count': total_examples - correct,
                        }

                        with open(tmp_summary_file, 'w', encoding='utf-8', errors='surrogatepass') as f:
                            json.dump(summary_payload, f, ensure_ascii=False, indent=2)

                        analyzer.generate_detailed_report(
                            str(tmp_file_wo_error),
                            output_file=output_file_wo_error
                        )

                        kept_count = len(records_for_summary)
                        print(
                            f"Generated error-filtered analysis report: {output_file_wo_error}"
                            f" (removed {removed_count}, kept {kept_count})"
                        )
                    finally:
                        try:
                            tmp_file_wo_error.unlink()
                        except FileNotFoundError:
                            pass
                        try:
                            tmp_summary_file.unlink()
                        except FileNotFoundError:
                            pass
            else:
                print("⚠️ Could not load result data, skipping error-filtered report.")

            # Export to CSV if requested
            if args.export_csv:
                analyzer.export_to_csv(args.file, csv_file)
                print(f"📄 Exported to CSV: {csv_file}")

            # Generate plots if requested
            if args.plot:
                analyzer.plot_complexity_performance(args.file, plot_file)
                print(f"📈 Complexity plot saved: {plot_file}")

        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
    
    elif args.compare:
        # Compare multiple files
        print(f"🔄 Comparing {len(args.compare)} result files...")
        
        try:
            comparison = analyzer.compare_models(args.compare)
            
            print("\n📊 Model Comparison:")
            for i, model in enumerate(comparison.model_names):
                print(f"  {model}: {comparison.accuracies[i]:.3f} accuracy, {comparison.response_times[i]:.2f}s avg time")
            
            # Generate comparison plot if requested
            if args.plot:
                plot_file = "model_comparison.png"
                analyzer.plot_model_comparison(args.compare, plot_file)
                print(f"📈 Comparison plot saved: {plot_file}")
                
        except Exception as e:
            print(f"❌ Comparison failed: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
    
    else:
        # List available result files (now in folders)
        results_path = Path(args.results_dir)
        result_folders = [d for d in results_path.iterdir() if d.is_dir() and d.name.startswith('eval_')]
        
        if result_folders:
            print("📁 Available evaluation runs:")
            for folder in sorted(result_folders, reverse=True):  # Most recent first
                results_file = folder / "results.json"
                if results_file.exists():
                    print(f"  - {folder.name}/")
                    print(f"    └── results.json")
            print("\nUse --file <folder_name>/results.json to analyze a specific run")
            print("Use --compare <folder1>/results.json <folder2>/results.json ... to compare multiple runs")
        else:
            # Also check for old format files
            result_files = list(results_path.glob("*_results.json"))
            if result_files:
                print("📁 Available result files (old format):")
                for f in result_files:
                    print(f"  - {f.name}")
                print("\nUse --file <filename> to analyze a specific file")
            else:
                print(f"❌ No result files found in {args.results_dir}")


def cmd_config(args):
    """Manage configurations"""
    config_manager = ConfigManager()
    
    if args.list:
        # List available configurations
        configs = config_manager.list_configs()
        if configs:
            print("📝 Available configurations:")
            for config in configs:
                print(f"  - {config}")
        else:
            print("❌ No configurations found")
    
    elif args.show:
        # Show specific configuration
        try:
            config = config_manager.load_config(args.show)
            print(f"📄 Configuration: {args.show}")
            print(f"  Model: {config.model.provider} - {config.model.model_name}")
            print(f"  Data: {config.evaluation.data_path}")
            print(f"  Template: {config.evaluation.prompt_template}")
            if config.evaluation.max_examples:
                print(f"  Max examples: {config.evaluation.max_examples}")
            
            # Validate
            issues = config_manager.validate_config(config)
            if issues:
                print("⚠️  Validation issues:")
                for issue in issues:
                    print(f"    - {issue}")
            else:
                print("✅ Configuration is valid")
                
        except Exception as e:
            print(f"❌ Failed to load configuration: {e}")
    
    elif args.create:
        # Create new configuration
        try:
            config = config_manager.create_config(
                provider=args.provider,
                model_name=args.model_name,
                data_path=args.data_path,
                output_name=args.create,
                base_url=args.base_url,
                max_examples=args.max_examples,
                prompt_template=args.template or "json_direct"
            )
            print(f"✅ Created configuration: {args.create}.yaml")
            
        except Exception as e:
            print(f"❌ Failed to create configuration: {e}")
    
    elif args.presets:
        # Show preset configurations
        presets = get_preset_configs()
        print("🎯 Available presets:")
        for name, config in presets.items():
            model = config['model']
            print(f"  - {name}: {model['provider']} - {model['model_name']}")


def cmd_test(args):
    """Test model connection"""
    print("🔧 Testing Model Connection...")
    
    try:
        # Create LLM interface
        provider_config = {}
        if args.base_url:
            provider_config['base_url'] = args.base_url
        if args.api_key:
            provider_config['api_key'] = args.api_key
        
        llm = LLMInterface(
            config.model.provider,
            config.model.model_name,
            **(config.model.provider_config or {})
        )
        
        # Test with simple prompt
        test_prompt = "Hello, how are you? Please respond briefly."
        print(f"🤖 Testing with prompt: {test_prompt}")
        
        response = llm.completion(test_prompt, max_tokens=100)
        
        if response.error:
            print(f"❌ Test failed: {response.error}")
        else:
            print(f"✅ Connection successful!")
            print(f"📝 Response: {response.response_text[:100]}...")
            
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Logical Reasoning Evaluation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with 10 examples
  python run_evaluation.py evaluate --preset quick_test

  # Full evaluation with custom config
  python run_evaluation.py evaluate --config my_config.yaml

  # Evaluate with specific parameters
  python run_evaluation.py evaluate --config ollama_qwen.yaml --max-examples 50 --template json_cot

  # Analyze results
  python run_evaluation.py analyze --file results_20241201_120000.json

  # Compare multiple models
  python run_evaluation.py analyze --compare model1_results.json model2_results.json --plot

  # Create new configuration
  python run_evaluation.py config --create my_config --provider ollama --model-name qwen2.5:32b --data-path data/ProverQA/unifiedinput.json

  # Test model connection
  python run_evaluation.py test --provider ollama --model-name qwen2.5:32b --base-url http://localhost:11434
        """
    )
    
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO', help='Log level')
    parser.add_argument('--log-file', help='Write logs to file (rotating, 5MB * 5). Keeps console output.')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Run evaluation')
    eval_parser.add_argument('--config', '-c', help='Configuration file')
    eval_parser.add_argument('--preset', '-p', help='Preset configuration')
    eval_parser.add_argument('--max-examples', type=int, help='Maximum number of examples')
    eval_parser.add_argument('--start-idx', type=int, help='Start index')
    eval_parser.add_argument('--end-idx', type=int, help='End index')
    eval_parser.add_argument('--output-dir', help='Output directory')
    eval_parser.add_argument('--template', choices=['direct', 'cot', 'json_direct', 'json_cot'], help='Prompt template')
    eval_parser.add_argument('--temperature', type=float, help='Model temperature')
    eval_parser.add_argument('--max-tokens', type=int, help='Maximum tokens')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze results')
    analyze_parser.add_argument('--file', '-f', help='Specific result file to analyze')
    analyze_parser.add_argument('--compare', nargs='+', help='Compare multiple result files')
    analyze_parser.add_argument('--results-dir', default='results/evaluation', help='Results directory')
    analyze_parser.add_argument('--export-csv', action='store_true', help='Export to CSV')
    analyze_parser.add_argument('--plot', action='store_true', help='Generate plots')
    analyze_parser.add_argument('--quiet', '-q', action='store_true', help='Quiet output')
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Manage configurations')
    config_parser.add_argument('--list', '-l', action='store_true', help='List configurations')
    config_parser.add_argument('--show', '-s', help='Show specific configuration')
    config_parser.add_argument('--create', help='Create new configuration')
    config_parser.add_argument('--presets', action='store_true', help='Show preset configurations')
    config_parser.add_argument('--provider', help='Model provider')
    config_parser.add_argument('--model-name', help='Model name')
    config_parser.add_argument('--data-path', help='Data file path')
    config_parser.add_argument('--base-url', help='Base URL for local models')
    config_parser.add_argument('--max-examples', type=int, help='Maximum examples')
    config_parser.add_argument('--template', help='Prompt template')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test model connection')
    test_parser.add_argument('--provider', required=True, help='Model provider')
    test_parser.add_argument('--model-name', required=True, help='Model name')
    test_parser.add_argument('--base-url', help='Base URL for local models')
    test_parser.add_argument('--api-key', help='API key')
    
    args = parser.parse_args()
    
    # Auto log file in evaluation/logs if none specified
    if not args.log_file and args.command:
        try:
            from pathlib import Path as _Path
            default_logs_dir = _Path(__file__).parent / 'evaluation' / 'logs'
            default_logs_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            auto_name = f"{args.command}_log_{ts}.log"
            args.log_file = str(default_logs_dir / auto_name)
        except Exception as _e:
            # Fallback silently if directory creation fails
            pass

    # Setup logging
    setup_logging(args.log_level, args.log_file)
    if args.log_file:
        logging.getLogger(__name__).info(f"Logging to file: {args.log_file}")
    
    if not args.command:
        parser.print_help()
        return
    
    # Route to appropriate command
    if args.command == 'evaluate':
        cmd_evaluate(args)
    elif args.command == 'analyze':
        cmd_analyze(args)
    elif args.command == 'config':
        cmd_config(args)
    elif args.command == 'test':
        cmd_test(args)


if __name__ == "__main__":
    main()
