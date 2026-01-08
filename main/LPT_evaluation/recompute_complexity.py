"""
重新计算已有评估结果的复杂度分数
不需要重新运行大模型，只重新计算复杂度
支持 NSA-LR 和 ProverQA 数据集
"""
import json
import os
import sys
from pathlib import Path

# 添加src目录到Python路径
sys.path.append('src')
sys.path.append('evaluation/utils')

from complexity_analyzer import ComplexityAnalyzer, load_folio_data
from data_loader import count_hops

def load_proverqa_data(file_path: str):
    """加载ProverQA数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def detect_dataset_type(evaluation_folder: str):
    """检测数据集类型"""
    # 优先检查路径中是否包含 proverqa
    if 'proverqa' in evaluation_folder.lower():
        return 'proverqa'
    # 其次检查 folio (因为 folio_eval 前缀可能出现在 ProverQA 结果中，所以要先排除 proverqa)
    elif 'folio' in evaluation_folder.lower() and 'proverqa' not in evaluation_folder.lower():
        return 'folio'
    else:
        # 默认 ProverQA
        return 'proverqa'

def get_proverqa_difficulty(evaluation_folder: str):
    """从路径中提取ProverQA难度级别"""
    if 'easy' in evaluation_folder.lower():
        return 'easy'
    elif 'medium' in evaluation_folder.lower():
        return 'medium'
    elif 'hard' in evaluation_folder.lower():
        return 'hard'
    else:
        return 'easy'  # 默认为easy

def recompute_evaluation_complexity(evaluation_folder: str, dataset_override: str = None):
    """重新计算评估结果的复杂度"""

    # 检测数据集类型
    if dataset_override:
        dataset_type = dataset_override
    else:
        dataset_type = detect_dataset_type(evaluation_folder)
    print(f"检测到数据集类型: {dataset_type}")
    
    # 读取原评估结果
    results_file = f"{evaluation_folder}/results.json"
    if not os.path.exists(results_file):
        print(f"结果文件不存在: {results_file}")
        return
    
    with open(results_file, 'r', encoding='utf-8') as f:
        evaluation_data = json.load(f)
    
    # 检查数据格式 - 可能是列表或字典
    if isinstance(evaluation_data, list):
        print(f"评估结果是列表格式，包含 {len(evaluation_data)} 个样本")
        results_list = evaluation_data
    else:
        print("评估结果是字典格式")
        results_list = evaluation_data.get('results', [])
    
    print(f"重新计算 {len(results_list)} 个样本的复杂度...")
    
    # 根据数据集类型加载原始数据
    if dataset_type == 'folio':
        # 加载FOLIO原始数据
        original_data = load_folio_data('data/folio_v2_train.jsonl')
    else:  # proverqa
        # 加载ProverQA原始数据
        # 默认使用 unifiedinput.json
        proverqa_file = 'data/ProverQA/unified_version.json'
        if os.path.exists(proverqa_file):
            original_data = load_proverqa_data(proverqa_file)
            print(f"加载ProverQA统一数据: {proverqa_file}，共{len(original_data)}个样本")
        else:
            difficulty = get_proverqa_difficulty(evaluation_folder)
            proverqa_file = f'data/ProverQA/{difficulty}.json'
            original_data = load_proverqa_data(proverqa_file)
            print(f"加载ProverQA {difficulty}数据，共{len(original_data)}个样本")
    
    # 创建新的复杂度分析器（使用修改后的计算方法）
    analyzer = ComplexityAnalyzer()
    
    # 重新计算每个样本的复杂度
    updated_results = []
    for i, result in enumerate(results_list):
        # 获取对应的原始数据
        if dataset_type == 'folio':
            original_sample = original_data[i]
            # 计算跳数（如果有reasoning字段）
            reasoning = original_sample.get('reasoning', '')
            hop_count = count_hops(reasoning)
            # 重新计算复杂度，传入hop_count
            complexity = analyzer.analyze_folio_sample(original_sample, hop_count=hop_count)
        else:  # proverqa
            # 对于ProverQA，由于可能存在ID重置的情况（如0-499, 0-499...），
            # 且我们默认使用 unifiedinput.json (0-1499)，
            # 因此直接使用索引 i 来对应原始数据是最准确的（假设结果是按顺序生成的）。
            if i < len(original_data):
                original_sample = original_data[i]
                # 将ProverQA格式转换为FOLIO格式进行复杂度分析
                # 提取nl2fol中的逻辑表达式
                nl2fol = original_sample.get('nl2fol', {})

                # 将nl2fol值列表合并为前提
                premises_fol = []
                for key, fol_expr in nl2fol.items():
                    if fol_expr and fol_expr.strip():
                        premises_fol.append(fol_expr)

                # 结论是conclusion_fol字段
                conclusion_fol = original_sample.get('conclusion_fol', '')

                # 计算跳数（从reasoning字段）
                reasoning = original_sample.get('reasoning', '')
                hop_count = count_hops(reasoning)

                folio_like_sample = {
                    'premises-FOL': premises_fol,
                    'conclusion-FOL': conclusion_fol
                }
                # 重新计算复杂度，传入hop_count
                complexity = analyzer.analyze_folio_sample(folio_like_sample, hop_count=hop_count)
            else:
                print(f"警告: 索引 {i} 超出数据范围 (共 {len(original_data)} 条)，使用默认复杂度")
                complexity = analyzer.analyze_folio_sample({'premises-FOL': [], 'conclusion-FOL': ''})
                hop_count = 0

        # 更新结果中的复杂度信息
        result['complexity_score'] = complexity.total_complexity.total
        result['hop_count'] = hop_count

        updated_results.append(result)
    
    # 创建新的结果文件夹
    new_folder = f"{evaluation_folder}_semantic_only"
    os.makedirs(new_folder, exist_ok=True)
    
    # 保存更新后的结果
    if isinstance(evaluation_data, list):
        # 直接保存列表
        with open(f"{new_folder}/results.json", 'w', encoding='utf-8') as f:
            json.dump(updated_results, f, indent=2, ensure_ascii=False)
    else:
        # 保存字典格式
        evaluation_data['results'] = updated_results
        with open(f"{new_folder}/results.json", 'w', encoding='utf-8') as f:
            json.dump(evaluation_data, f, indent=2, ensure_ascii=False)
    
    # 复制其他文件
    for file in ['config.yaml', 'summary.json']:
        src = f"{evaluation_folder}/{file}"
        dst = f"{new_folder}/{file}"
        if os.path.exists(src):
            import shutil
            shutil.copy2(src, dst)
    
    print(f"✅ 新结果已保存到: {new_folder}")
    return new_folder

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='重新计算评估结果的复杂度分数')
    parser.add_argument('--folder', type=str, help='评估结果文件夹路径')
    parser.add_argument('--auto', action='store_true', help='自动处理最新的CoT评估结果')
    parser.add_argument('--dataset', type=str, choices=['folio', 'proverqa'], help='强制指定数据集类型 (folio 或 proverqa)')

    args = parser.parse_args()
    
    if args.folder:
        # 处理指定的文件夹
        if os.path.exists(args.folder):
            print(f"处理指定的评估结果: {args.folder}")
            new_folder = recompute_evaluation_complexity(args.folder, args.dataset)
        else:
            print(f"文件夹不存在: {args.folder}")
    elif args.auto:
        # 自动处理最新的CoT评估结果
        results_dir = "results/evaluation"
        if os.path.exists(results_dir):
            folders = [f for f in os.listdir(results_dir) if f.startswith("folio_eval") and "cot" in f]
            if folders:
                latest_folder = sorted(folders)[-1]
                folder_path = f"{results_dir}/{latest_folder}"
                print(f"处理最新的FOLIO CoT评估结果: {folder_path}")
                new_folder = recompute_evaluation_complexity(folder_path, args.dataset)
            else:
                print("未找到FOLIO CoT评估结果")
        else:
            print("FOLIO评估结果目录不存在")
    else:
        # 默认行为：查找最新的CoT评估结果
        results_dir = "results/evaluation"
        if os.path.exists(results_dir):
            folders = [f for f in os.listdir(results_dir) if f.startswith("folio_eval") and "cot" in f]
            if folders:
                latest_folder = sorted(folders)[-1]
                folder_path = f"{results_dir}/{latest_folder}"
                print(f"处理最新的CoT评估结果: {folder_path}")
                new_folder = recompute_evaluation_complexity(folder_path, args.dataset)
            else:
                print("未找到CoT评估结果")
        else:
            print("评估结果目录不存在")