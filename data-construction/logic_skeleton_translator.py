import os
import json
import random
import pickle
import argparse
import time
import logging

import numpy as np

from utils.logic_translator.translator import Translator
from utils.logic_translator.noise import NoiseTranslator
from utils.logic_translator.generator import ProblemGenerator


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'logic_skeleton_translator_{time.strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )
    logger = logging.getLogger(__name__)
    
    parser = argparse.ArgumentParser()
    
    # required parameters
    parser.add_argument("--num", type=int, default=300)
    parser.add_argument("--mode", type=str, default='hard')
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=300)
    parser.add_argument("--data_dir", type=str, default="outputs/logic_data")
    parser.add_argument("--output_dir", type=str, default="outputs/translated_data")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3.1-70B-Instruct")
    
    # For local models
    parser.add_argument("--base_url", type=str, default="EMPTY")
    parser.add_argument("--api_key", type=str, default="EMPTY")
    
    # default parameters
    parser.add_argument("--predicate_path", type=str, default="data/wordnet_predicates.json")
    parser.add_argument("--example_path", type=str, default="data/translation_examples.json")
    parser.add_argument("--name_path", type=str, default="data/names")
    parser.add_argument("--seed", type=int, default=741)
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    logger.info(f"Starting logic skeleton translation with args: {vars(args)}")
    
    seed_everything(args.seed)
    logger.info(f"Set random seed to {args.seed}")
    
    start_time = time.time()
    
    # load dataset
    logger.info(f"Loading dataset from {args.data_dir}/{args.mode}-{args.num}.pickle")
    with open(f'{args.data_dir}/{args.mode}-{args.num}.pickle', 'rb') as f:
        logic_data = pickle.load(f)
        
    # translate facts and rules（每10个实时保存一次）
    logger.info("Initializing translator and starting facts and rules translation with periodic saving...")
    translator = Translator(args)
    translated_problems = []
    save_every = 1
    output_path = f"{args.output_dir}/{args.mode}-{args.num}-{args.start}_{args.end}.json"
    temp_path = output_path + ".tmp"
    total = len(logic_data)
    
    existing_count = 0
    if os.path.exists(temp_path):
        with open(temp_path, "r") as f:
            translated_problems = json.load(f)
        existing_count = len(translated_problems)
        logger.info(f"Found existing temporary file with {existing_count} translated problems. Resuming from there.")
    else:
        logger.info("No existing temporary file found. Starting fresh translation.")
        
    total=len(logic_data)
    for idx, problem in enumerate(logic_data):
        if idx < args.start or idx >= args.end:
            continue
        if idx < args.start + existing_count:
            continue
        result = translator.translate_rules_and_facts(data=[problem])
        if result and result[0] is not None:
            translated_problems.append(result[0])
        if (len(translated_problems) % save_every == 0 and len(translated_problems) > 0) or (idx == total - 1):
            # 实时保存到临时文件
            with open(temp_path, "w") as f:
                json.dump(translated_problems, f, indent=2, ensure_ascii=False)
            logger.info(f"[AutoSave] Saved {len(translated_problems)} problems to {temp_path}")

    # Filter out None results from translation
    initial_count = len(translated_problems)
    translated_problems = [p for p in translated_problems if p is not None]
    if len(translated_problems) < initial_count:
        logger.warning(f"Filtered out {initial_count - len(translated_problems)} problems that failed translation.")

    if not translated_problems:
        logger.error("No problems were successfully translated. Exiting.")
        exit()
    # 最终保存到正式文件
    with open(output_path, "w") as f:
        json.dump(translated_problems, f, indent=2, ensure_ascii=False)
    logger.info(f"[FinalSave] Saved all {len(translated_problems)} problems to {output_path}")
    
    # translate distracting facts and rules（每1个实时保存一次）
    logger.info("Translate distracting facts and rules with periodic saving...")
    noise_temp_path = f"{output_path}.noise.tmp"
    save_every_noise = 1
    total_noise = len(translated_problems)
    
    # 如果存在噪声阶段的临时文件，则从中恢复
    noise_processed = []
    existing_noise_count = 0
    if os.path.exists(noise_temp_path):
        with open(noise_temp_path, "r") as f:
            noise_processed = json.load(f)
        existing_noise_count = len(noise_processed)
        logger.info(f"Found noise temp file with {existing_noise_count} items. Resuming...")
    else:
        logger.info("No noise temp file found. Starting fresh noise translation.")
    
    for idx, item in enumerate(translated_problems):
        if idx < existing_noise_count:
            continue

        noise_translator = NoiseTranslator(args, translated_data=[item])
        result_list = noise_translator.create_distracting_rules()
        updated_item = result_list[0] if result_list and result_list[0] is not None else item
        noise_processed.append(updated_item)

        if (len(noise_processed) % save_every_noise == 0 and len(noise_processed) > 0) or (idx == total_noise - 1):
            with open(noise_temp_path, "w") as f:
                json.dump(noise_processed, f, indent=2, ensure_ascii=False)
            logger.info(f"[AutoSave-Noise] Saved {len(noise_processed)} items to {noise_temp_path}")
    
    # 将已完成的噪声翻译结果合并回原始列表
    if len(noise_processed) > 0:
        translated_problems = noise_processed + translated_problems[len(noise_processed):]
    
    # generate problems
    logger.info("Generating final problems...")
    problem_generator = ProblemGenerator(args, translated_data=translated_problems)
    translated_problems = problem_generator.create_problems()

    # save result
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f"Created output directory: {args.output_dir}")

    output_path = f"{args.output_dir}/{args.mode}-{args.num}-{args.start}_{args.end}.json"
    logger.info(f"Saving translated problems to {output_path}")
    with open(output_path, "w") as f:
        json.dump(translated_problems, f, indent=2, ensure_ascii=False)
        
    duration = time.time() - start_time
    logger.info(f"Total time: {duration:.2f} seconds")
    logger.info(f"Average time per problem: {duration / args.num:.2f} seconds")
    
    logger.info("Logic skeleton translation completed successfully.")
    
