# ProverQA with Direct (Ollama Version) - No reasoning
import argparse
import json
import time
import re
from tqdm import tqdm
import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.ollama_utils import OllamaClient

TRY_CNT = 16

def get_parser():
    parser = argparse.ArgumentParser(description="Direct evaluation with Ollama")
    parser.add_argument('--temperature', type=float, default=0, help='temperature')
    parser.add_argument('--majoritycnt', type=int, choices=range(1, 101), default=1, help='numbers of majority voting times')
    parser.add_argument('--verbose', action='store_true', default=True, help='verbose mode')
    parser.add_argument('--model', type=str, default='qwen2.5-7b-instruct', help='model to use')
    parser.add_argument('--dataset', type=str, default='data/medium.json', help='dataset to use')
    parser.add_argument('--resume', type=str, default=None, help='resume from existing result file')
    parser.add_argument('--skip_existing', action='store_true', default=True, help='skip already processed examples when resuming')
    parser.add_argument('--output_file', type=str, default=None, help='specify output file name (useful for resuming)')
    return parser

parser = get_parser()
args = parser.parse_args()

client = OllamaClient(model_name=args.model, temperature=args.temperature)

def get_direct_prompt(premises, conclusion):
    """Create direct prompt without reasoning"""
    prompt = f"""Based only on the given premises, determine if the hypothesis is True, False, or Unknown.

Premises: {premises}

Hypothesis: {conclusion}

Based strictly on the premises provided, is the hypothesis True, False, or Unknown? Answer with only one word: "True", "False", or "Unknown".

Answer:"""
    return prompt

def parse_response(response):
    """Parse model response to extract answer"""
    response = response.strip().lower()

    # Extract the last word as it's likely the final answer
    words = response.split()
    if words:
        last_word = words[-1].strip('.,!?;"')

        # Check the last word first
        if last_word in ["true", "false", "unknown"]:
            return last_word.capitalize()

    # Check if response contains explicit answer statements
    if "answer is true" in response or "statement is true" in response:
        return "True"
    elif "answer is false" in response or "statement is false" in response:
        return "False"
    elif "answer is unknown" in response or "cannot determine" in response:
        return "Unknown"

    # Check positions of all possible answers
    true_pos = response.rfind("true")
    false_pos = response.rfind("false")
    unknown_pos = response.rfind("unknown")

    # Find which appears last
    max_pos = max(true_pos, false_pos, unknown_pos)
    if max_pos == true_pos and true_pos != -1:
        return "True"
    elif max_pos == false_pos and false_pos != -1:
        return "False"
    elif max_pos == unknown_pos and unknown_pos != -1:
        return "Unknown"

    # Default to Unknown if nothing is found
    return "Unknown"

def main():
    # Load the data from the JSON file
    with open(args.dataset, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Preprocess data
    for item in data:
        # ProverQA specific format
        item['premises'] = item.get('context', '')
        # Extract the statement from the question
        question = item.get('question', '')
        if 'Based on the above information, is the following statement true, false, or uncertain?' in question:
            # Extract the actual statement after the question
            parts = question.split('Based on the above information, is the following statement true, false, or uncertain?')
            if len(parts) > 1:
                item['conclusion'] = parts[1].strip().rstrip('.')
            else:
                item['conclusion'] = question.strip()
        else:
            item['conclusion'] = question.strip()

        # Map the answer
        answer = item.get('answer', '')
        if answer == 'A' or answer == 'True':
            item['label'] = 'True'
        elif answer == 'B' or answer == 'False':
            item['label'] = 'False'
        elif answer == 'C' or answer == 'Unknown':
            item['label'] = 'Unknown'
        else:
            item['label'] = 'Unknown'

        item['example_id'] = item.get('id', str(len(data)))

        # Cleanup
        if 'id' in item: del item['id']
        if 'answer' in item: del item['answer']
        if 'context' in item: del item['context']
        if 'question' in item: del item['question']

    t = time.localtime()
    dataset_name = args.dataset.split('/')[-1].split('.')[0]
    model_name = args.model.replace(':', '-')
    # Handle custom output file or resume
    if args.output_file:
        logfilename = args.output_file
    elif args.resume:
        logfilename = args.resume
    else:
        logfilename = f'results/proverqa/results-proverqa-direct-ollama--{model_name}--t{args.temperature}--{dataset_name}--k_{args.majoritycnt}--{time.strftime("%Y-%m-%d-%H-%M-%S", t)}.jsonl'

    # If resuming, make sure we don't overwrite existing header
    write_header = not (args.resume and os.path.exists(logfilename))

    os.makedirs(os.path.dirname(logfilename), exist_ok=True)

    # Write header only if not resuming or file doesn't exist
    if write_header:
        with open(logfilename, 'w') as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S", t) + '\n')
            f.write(f"Model: {args.model}\n")
            f.write(f"Temperature: {args.temperature}\n")
            f.write(f"Majority Cnt: {args.majoritycnt}\n")
            f.write(f"Dataset: {args.dataset}\n")
            f.write("--------------------------------\n")

    # Initialize counter for correct predictions
    correct_predictions = 0
    cnt = 0
    total_cnt = len(data)

    # Handle resume functionality
    processed_ids = set()

    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}")
        # Read existing results
        with open(args.resume, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('20'):  # Skip header lines
                    try:
                        result = json.loads(line)
                        if 'example_id' in result:
                            processed_ids.add(result['example_id'])
                    except:
                        pass

        print(f"Found {len(processed_ids)} already processed examples")

        # Filter out already processed examples if skip_existing is True
        if args.skip_existing:
            original_length = len(data)
            data = [item for item in data if item['example_id'] not in processed_ids]
            print(f"Skipping {original_length - len(data)} already processed examples")

    # Iterate over the data from the JSON file
    for example in tqdm(data, desc="Evaluating", unit="example"):
        cnt += 1
        print(f"-------------------------\n### Example ID: {example['example_id']} \t ( {cnt} / {total_cnt} )")
        premises = example['premises']
        conclusion = example['conclusion']

        if args.verbose:
            print("[Premises]:\t", premises)
            print("[Hypothesis]:\t", conclusion)

        # Majority vote for judgement
        judgement_cnt = {"True": 0, "False": 0, "Unknown": 0}

        for i in range(args.majoritycnt):
            try_cnt = 0
            judgement = "Unknown"

            while try_cnt < TRY_CNT:
                try:
                    prompt = get_direct_prompt(premises, conclusion)
                    response = client.generate(prompt)
                    judgement = parse_response(response)
                    break
                except Exception as e:
                    print(f"Direct generation failed: {e}")
                    try_cnt += 1
                    time.sleep(0.1)

            judgement_cnt[judgement] += 1

            if args.verbose:
                print(f"[Judgement {i+1}]: {judgement}")

        # Select the one with the highest count
        majority_judgement = max(judgement_cnt, key=judgement_cnt.get)

        # Calculate the number of correct predictions
        if majority_judgement == example["label"]:
            correct_predictions += 1

        print("[Prediction]: ", majority_judgement)
        print("[Actual]: ", example["label"])

        # Calculate and print the running accuracy
        accuracy = correct_predictions / cnt
        print("[Running Average Accuracy]: ", accuracy)

        result = {
            "example_id": example["example_id"],
            "prediction": majority_judgement,
            "actual": example["label"],
            "accuracy": accuracy,
            "judgement_cnt": judgement_cnt,
        }

        # Append result to file
        with open(logfilename, 'a') as f:
            json.dump(result, f, indent=4)
            f.write('\n')

if __name__ == "__main__":
    main()