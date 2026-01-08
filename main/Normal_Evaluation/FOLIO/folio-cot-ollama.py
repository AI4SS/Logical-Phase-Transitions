# FOLIO with Chain-of-Thought (Ollama Version)
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
    parser = argparse.ArgumentParser(description="Chain-of-Thought with Ollama")
    parser.add_argument('--temperature', type=float, default=0.0, help='temperature')
    parser.add_argument('--majoritycnt', type=int, choices=range(1, 101), default=1, help='numbers of majority voting times')
    parser.add_argument('--verbose', action='store_true', default=True, help='verbose mode')
    parser.add_argument('--model', type=str, default='qwen2.5-7b-instruct', help='model to use')
    parser.add_argument('--dataset', type=str, default='data/folio/folio-dev.json', help='dataset to use')
    parser.add_argument('--resume', type=str, default=None, help='resume from existing result file')
    parser.add_argument('--skip_existing', action='store_true', default=True, help='skip already processed examples when resuming')
    parser.add_argument('--output_file', type=str, default=None, help='specify output file name (useful for resuming)')
    return parser

parser = get_parser()
args = parser.parse_args()

client = OllamaClient(model_name=args.model, temperature=args.temperature)

def get_cot_prompt(premises, conclusion):
    """Create Chain-of-Thought prompt"""
    prompt = f"""Suppose you are one of the greatest AI scientists, logicians and mathematicians. Let's think step by step.

Read and analyze the "Premises" first, then using First-Order Logic (FOL) to judge whether the "Hypothesis" is True, False or Unknown.
Please make sure your reasoning is directly deduced from the "Premises" other than introducing unsourced common knowledge and unsourced information by common sense reasoning.
Provide consise reasoning(no more than 3 sentences)

Example:
Premises: "Miroslav Venhoda was a Czech choral conductor who specialized in the performance of Renaissance and Baroque music. Any choral conductor is a musician. Some musicians love music. Miroslav Venhoda published a book in 1946 called Method of Studying Gregorian Chant."
Hypothesis: "A Czech person wrote a book in 1946."
Output: {{"reasoning": "Miroslav Venhoda, who is specified as a Czech choral conductor, published a book in 1946. Thus, it is true that a Czech person wrote a book in 1946.", "answer": "True"}}

Example:
Premises: "All eels are fish. No fish are plants. A thing is either a plant or animal. Nothing that breathes is paper. All animals breathe. If a sea eel is either an eel or a plant, then a sea eel is an eel or an animal."
Hypothesis: "Sea eel is an eel."
Output: {{"reasoning": "All eels are fish and a sea eel is either an eel or a plant. No fish are plants, so a sea eel must be an eel. However, there's no direct information confirming this.", "answer": "Unknown"}}

Example:
Premises: "Miroslav Venhoda was a Czech choral conductor who specialized in the performance of Renaissance and Baroque music. Any choral conductor is a musician. Some musicians love music. Miroslav Venhoda published a book in 1946 called Method of Studying Gregorian Chant."
Hypothesis: "No choral conductor specialized in the performance of Renaissance."
Output: {{"reasoning": "Miroslav Venhoda, a choral conductor, specialized in Renaissance and Baroque music. Therefore, it is false that no choral conductor specialized in Renaissance.", "answer": "False"}}

Now please analyze the following:
Premises: "{premises}"
Hypothesis: "{conclusion}"
Output:"""
    return prompt

def parse_response(response):
    """Parse model response to extract reasoning and answer"""
    try:
        # Try to parse as JSON
        if "{" in response and "}" in response:
            json_match = re.search(r'\{.*?\}', response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                reasoning = parsed.get("reasoning", "")
                answer = parsed.get("answer", "")
                if answer in ["True", "False", "Unknown"]:
                    return reasoning, answer

        # Fallback: look for reasoning and answer patterns
        reasoning_match = re.search(r'reasoning["\s:]+([^"}]+)', response, re.IGNORECASE)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
        else:
            # Extract everything before the answer as reasoning
            if "answer" in response.lower():
                reasoning = response.lower().split("answer")[0].strip()
            else:
                # Extract last sentence as reasoning
                sentences = response.split('.')
                reasoning = sentences[-2].strip() if len(sentences) > 1 else response.strip()

        # Look for answer with better logic
        answer = "Unknown"
        response_lower = response.lower()

        # Check for explicit answer statements
        if "answer is true" in response_lower or "statement is true" in response_lower:
            answer = "True"
        elif "answer is false" in response_lower or "statement is false" in response_lower:
            answer = "False"
        elif "answer is unknown" in response_lower or "cannot determine" in response_lower:
            answer = "Unknown"
        else:
            # Check positions of all possible answers
            true_pos = response_lower.rfind("true")
            false_pos = response_lower.rfind("false")
            unknown_pos = response_lower.rfind("unknown")

            # Find which appears last
            max_pos = max(true_pos, false_pos, unknown_pos)
            if max_pos == true_pos and true_pos != -1:
                answer = "True"
            elif max_pos == false_pos and false_pos != -1:
                answer = "False"
            elif max_pos == unknown_pos and unknown_pos != -1:
                answer = "Unknown"

        return reasoning, answer
    except Exception:
        # Last resort: return the response as reasoning and default to Unknown
        return response.strip(), "Unknown"

def main():
    # Load the data from the JSON file
    with open(args.dataset, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Preprocess data
    for item in data:
        conclusion = re.search(r"\? ?(.*?)$", item['question'], re.S)
        conclusion = conclusion.group(1) if conclusion else item['question']

        clauses = item['context'].split('.')
        results = re.search(r'If (.*), (?:then )(.*)(?=$)', conclusion, re.S)
        item['premises'] = [clause.strip() + '.' for clause in clauses if clause.strip()]
        if results:
            premise = results.group(1)
            hypothesis = results.group(2)
            item['conclusion'] = hypothesis
            item['premises'].append(premise)
        else:
            item['conclusion'] = conclusion

        if item['answer'] == 'A':
            item['label'] = 'True'
        elif item['answer'] == 'B':
            item['label'] = 'False'
        elif item['answer'] == 'C':
            item['label'] = 'Unknown'

        item['example_id'] = item['id']
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
        logfilename = f'results/folio/results-folio-cot-ollama--{model_name}--t{args.temperature}--{dataset_name}--k_{args.majoritycnt}--{time.strftime("%Y-%m-%d-%H-%M-%S", t)}.jsonl'

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
        premises = ' '.join(example['premises'])
        conclusion = example['conclusion']

        if args.verbose:
            print("[Premises]:\t", premises)
            print("[Hypothesis]:\t", conclusion)

        # Majority vote for judgement
        judgement_cnt = {"True": 0, "False": 0, "Unknown": 0}
        reasoning_list = []

        for i in range(args.majoritycnt):
            try_cnt = 0
            reasoning = ""
            judgement = "Unknown"

            while try_cnt < TRY_CNT:
                try:
                    prompt = get_cot_prompt(premises, conclusion)
                    response = client.generate(prompt)
                    reasoning, judgement = parse_response(response)

                    # Validate judgement
                    if judgement not in ["True", "False", "Unknown"]:
                        # Try to extract from reasoning
                        if "true" in reasoning.lower():
                            judgement = "True"
                        elif "false" in reasoning.lower():
                            judgement = "False"
                        else:
                            judgement = "Unknown"

                    break
                except Exception as e:
                    print(f"COT generation failed: {e}")
                    try_cnt += 1
                    time.sleep(0.1)

            judgement_cnt[judgement] += 1
            reasoning_list.append(reasoning)

            if args.verbose:
                print(f"[Reasoning {i+1}]: {reasoning}")
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
            "reasoning_list": reasoning_list,
        }

        # Append result to file
        with open(logfilename, 'a') as f:
            json.dump(result, f, indent=4)
            f.write('\n')

if __name__ == "__main__":
    main()