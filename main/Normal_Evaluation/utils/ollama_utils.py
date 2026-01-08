import ollama
import time
import re

class OllamaClient:
    def __init__(self, model_name="gemma3:1b", temperature=0.1, max_tokens=4096):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt, stop=None, max_tokens=None):
        options = {
            "temperature": self.temperature,
            "num_predict": max_tokens if max_tokens is not None else self.max_tokens,
        }
        if stop:
            options["stop"] = stop

        try:
            response = ollama.generate(model=self.model_name, prompt=prompt, options=options)
            return response['response']
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            return ""

    def chat(self, messages, stop=None, max_tokens=None):
        options = {
            "temperature": self.temperature,
            "num_predict": max_tokens if max_tokens is not None else self.max_tokens,
        }
        if stop:
            options["stop"] = stop
        
        try:
            response = ollama.chat(model=self.model_name, messages=messages, options=options)
            return response['message']['content']
        except Exception as e:
            print(f"Error calling Ollama chat: {e}")
            return ""

# --- Prompts and Examples ---

# Common System Prompt
SYSTEM_PROMPT = """Suppose you are one of the greatest AI scientists, logicians and mathematicians. Let us think step by step.
Please make sure your reasoning is directly deduced from the "Premises" and "Propositions" other than introducing unsourced common knowledge and unsourced information by common sense reasoning."""

# 1. gen_proposition
gen_proposition_examples = [
    {'premises': 'All eels are fish. No fish are plants. All animals breathe.',
     'proposition': 'No eels are plants.',
     'conclusion': 'Sea eel is an eel.',
     'explanation': 'This expression is deduced from the two premises as follows: if x is an eel, then it is a fish (from Premise 1), and if it is a fish, then it is not a plant (from Premise 2). Thus, if x is an eel, then it is not a plant.'},
    {'premises': 'All eels are fish. A thing is either a plant or animal. No fish are plants.',
     'proposition': 'All eels are animals.',
     'conclusion': 'Sea eel is an eel.',
     'explanation': 'This statement follows from the premises as follows: If x is an eel, then it is a fish (from Premise 1). If x is a thing (which includes being a fish, hence an eel), then it is either a plant or an animal (from Premise 2). Since it cannot be a plant (because it is a fish and no fish is a plant), it must be an animal. Thus, if x is an eel, it is an animal.'},
    {'premises': 'A thing is either a plant or animal. All animals breathe.',
     'proposition': 'All things that breathe are animals.',
     'conclusion': 'Sea eel is an eel.',
     'explanation': 'This statement is deduced from the premises as follows: If x is a thing, then it is either a plant or an animal (from Premise 1), and if x is an animal, then it breathes (from Premise 2). Therefore, if a thing breathes, it must be an animal, because it can not be a plant that breathes based on these premises.'},
    {
     'premises': 'If you keep statement A, you must keep statement B and statement C. If you keep statement D, you must delete both statement E and statement C.Statement A is important information and cannot be deleted.Statement E and statement F should be saved at the same time.',
     'proposition': 'You must keep statement B and statement C',
     'conclusion': 'D need to be kept',
     'explanation': 'This statement is deduced from the premises as follows:If you keep statement A, you must keep statement B and statement C. Statement A is important information and cannot be deleted.Therefore, if A is saved, then B and C are saved.'
    }
]

def get_gen_proposition_prompt(premises, conclusion):
    prompt = f"{SYSTEM_PROMPT}\n"
    prompt += """Please use Logical Reasoning Rules(LRR) to deduce a "Proposition" from two given "Premises" and the proposition does not include "if".
Logical Reasoning Rules(LRR):
1. "Two premises": "If A,then B. A is true." then "Proposition": "B is true."
2. "Two premises": "If A,then B. B is not true." then "Proposition": "A is not true"
3. "Two premises": "A is either C or D. A is not C." then "Proposition": "A is D."
Please make sure that the "Proposition" is logically correct.
Please make sure that the "Proposition" is not a duplicate of the "Premises".
Please make sure your reasoning is directly deduced from the "Premises" and "Propositions" other than introducing unsourced common knowledge and unsourced information by common sense reasoning.
Please remember that your "Proposition" should be useful to determine whether the "Hypothesis" is True, False or Unknown.
----\n"""
    
    for ex in gen_proposition_examples:
        prompt += f"""---
"Premises": "{ex['premises']}"
We want to deduce more propositions to determine the correctness of the following "Hypothesis":
"Hypothesis": "{ex['conclusion']}"
Can you deduce a new "Proposition" from at least two given "Premises"?
"Proposition": "{ex['proposition']}"
"""
    
    prompt += f"""---
"Premises": "{premises}"
We want to deduce more propositions to determine the correctness of the following "Hypothesis":
"Hypothesis": "{conclusion}"
Can you deduce a new "Proposition" from at least two given "Premises"?
"Proposition": """
    return prompt

# 2. is_something
def get_is_something_prompt(proposition):
    prompt = f"{SYSTEM_PROMPT}\n"
    prompt += """Please determine whether there is a new useful "Proposition". Reply with True or False.
----\n"""
    
    examples = [
        ('There is no new proposition that can be deduced from the given premises to determine the correctness of the hypothesis.', 'False'),
        ('A Czech person wrote a book in 1946.', 'True'),
        ('There is no new proposition that can be deduced from the given premises that would be useful in determining the correctness of the given hypothesis.', 'False'),
        ('None of the premises provide information to deduce a proposition related to a Czech person writing a book in 1946.', 'False')
    ]
    
    for prop, label in examples:
        prompt += f"""---
"Proposition": "{prop}"
{label}
"""
    prompt += f"""---
"Proposition": "{proposition}"
"""
    return prompt

# 3. validate_deduction
validate_deduction_examples = [
    {'premises': 'All eels are fish. No fish are plants.',
     'proposition': 'No eels are plants.',
     'validation': 'True'},
    {'premises': 'If bear is red,then the bear is rough. The bear is red.',
     'proposition': 'The bear is rough.',
     'validation': 'True'},
    {'premises': 'Nothing that breathes is paper. All animals breathe.',
     'proposition': 'All animals are paper.',
     'validation': 'False'},
    {'premises': 'A thing is either a plant or animal. All animals breathe.',
     'proposition': 'All things that breathe are animals.',
     'validation': 'True'},
    {'premises': 'Miroslav Venhoda was a Czech choral conductor who specialized in the performance of Renaissance and Baroque music. Any choral conductor is a musician.',
     'proposition': 'Miroslav Venhoda, being a Czech choral conductor specializing in Renaissance and Baroque music, is also a musician.',
     'validation': 'True'},
    {'premises': 'Any choral conductor is a musician. Some musicians love music.',
     'proposition': 'All choral conductor love music.',
     'validation': 'False'},
    {'premises': 'Any choral conductor is a musician. Miroslav Venhoda published a book in 1946 called Method of Studying Gregorian Chant.',
     'proposition': 'Miroslav Venhoda, who published a book in 1946 called Method of Studying Gregorian Chant, is a musician as he is a choral conductor.',
     'validation': 'True'}
]

def get_validate_deduction_prompt(premises, proposition):
    prompt = f"{SYSTEM_PROMPT}\n"
    prompt += """Please use the Logical Reasoning Rules(LRR) to determine whether the deduction of the given "Premises" to a "Proposition" is valid or not, reply with True or False.
Logical Reasoning Rules(LRR):
1. "Two premises": "If A,then B. A is true." then "Proposition": "B is true."
2. "Two premises": "If A,then B. If B,then C." then "Proposition": "If A, then C."
3. "Two premises": "If A,then B. B is not true." then "Proposition": "A is not true"
4. "Two premises": "A is either C or D. A is not C." then "Proposition": "A is D."
----\n"""
    for ex in validate_deduction_examples:
        prompt += f"""---
"Premises": "{ex['premises']}"
"Proposition": "{ex['proposition']}"
"Judgement": "Is this deduction valid? {ex['validation']}"
"""
    prompt += f"""---
"Premises": "{premises}"
"Proposition": "{proposition}"
"Judgement": "Is this deduction valid? """
    return prompt

# 4. duplicated_deduction
duplicated_deduction_examples = [
    {'premises': 'Miroslav Venhoda was a Czech choral conductor who specialized in the performance of Renaissance and Baroque music. Any choral conductor is a musician. Some musicians love music. Miroslav Venhoda published a book in 1946 called Method of Studying Gregorian Chant.',
     'proposition': 'If someone is a choral conductor, then he is a musician.',
     'duplicated': 'True',
     'explanation': '"If someone is a choral conductor, then he is a musician." can be derived only using "Any choral conductor is a musician". So the answer is true.'
     },
    {'premises': 'All eels are fish. No fish are plants. A thing is either a plant or animal. Nothing that breathes is paper. All animals breathe. If a sea eel is either an eel or a plant, then a sea eel is an eel or an animal.',
     'proposition': 'No eels are plants.',
     'duplicated': 'False',
     'explanation': '"No eels are plants." can be derived using "All eels are fish." and "No fish are plants." So the answer is false.'
     }
]

def get_duplicated_deduction_prompt(premises, proposition):
    prompt = f"{SYSTEM_PROMPT}\n"
    prompt += """Can this "proposition" can be derived using only one "premise"?Please reply with True or False.
----\n"""
    for ex in duplicated_deduction_examples:
        prompt += f"""---
"Premises": "{ex['premises']}"
"Proposition": "{ex['proposition']}"
Can this "proposition" can be derived using only one "premise"?
"Judgement": "{ex['duplicated']}"
"explanation": "{ex['explanation']}"
"""
    prompt += f"""---
"Premises": "{premises}"
"Proposition": "{proposition}"
Can this "proposition" can be derived using only one "premise"?
"Judgement": " """
    return prompt

# 5. sourced_deduction
sourced_deduction_examples = [
    {'premises': 'All eels are fish. No fish are plants.',
     'proposition': 'No eels are plants.',
     'sourced': 'True'},
     {
      'premises': 'Nothing that breathes is paper. All animals breathe.',
      'proposition': 'All animals need food.',
      'sourced': 'False'}
]

def get_sourced_deduction_prompt(premises, proposition):
    prompt = f"{SYSTEM_PROMPT}\n"
    prompt += """Please determine whether the "Proposition" is directly deduced from the "Premises" with certainty other than introducing unsourced information by common sense reasoning, reply with True or False.
----\n"""
    for ex in sourced_deduction_examples:
        prompt += f"""---
"Premises": "{ex['premises']}"
"Proposition": "{ex['proposition']}"
"Judgement": "Is this proposition directly deduced from the premises? {ex['sourced']}"
"""
    prompt += f"""---
"Premises": "{premises}"
"Proposition": "{proposition}"
"Judgement": "Is this proposition directly deduced from the premises? """
    return prompt

# 6. structure_program (Common logic for structure_program, structure_program_wocot, structure_program_memory)
# We can combine these or keep them separate. The examples differ slightly.
# Let's use a unified function with a mode.

structure_program_examples = [
    {'premises': 'Miroslav Venhoda was a Czech choral conductor who specialized in the performance of Renaissance and Baroque music. Any choral conductor is a musician. Some musicians love music. Miroslav Venhoda published a book in 1946 called Method of Studying Gregorian Chant.',
     'propositions': 'Miroslav Venhoda, who published a book in 1946 called Method of Studying Gregorian Chant, is a musician as he is a choral conductor.',
     'conclusion': 'A Czech person wrote a book in 1946.',
     'reasoning': "Miroslav Venhoda, who is specified as a Czech choral conductor, published a book in 1946. Thus, it is true that a Czech person wrote a book in 1946.",
     'judgement': 'True'},
    {'premises': 'All eels are fish. No fish are plants. A thing is either a plant or animal. Nothing that breathes is paper. All animals breathe. If a sea eel is either an eel or a plant, then a sea eel is an eel or an animal.',
     'propositions': 'No eels are plants. All eels are animals.',
     'conclusion': 'Sea eel is an eel.',
     'reasoning': "all eels are fish and a sea eel is either an eel or a plant. It's also stated that no fish are plants. Therefore, a sea eel can't be a plant and must be an eel. However, there's no direct information about a sea eel being an eel.",
     'judgement': 'Unknown'},
    {'premises': 'Miroslav Venhoda was a Czech choral conductor who specialized in the performance of Renaissance and Baroque music. Any choral conductor is a musician. Some musicians love music. Miroslav Venhoda published a book in 1946 called Method of Studying Gregorian Chant.',
     'propositions': 'Miroslav Venhoda specialized in the performance of Renaissance and Baroque music,is a musician.',
     'conclusion': 'No choral conductor specialized in the performance of Renaissance.',
     'reasoning': "Miroslav Venhoda, a choral conductor, specialized in the performance of Renaissance and Baroque music. Thus, it is false to conclude that no choral conductor specialized in the performance of Renaissance.",
     'judgement': 'False'},
    {'premises': 'Anne is not furry. Anne is white. Bob is blue. Bob is cold. Bob is young. Erin is blue. Harry is not young. If someone is rough then they are cold. If someone is rough then they are white.If Harry is red, then Harry is cold. All white people are red. Red, rough people are young. If someone is blue then they are rough. If Anne is not red then Anne is young. Cold, young people are not furry.',
     'propositions': 'If Erin is young, then she is not furry.If Erin is cold and young, then she is not furry. If Erin is rough, then she is not furry.Erin is rough.If Erin is red and rough, then she is not furry.Erin is cold.',
     'conclusion': 'Erin is not furry.',
     'reasoning': "We know from the propositions that If Erin is rough, then she is not furry. And Erin is rough. So we know that Erin is not furry. So the Hypothesis is true.",
     'judgement': 'True'},
]

def get_structure_program_prompt(premises, propositions, conclusion, infer_history=None, mode="standard"):
    prompt = f"{SYSTEM_PROMPT}\n"
    if mode == "memory":
        prompt += """Read and analyze the "Premises" first, then use "Propositions" to reasoning whether the "Hypothesis" is True, False or Unknown.
Please make sure your reasoning is directly deduced from the "Premises" and "Propositions" other than introducing unsourced common knowledge and unsourced information by common sense reasoning.
If there are propositions that directly determine whether the hypothesis is true or false, you do not need to analyze the premises; otherwise, you should reason with both propositions and premises.
----\n"""
    else:
        prompt += """Read and analyze the "Premises" first, then judge whether the "Hypothesis" is True, False or Unknown.
Please make sure your reasoning is directly deduced from the "Premises" and "Propositions" other than introducing unsourced common knowledge and unsourced information by common sense reasoning.
----\n"""

    for ex in structure_program_examples:
        prompt += f"""---
"Premises": "{ex['premises']}"
"Hypothesis": "{ex['conclusion']}"
"Thoughts": "Let us think step by step. From the premises, we can deduce propositions: {ex['propositions']}"
"""
        if mode == "memory" and 'memory' in ex: # Example doesn't strictly have memory field in the list above, but logic implies it
             pass # Simplified for now as examples are shared
        
        prompt += f""""Reasoning": "Let us think step by step, {ex['reasoning']}"
"Recall the Hypothesis": "{ex['conclusion']}"
"Judgement": "Now we know that the Hypothesis is {ex['judgement']}"
"""

    prompt += f"""---
"Premises": "{premises}"
"Hypothesis": "{conclusion}"
"Thoughts": "Let us think step by step. From the premises, we can deduce propositions: {propositions}"
"""
    if infer_history:
        prompt += f""""Recall the Propositions Deduction history": "{infer_history}"
"""
    
    prompt += """"Reasoning": "Let us think step by step,"""
    return prompt

# 7. condition_select_score (1, 2, 3)
conditions_scores_examples = [
    {
        'determinate_premise': 'C need to be kept.',
        'indeterminate_premise': 'If you keep statement A, you must keep statement B and statement C. If you keep statement D, you must delete both statement E and statement C.Statement A is important information and cannot be deleted.Statement E and statement F should be saved at the same time.',
        'Hypothesis': 'D and C need to be kept',
        'last_history': 'In the last round, we use this "most relevant premise": "If you keep statement A, you must keep statement B and statement C."and got a "New Proposition": If you keep statement D, you must delete both statement E and statement C.',
        'explanation': 'From the determinate_premise, select the "Most relevant premise" which has the same subject with "Hypothesis",for this premise it is C.',
        'Most_relevant_premise': 'C need to be kept.(0.25)',
        'Other_premises_scores': 'If you keep statement A, you must keep statement B and statement C.(0.25)  If you keep statement D, you must delete both statement E and statement C.(0.25) Statement A is important information and cannot be deleted.(0.0) Statement E and statement F should be saved at the same time.(0.0)',
        'Results': 'If you keep statement D, you must delete both statement E and statement C. If you keep statement A, you must keep statement B and statement C.C need to be kept.'
    },
    {
        'determinate_premise': 'The bear is big. Bear is blue.',
        'indeterminate_premise': 'The tiger is rough. If bear is big, then bear is red. If someone is big, then they are nice.',
        'Hypothesis': 'bear is rough.',
        'last_history': 'There\'s no Last_reasoning_history yet, because this is the first derivation.',
        'Most_relevant_premise': 'The bear is big.(0.25)',
        'Other_premises_scores': 'If bear is big, then bear is red.(0.8)The tiger is rough.(0.0) If someone is big, then they are nice.(0.55)',
        'Results': 'The bear is big.If someone is big, then they are nice.If bear is big, then bear is red.',
        'explanation': 'The scores of "If bear is big, then bear is red." is 0.8, because they have the same noun and adjective +0.55, and "The bear is big." is the premise of "If bear is big, then bear is red."+0.25, so the score is 0.8.'
    }
]

def get_condition_select_score_1_prompt(determinate_premise, indeterminate_premise, hypothesis, last_history):
    prompt = f"{SYSTEM_PROMPT}\n"
    prompt += """Read and analyze the "determinate_premise" and "indeterminate_premise" first, then selecting several premises from them. 
Read the "Last_reasoning_history".If we got a "false Proposition" in history,when you select "Most_relevant_premise",do not choose the same "Most relevant premise" in history as your answer.
Please follow these steps:
1.From the determinate_premise, select the "Most relevant premise" which has the same subject with "Hypothesis", and give a score from 0 to 1.
2.You need to assess how the "Most relevant premise" relates to all the other "determinate_premise" and "indeterminate_premise",based on Relevance scoring rules.
3.The "determinate_premise" and "indeterminate_premise" with scores higher than 0.25 will be used as the final results, along with Most_relevant_premise.
Relevance scoring rules:
1. When scoring relevance, 0.25 added for each noun or 0.3 added for each adjective that is the same between two sentences.
2. Scores start to accumulate from 0 points, and the upper limit is 1 point.
3. If sentence p1 is a hypothetical premise of sentence p2,then add 0.25 to p2. for example: measure "if A then B." and "A is true." Then add 0.25 points to "if A then B".
----\n"""
    for ex in conditions_scores_examples:
        prompt += f"""---
"determinate_premise": "{ex['determinate_premise']}"
"indeterminate_premise": "{ex['indeterminate_premise']}"
"Hypothesis": "{ex['Hypothesis']}"
"Last_reasoning_history": "{ex['last_history']}"
Can you select the premise from the "determinate_premises" that scores the highest score for Relevance scoring rules to the "hypothesis"?
"Most_relevant_premise": "{ex['Most_relevant_premise']}"
Can you assess how the "Most relevant premise" relates to all the other "determinate_premise" and "indeterminate_premise" accoding to Relevance scoring rules?
"Other_premises_scores": "{ex['Other_premises_scores']}"
"Results": "{ex['Results']}"
"""
    prompt += f"""---
"determinate_premise": "{determinate_premise}"
"indeterminate_premise": "{indeterminate_premise}"
"Hypothesis": "{hypothesis}"
"Last_reasoning_history": "{last_history}"
Can you select the premise from the "determinate_premises" that scores the highest score for Relevance scoring rules to the "hypothesis"?
"Most_relevant_premise": """
    return prompt

# 8. useful_deduction
useful_deduction_examples = [
    {'Premise': 'Miroslav Venhoda, who published a book in 1946 called Method of Studying Gregorian Chant, is a musician as he is a choral conductor.',
     'Hypothesis': 'A Czech person wrote a book in 1946.',
    'Explanation': 'This premise and Hypothesis contain the same noun(book and 1946), and it is not in the structure of "if..." or "if...then...".',
     'usefulness': 'True'},
    {'Premise': 'All rabbits are cute.',
     'Hypothesis': 'Rock is a turtle or cute.',
    'Explanation': 'This premise and Hypothesis contain the same adjective(cute), and it is not in the structure of "if..." or "if...then...".',
     'usefulness': 'True'},
    {'Premise': 'No animals are paper.',
     'Hypothesis': 'Sea eel is an eel.',
    'Explanation': 'This premise is not in the structure of "if..." or "if...then...",but it has no same noun or adjective with Hypothesis.',
     'usefulness': 'False'},
    {'Premise': 'If no animals are paper, then there is no paper.',
     'Hypothesis': 'Sea eel is an eel.',
    'Explanation': 'This premise has no same noun or adjective with Hypothesis, and it is in the structure of "if...then...".',
     'usefulness': 'False'},
    {'Premise': 'If sea eel is an animal,it is an eel.',
     'Hypothesis': 'Sea eel is an eel.',
    'Explanation': 'This premise has the same noun(sea eel) with Hypothesis, but it is in the structure of "if...".',
     'usefulness': 'False'}
]

def get_useful_deduction_prompt(premise, hypothesis):
    prompt = f"{SYSTEM_PROMPT}\n"
    prompt += """First, read and analyze the following definition:
Determinate_premise: The premise contains the same noun or adjective as the Hypothesis,and the premise is not in the structure of "if..." or "if...then...".
Second, read and analyze the "Premise" and "Hypothesis" .Judge "Premise" is "determinate_premise" or not.
Third,please make sure your classification decisions are derived directly from definitions, rather than unsourced common sense.
----\n"""
    for ex in useful_deduction_examples:
        prompt += f"""---
"Premise": "{ex['Premise']}"
"Hypothesis": "{ex['Hypothesis']}"
"Explanation": "{ex['Explanation']}"
"Judgement": "Is this "Premise" a "determinate_premise" or not?{ex['usefulness']}"
"""
    prompt += f"""---
"Premise": "{premise}"
"Hypothesis": "{hypothesis}"
"Explanation": """
    return prompt

# 9. premise_classification
premise_classification_examples = [
    {'Premises': 'If people perform in school talent shows often, then they attend and are very engaged with school events. All people who are inactive and disinterested members of their community chaperone high school dances.  Bonnie either both attends and is very engaged with school events and is a student who attends the school, or she neither attends and is very engaged with school events nor is a student who attends the school.',
     'Hypothesis': 'Bonnie performs in school talent shows often.',
    'determinate_premise': 'If people perform in school talent shows often, then they attend and are very engaged with school events. Bonnie either both attends and is very engaged with school events and is a student who attends the school, or she neither attends and is very engaged with school events nor is a student who attends the school.',
    'indeterminate_premise': 'All people who are inactive and disinterested members of their community chaperone high school dances.',
    'Explanation': 'If people perform in school talent shows often, then they attend and are very engaged with school events.(intersection:school talent shows) Bonnie either both attends and is very engaged with school events and is a student who attends the school, or she neither attends and is very engaged with school events nor is a student who attends the school.(intersection:Bonnie)All people who are inactive and disinterested members of their community chaperone high school dances.(No intersection)'},
    {'Premises': 'No animals are paper. Sea eel is an animal.Paper is not an animal.',
     'Hypothesis': 'Sea eel is an eel.',
    'determinate_premise': 'Sea eel is an animal.',
    'indeterminate_premise': 'No animals are paper. Paper is not an animal.',
    'Explanation': 'Sea eel is an animal.(intersection:Sea eel) No animals are paper.(No intersection) Paper is not an animal.(No intersection)'}
]

def get_premise_classification_prompt(premises, hypothesis):
    prompt = f"{SYSTEM_PROMPT}\n"
    prompt += """Read and analyze the "Premises" and "Hypothesis" first.Please divide "Premises" into "determinate_premise" and "indeterminate_premise".
If the object discussed by "Premises" and "Hypothesis "has an intersection, then the Premise should be considered as "determinate_premise", otherwise it is "indeterminate_premise".
Please divide "Premises" into "determinate_premise" and "indeterminate_premise" according to "Hypothesis".
----\n"""
    for ex in premise_classification_examples:
        prompt += f"""---
"Premises": "{ex['Premises']}"
"Hypothesis": "{ex['Hypothesis']}"
Can you divide "Premises" into "determinate_premise" and "indeterminate_premise" according to "Hypothesis"?
"determinate_premise": "{ex['determinate_premise']}"
"indeterminate_premise": "{ex['indeterminate_premise']}"
"""
    prompt += f"""---
"Premises": "{premises}"
"Hypothesis": "{hypothesis}"
Can you divide "Premises" into "determinate_premise" and "indeterminate_premise" according to "Hypothesis"?
"determinate_premise": """
    return prompt
