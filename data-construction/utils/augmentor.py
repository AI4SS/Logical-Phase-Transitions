import json
import random
from copy import deepcopy
from typing import List, Dict

import numpy as np
from openai import OpenAI
from tqdm.auto import tqdm


class DataAugmentor:
    
    def __init__(self, args) -> None:
        self.args = args
        
        self.model_name = args.model_name
        if args.api_key == "EMPTY" and args.base_url == "EMPTY":
            self.client = OpenAI()
        else:
            self.client = OpenAI(
                api_key=args.api_key,
                base_url=args.base_url
            )
        self.err_cnt = 0
        
    def step_augment(self, data: List, shuffled: bool, has_noise1: float, has_noise2: float, name_list: List, start: int, end: int) -> List:
        """
        1. Introduce noise
        2. Generate reasoning chain and then create problems at each step
        3. Return the result
        """
        unmatched_context = 0
        current_dataset = []
        err_cnt = 0
        
        cnt = -1
        pbar = tqdm(data)
        for item in pbar:
            cnt += 1
            if cnt < start or cnt >= end:
                continue
            
            # symbolic representation
            nl2fol = {}
            for i in range(len(item['facts'])):
                nl2fol[item['facts'][i]] = item['facts_fol'][i]
            for i in range(len(item['rules'])):
                nl2fol[item['rules'][i]] = item['rules_fol'][i]
            for i in range(len(item['distracting_facts'])):
                nl2fol[item['distracting_facts'][i]] = item['distracting_facts_fol'][i]
            for i in range(len(item['distracting_rules'])):
                nl2fol[item['distracting_rules'][i]] = item['distracting_rules_fol'][i]
            
            base_context = deepcopy(item['context'])
            err_flag = False
            
            # 1: introduce noise
            noise_list = self.get_noise(item=item, base_context=base_context, name_list=name_list, has_noise1=has_noise1, has_noise2=has_noise2)

            # augment problems
            accumulated_context = []
            accumulated_reasoning_chains = ""
            accumulated_reasoning_chains_fol = ""
            
            # problem quality control
            check_context = []
            for chain in item['reasoning_chains']:
                for fact in chain['facts']:
                    if fact in base_context:
                        check_context.append(fact)
                
                if chain['rules'] is not None:
                    check_context.append(chain['rules'])
            
            try:
                assert set(base_context) == set(check_context)
            except:
                unmatched_context += 1
                continue
            
            # 2: generate reasoning chains
            for chain_idx in range(len(item['reasoning_chains'])):
                current_problem = {'id': len(current_dataset)}
                final_conclusion = None
                final_conclusion_fol = None
                
                chain = item['reasoning_chains'][chain_idx]
                chain_fol = item['reasoning_chains_fol'][chain_idx]
                if chain['conclusion'] is None or chain == item['reasoning_chains'][-1]:  # in step augmentation, we ignore the last step
                    err_flag = True
                    break
                else:
                    # get facts
                    for i in range(len(chain['facts'])):
                        if item['name'].lower() in chain_fol['facts'][i].lower():
                            if item['name'] not in chain['facts'][i]:  # quality control
                                err_cnt += 1
                                err_flag = True
                                break
                        
                        accumulated_reasoning_chains += f"fact{i + 1}: {chain['facts'][i]}\n"
                        # parallel FOL facts
                        try:
                            fol_fact = chain_fol['facts'][i]
                        except Exception:
                            fol_fact = ""
                        accumulated_reasoning_chains_fol += f"fact{i + 1}: {fol_fact}\n"
                        if chain['facts'][i] in base_context:
                            accumulated_context.append(chain['facts'][i])
                    
                    # get rules
                    if chain['rules'] is not None:
                        if item['name'].lower() in chain_fol['rules'].lower():  # quality control
                            if item['name'] not in chain['rules']:
                                err_cnt += 1
                                err_flag = True
                                break
                    
                        accumulated_reasoning_chains += f"rule: {chain['rules']}\n"
                        # parallel FOL rule
                        accumulated_reasoning_chains_fol += f"rule: {chain_fol.get('rules', '')}\n"
                        accumulated_context.append(chain['rules'])
                        
                        # conclusion
                        if item['name'].lower() in chain_fol['conclusion'].lower():
                            if item['name'] not in chain['conclusion']:
                                err_cnt += 1
                                err_flag = True
                                break
                        
                        accumulated_reasoning_chains += f"conclusion: {chain['conclusion']}\n\n"
                        # parallel FOL conclusion
                        accumulated_reasoning_chains_fol += f"conclusion: {chain_fol.get('conclusion', '')}\n\n"
                    else:
                        accumulated_reasoning_chains += "\n"
                        assert chain == item['reasoning_chains'][-1]
                    
                    current_problem['options'] = random.sample([["A) True", "B) False"], ["A) True", "B) False", "C) Uncertain"]], 1)[0]
                    current_question = "Based on the above information, is the following statement true, false, or uncertain? " if current_problem['options'] == ["A) True", "B) False", "C) Uncertain"] else "Based on the above information, is the following statement true, or false? "
                    
                    current_anwer = random.sample(['True', 'False'], 1)[0]
                    current_problem['answer'] = 'A' if current_anwer == "True" else "B"
                    
                    if current_problem['answer'] == 'A':
                        current_question += chain['conclusion']
                        current_problem['reasoning'] = accumulated_reasoning_chains + f"Therefore, it is true that {chain['conclusion']} The correct option is: A."
                        current_problem['reasoning_fol'] = accumulated_reasoning_chains_fol + f"Therefore, it is true that {chain_fol.get('conclusion', '')} The correct option is: A."
                        final_conclusion = chain['conclusion']
                        final_conclusion_fol = chain_fol.get('conclusion', "")
                    else:
                        # get the negated conclusion
                        negation_message = [
                            {'role': 'system', 'content': "You are a language expert skilled in transforming sentences into their negative forms. Your answer should in JSON format with key: negation"},
                            {'role': 'user', 'content': "Sapphire is conventional."},
                            {'role': 'assistant', 'content': "{\n  \"negation\": \"Sapphire is not conventional.\"\n}"},
                            {'role': 'user', 'content': "Brantley does not conduct research."},
                            {'role': 'assistant', 'content': "{\n  \"negation\": \"Brantley conducts research.\"\n}"},
                            {'role': 'user', 'content': chain['conclusion']}
                        ]
                        negated_conclusion = self.send_request(messages=negation_message, key='negation')
                        current_question += negated_conclusion
                        final_conclusion = negated_conclusion
                        fol_conclusion = chain_fol.get('conclusion', "")
                        final_conclusion_fol = f"¬({fol_conclusion})" if fol_conclusion else ""
                            
                        current_problem['reasoning'] = accumulated_reasoning_chains + f"Therefore, it is false that {negated_conclusion} The correct option is: B."
                        current_problem['reasoning_fol'] = accumulated_reasoning_chains_fol + f"Therefore, it is false that {final_conclusion_fol} The correct option is: B."
                        
                    current_problem['question'] = current_question

                    if final_conclusion is None:
                        final_conclusion = chain['conclusion']
                    if final_conclusion_fol is None:
                        final_conclusion_fol = chain_fol.get('conclusion', "")

                    current_problem['conclusion'] = final_conclusion
                    current_problem['conclusion_fol'] = final_conclusion_fol
                
                # get context
                current_problem['context'] = deepcopy(accumulated_context)
                
                remaining_context = []  # the remaining context can be used as distractions
                for b_context in base_context:
                    if b_context not in accumulated_context:
                        remaining_context.append(b_context)

                if len(remaining_context) != 0:
                    sampled_base_context_num = random.sample(range(len(remaining_context)), 1)[0]
                    current_problem['context'].extend(remaining_context[:sampled_base_context_num])
                
                if len(noise_list) != 0:
                    sampled_noise_num = random.sample(range(len(noise_list)), 1)[0]
                    current_problem['context'].extend(noise_list[:sampled_noise_num])
                
                if shuffled:
                    random.shuffle(current_problem['context'])
                
                current_problem['context'] = " ".join(current_problem['context'])
                
                # 添加context_fol逻辑
                context_fol_list = []
                for context_item in accumulated_context:
                    if context_item in nl2fol:
                        context_fol_list.append(nl2fol[context_item])
                    else:
                        context_fol_list.append("")  # 如果找不到对应的FOL，添加空字符串
                current_problem['context_fol'] = "\n".join(context_fol_list)
                
                # symbolic representation, background story and keywords
                current_problem['nl2fol'] = deepcopy(nl2fol)
                current_problem['background_story'] = deepcopy(item['background_story'])
                current_problem['name'] = deepcopy(item['name'])
                current_problem['keyword'] = deepcopy(item['keyword'])
                current_problem['subject_category'] = deepcopy(item['subject_category'])
                
                current_dataset.append(current_problem)
            
            if err_flag:
                continue
        
        print(unmatched_context)
        return current_dataset
    
    def uncertain_augment(self, data: List, shuffled: bool, has_noise1: float, has_noise2: float, name_list: List, start: int, end: int) -> List:
        """
        1. Load word list and introduce noise
        2. Generate reasoning chain and then create problems at each step
        3. Return the result
        """
        unmatched_context = 0
        current_dataset = []
        
        # load word list
        with open(self.args.predicate_file, 'r') as f:
            word_list = json.load(f)['words']
        
        cnt = -1
        pbar = tqdm(data)
        for item in pbar:
            cnt += 1
            if cnt < start or cnt >= end:
                continue
            
            # symbolic representation
            nl2fol = {}
            for i in range(len(item['facts'])):
                nl2fol[item['facts'][i]] = item['facts_fol'][i]
            for i in range(len(item['rules'])):
                nl2fol[item['rules'][i]] = item['rules_fol'][i]
            for i in range(len(item['distracting_facts'])):
                nl2fol[item['distracting_facts'][i]] = item['distracting_facts_fol'][i]
            for i in range(len(item['distracting_rules'])):
                nl2fol[item['distracting_rules'][i]] = item['distracting_rules_fol'][i]
                
            base_context = deepcopy(item['context'])
            err_flag = False
            
            noise_list = self.get_noise(item=item, base_context=base_context, name_list=name_list, has_noise1=has_noise1, has_noise2=has_noise2)
            
            # start augmenting
            accumulated_context = []
            accumulated_reasoning_chains = ""
            
            # quality control
            check_context = []
            for chain in item['reasoning_chains']:
                for fact in chain['facts']:
                    if fact in base_context:
                        check_context.append(fact)

                if chain['rules'] is not None:
                    check_context.append(chain['rules'])
            
            try:
                assert set(base_context) == set(check_context)
            except:
                unmatched_context += 1
                continue
            
            # Break down each step of reasoning to generate a new problem with uncertain answer
            for chain_idx in range(len(item['reasoning_chains'])):
                current_problem = {'id': len(current_dataset)}
                final_conclusion = None
                final_conclusion_fol = None
                
                chain = item['reasoning_chains'][chain_idx]
                chain_fol = item['reasoning_chains_fol'][chain_idx]
                
                if chain['conclusion'] is None or chain == item['reasoning_chains'][-1]:
                    break  # uncertain problems is not needed when performing uncertainty augmentation
                else:
                    current_problem['answer'] = "C"
                    current_problem['options'] = ["A) True", "B) False", "C) Uncertain"]
                    
                    # get facts
                    for i in range(len(chain['facts'])):
                        if item['name'].lower() in chain_fol['facts'][i].lower():
                            if item['name'] not in chain['facts'][i]:
                                err_flag = True
                                break
                            
                        accumulated_reasoning_chains += f"fact{i + 1}: {chain['facts'][i]}\n"
                        if chain['facts'][i] in base_context:
                            accumulated_context.append(chain['facts'][i])
                    
                    # get rules
                    if chain['rules'] is None:
                        err_flag = True
                        break
                        
                    if item['name'].lower() in chain_fol['rules'].lower():
                        if item['name'] not in chain['rules']:
                            err_flag = True
                            break
                    
                    accumulated_reasoning_chains += f"rule: {chain['rules']}\n"
                    accumulated_context.append(chain['rules'])
                    
                    # conclusion
                    if item['name'].lower() in chain_fol['conclusion'].lower():
                        if item['name'] not in chain['conclusion']:
                            err_flag = True
                            break
                    accumulated_reasoning_chains += f"conclusion: {chain['conclusion']}\n\n"
                    
                    # there are two types of uncertain conclusions: replaced subject and completely unrelated facts or rules
                    uncertain_type = np.random.choice(
                        a=np.array([0, 1]),
                        size=1,
                        replace=True,
                        p=[0.7, 0.3]
                    )
                    if item['name'] not in chain['conclusion']:
                        uncertain_type = 1
                        print("Found Universal Conclusions")
                        
                    if uncertain_type == 0:  # replaced names
                        noise_name = random.sample(name_list, 1)[0]
                        while noise_name == item['name']:
                            noise_name = random.sample(name_list, 1)[0]
                        noise_conclusion = chain['conclusion'].replace(item['name'], noise_name)
                        
                        current_question = "Based on the above information, is the following statement true, false, or uncertain? "
                        current_question += noise_conclusion
                        current_problem['question'] = current_question
                        final_conclusion = noise_conclusion
                        final_conclusion_fol = ""

                        current_problem['reasoning'] = accumulated_reasoning_chains + f"According to the context and the conclusions that have already been drawn, we can deduce that it is true that {chain['conclusion']} But it is uncertain that {noise_conclusion} The correct option is: C."
                    else:  # completely unrelated facts or rules
                        unrelated_message = [
                            {'role': 'system', 'content': "You are a language expert skilled in transforming a keyword and name into a statement about a fact or a commonse rule. Your answer should be simple and natural with no more than 10 words. Your answer should in JSON format with key: statement"},
                            {'role': 'user', 'content': "keyword: conventional\nname: Sapphire"},
                            {'role': 'assistant', 'content': "{\n  \"statement\": \"Sapphire is a conventional person.\"\n}"},
                            {'role': 'user', 'content': "keyword: research\nname: Brantley"},
                            {'role': 'assistant', 'content': "{\n  \"statement\": \"If Brantley conducts research, then he is a researcher or a student.\"\n}"},
                            {'role': 'user', 'content': f"keyword: {random.sample(word_list, 1)[0]}\nname: {item['name']}"}
                        ]
                        unrelated_conclusion = self.send_request(unrelated_message, key='statement')    
                        current_question = "Based on the above information, is the following statement true, false, or uncertain? "
                        current_question += unrelated_conclusion
                        current_problem['question'] = current_question
                        final_conclusion = unrelated_conclusion
                        final_conclusion_fol = ""
                        
                        current_problem['reasoning'] = accumulated_reasoning_chains + f"According to the context and the conclusions that have already been drawn, we can deduce that it is uncertain that {unrelated_conclusion} The correct option is: C."
                    
                    # get context
                    current_problem['context'] = deepcopy(accumulated_context)

                    remaining_context = []
                    for b_context in base_context:
                        if b_context not in accumulated_context:
                            remaining_context.append(b_context)
                            
                    if len(remaining_context) != 0:  # The premises that were not selected can be used as a distraction.
                        sampled_base_context_num = random.sample(range(len(remaining_context)), 1)[0]
                        current_problem['context'].extend(remaining_context[:sampled_base_context_num])
                    
                    if len(noise_list) != 0:
                        sampled_noise_num = random.sample(range(len(noise_list)), 1)[0]
                        current_problem['context'].extend(noise_list[:sampled_noise_num])
                        
                    if shuffled:
                        random.shuffle(current_problem['context'])
                        
                    current_problem['context'] = " ".join(current_problem['context'])
                    
                    # symbolic representation, background story and keywords
                    current_problem['nl2fol'] = deepcopy(nl2fol)
                    current_problem['background_story'] = deepcopy(item['background_story'])
                    current_problem['name'] = deepcopy(item['name'])
                    current_problem['keyword'] = deepcopy(item['keyword'])
                    current_problem['subject_category'] = deepcopy(item['subject_category'])

                    if final_conclusion is None:
                        final_conclusion = chain['conclusion']
                    if final_conclusion_fol is None:
                        final_conclusion_fol = chain_fol.get('conclusion', "")

                    current_problem['conclusion'] = final_conclusion
                    current_problem['conclusion_fol'] = final_conclusion_fol
                
                    current_dataset.append(current_problem)
                    
            if err_flag:
                continue
            
        return current_dataset
                    
    def normal_generation(self, data: List, shuffled: bool, has_noise1: float, has_noise2: float, name_list: List, start: int, end: int) -> List:
        """
        1. Introduce noise
        2. Generate reasoning chain and then create problems
        3. Return the result
        """
        current_dataset = []
        err_cnt = 0
        
        # creating problems
        cnt = -1
        pbar = tqdm(data)
        for item in pbar:
            cnt += 1
            if cnt < start or cnt >= end:
                continue
            
            base_context = deepcopy(item['context'])
            err_flag = False
            
            noise_list = self.get_noise(item=item, base_context=base_context, name_list=name_list, has_noise1=has_noise1, has_noise2=has_noise2)
            
            # create problems
            context = []
            reasoning_chains = ""
            reasoning_chains_fol = ""
            current_problem = {'id': len(current_dataset)}
            final_conclusion = None
            final_conclusion_fol = None
            
            for chain_idx in range(len(item['reasoning_chains'])):
                chain = item['reasoning_chains'][chain_idx]
                chain_fol = item['reasoning_chains_fol'][chain_idx]
                if chain['conclusion'] is None:
                    if chain == item['reasoning_chains'][-1]:
                        current_problem['options'] = ["A) True", "B) False", "C) Uncertain"]
                        assert item['answer'] == "Uncertain"
                        current_problem['answer'] = "C"
                        
                        current_question = "Based on the above information, is the following statement true, false, or uncertain? "
                        current_question += item['conclusion']
                        current_problem['question'] = current_question
                        
                        current_problem['reasoning'] = reasoning_chains + f"According to the context and the conclusions that have already been drawn, we can deduce that it is uncertain that {item['conclusion']} The correct option is: C."
                        # build a parallel FOL reasoning string for uncertain case
                        final_conclusion = item.get('conclusion', "")
                        final_conclusion_fol = item.get('conclusion_fol', "")
                        reasoning_chains_fol += f"conclusion: {final_conclusion_fol}\n\n"
                        current_problem['reasoning_fol'] = reasoning_chains_fol + f"uncertain: {final_conclusion_fol} The correct option is: C."
                    else:
                        continue
                else:
                    # facts
                    for i in range(len(chain['facts'])):
                        if item['name'].lower() in chain_fol['facts'][i].lower():
                            if item['name'] not in chain['facts'][i]:
                                err_cnt += 1
                                err_flag = True
                                break
                        
                        reasoning_chains += f"fact{i + 1}: {chain['facts'][i]}\n"
                        # parallel FOL facts
                        try:
                            fol_fact = chain_fol['facts'][i]
                        except Exception:
                            fol_fact = ""
                        reasoning_chains_fol += f"fact{i + 1}: {fol_fact}\n"
                        if chain['facts'][i] in base_context:
                            context.append(chain['facts'][i])

                    # rules
                    if chain['rules'] is None:
                        err_cnt += 1
                        err_flag = True
                        break
                    
                    if item['name'].lower() in chain_fol['rules'].lower():
                        if item['name'] not in chain['rules']:
                            err_cnt += 1
                            err_flag = True
                            break

                    reasoning_chains += f"rule: {chain['rules']}\n"
                    # parallel FOL rule
                    reasoning_chains_fol += f"rule: {chain_fol.get('rules', '')}\n"
                    context.append(chain['rules'])
                    
                    # conclusion
                    if item['name'].lower() in chain_fol['conclusion'].lower():
                        if item['name'] not in chain['conclusion']:
                            err_cnt += 1
                            err_flag = True
                            break
                        
                    current_problem['options'] = ["A) True", "B) False", "C) Uncertain"]
                    
                    if chain == item['reasoning_chains'][-1]:
                        current_anwer = item['answer']
                        assert current_anwer != 'Uncertain'
                        
                        current_problem['answer'] = 'A' if current_anwer == "True" else "B"
                        
                        if current_anwer == "False":
                            negation_message = [
                                {'role': 'system', 'content': "You are a language expert skilled in transforming sentences into their negative forms. Your answer should in JSON format with key: negation"},
                                {'role': 'user', 'content': "Sapphire is conventional."},
                                {'role': 'assistant', 'content': "{\n  \"negation\": \"Sapphire is not conventional.\"\n}"},
                                {'role': 'user', 'content': "Brantley does not conduct research."},
                                {'role': 'assistant', 'content': "{\n  \"negation\": \"Brantley conducts research.\"\n}"},
                                {'role': 'user', 'content': chain['conclusion']}
                            ]
                            negated_conclusion = self.send_request(messages=negation_message, key='negation')
                            # fallback to original if API not used
                            if not negated_conclusion:
                                negated_conclusion = chain['conclusion']
                            reasoning_chains += f"conclusion: {negated_conclusion}\n\n"
                            # FOL: mark negated conclusion
                            fol_conc = chain_fol.get('conclusion', "")
                            reasoning_chains_fol += f"conclusion: ¬({fol_conc})\n\n"
                        else:
                            reasoning_chains += f"conclusion: {chain['conclusion']}\n\n"
                            reasoning_chains_fol += f"conclusion: {chain_fol.get('conclusion', '')}\n\n"
                        
                        current_question = "Based on the above information, is the following statement true, false, or uncertain? "
                        current_question += chain['conclusion']
                        current_problem['question'] = current_question
                        
                        current_problem['reasoning'] = reasoning_chains + f"Therefore, it is {current_anwer.lower()} that {chain['conclusion']} The correct option is: {current_problem['answer']}."
                        current_problem['reasoning_fol'] = reasoning_chains_fol + f"Therefore, it is {current_anwer.lower()} that {chain_fol.get('conclusion', '')} The correct option is: {current_problem['answer']}."
                        final_conclusion = chain['conclusion']
                        final_conclusion_fol = chain_fol.get('conclusion', "")
                    else:
                        reasoning_chains += f"conclusion: {chain['conclusion']}\n\n"
                        reasoning_chains_fol += f"conclusion: {chain_fol.get('conclusion', '')}\n\n"
                        
            if err_flag:
                continue 
            if final_conclusion is None:
                final_conclusion = item.get('conclusion', "")
            if final_conclusion_fol is None:
                final_conclusion_fol = item.get('conclusion_fol', "")

            current_problem['conclusion'] = final_conclusion
            current_problem['conclusion_fol'] = final_conclusion_fol
                
            # get current context
            current_problem['context'] = deepcopy(context)
            # 添加这行：提取 context_fol 并用换行符连接
            current_problem['context_fol'] = "\n".join(item['context_fol'])
            if len(noise_list) != 0:
                sampled_noise_num = random.sample(range(len(noise_list)), 1)[0]
                current_problem['context'].extend(noise_list[:sampled_noise_num])

                # Add corresponding FOL noise sentences to context_fol
                # Create nl2fol mapping here since it's needed for noise FOL conversion
                nl2fol = {}
                for i in range(len(item['facts'])):
                    nl2fol[item['facts'][i]] = item['facts_fol'][i]
                for i in range(len(item['rules'])):
                    nl2fol[item['rules'][i]] = item['rules_fol'][i]
                for i in range(len(item['distracting_facts'])):
                    nl2fol[item['distracting_facts'][i]] = item['distracting_facts_fol'][i]
                for i in range(len(item['distracting_rules'])):
                    nl2fol[item['distracting_rules'][i]] = item['distracting_rules_fol'][i]

                noise_fol_list = []
                for noise_sentence in noise_list[:sampled_noise_num]:
                    if noise_sentence in nl2fol:
                        noise_fol_list.append(nl2fol[noise_sentence])
                    else:
                        continue  # Skip if no FOL mapping found

                # Combine existing context_fol with noise FOL sentences
                if noise_fol_list:
                    current_problem['context_fol'] += "\n" + "\n".join(noise_fol_list)
             
            #    random.shuffle(current_problem['context'])
                
            current_problem['context'] = " ".join(current_problem['context'])
            
            # symbolic representation
            nl2fol = {}
            for i in range(len(item['facts'])):
                nl2fol[item['facts'][i]] = item['facts_fol'][i]
            for i in range(len(item['rules'])):
                nl2fol[item['rules'][i]] = item['rules_fol'][i]
            for i in range(len(item['distracting_facts'])):
                nl2fol[item['distracting_facts'][i]] = item['distracting_facts_fol'][i]
            for i in range(len(item['distracting_rules'])):
                nl2fol[item['distracting_rules'][i]] = item['distracting_rules_fol'][i]
            
            current_problem['nl2fol'] = nl2fol
            
            # background story and keywords
            current_problem['background_story'] = item['background_story']
            current_problem['name'] = item['name']
            current_problem['keyword'] = item['keyword']
            current_problem['subject_category'] = item['subject_category']
 
            current_dataset.append(current_problem)
            
        return current_dataset
    
    def get_noise(self, item: Dict, base_context: List, name_list: List, has_noise1: float, has_noise2: float) -> List:
        """ Introduce distractions """
        # get distraction 2
        d_premises_list = deepcopy(item['distracting_facts'])
        d_premises_list.extend(deepcopy(item['distracting_rules']))
        
        if has_noise2 == 1:
            noise2_rate = self.sample_rate()
            noise2_num = round(len(d_premises_list) * noise2_rate)
            noise2_list = random.sample(d_premises_list, noise2_num)
        else:
            noise2_list = []
                
        # get distraction 1
        if has_noise1 == 1:
            noise1_raw_list = deepcopy(base_context)
            random.shuffle(noise1_raw_list)
            noise1_list = []
                
            sampled_name_list = random.sample(name_list, 3)
                
            for i in range(len(noise1_raw_list)):
                if item['name'] in noise1_raw_list[i]:
                    selected_name = random.sample(sampled_name_list, 1)[0]
                    while selected_name == item['name']:
                        selected_name = random.sample(sampled_name_list, 1)[0]
                        
                    noise1_list.append(noise1_raw_list[i].replace(item['name'], selected_name))
        else:
            noise1_list = []
                
        noise_list = deepcopy(noise1_list)
        noise_list.extend(deepcopy(noise2_list))
            
        return noise_list
        
    def send_request(self, messages: List, key: str) -> str:
        while True:
            api_flag = False
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.6
                )
                answer_str = completion.choices[0].message.content.replace("```json", "").replace("```", "")
                result = eval(answer_str)[key]
                api_flag = True
            except:
                self.err_cnt += 1
                print(f"API error occured, wait for 2 seconds. Error count: {self.err_cnt}")
            
            if api_flag:
                break
        
        return result
            
    @staticmethod
    def sample_rate():
        mean = 0.5
        std_dev = 0.16
        sampled_rate = np.random.normal(loc=mean, scale=std_dev, size=1)[0]
    
        if sampled_rate > 1:
            sampled_rate = 1
        elif sampled_rate < 0:
            sampled_rate = 0
        else:
            sampled_rate = sampled_rate
        
        return sampled_rate
    
    
    
    
    
    
    