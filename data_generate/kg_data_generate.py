import json
import random
import re
from tqdm import tqdm
import os
from vllm import LLM, SamplingParams
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
import torch

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.5, top_p=0.9, max_tokens = 1024)

# Create an LLM.
llm = LLM(model="Your llm path")

KG_QA_MULTI_HOP_REASONING_DATA ="""I want you to act as a complex and fluent question and answer data generator, where your task is to generate complex and fluent questions and answers that require multi-hop reasoning based entirely on the given knowledge. Here are some examples:
\"""
<Knowledge>: ["Yao Ming", "spouse", "Ye Li"], ["Ye Li", "educated at", "Shanghai University of Sport"], ["Shanghai University of Sport", "establishment time", "November 1952"]
<Explanation>: To generate a question and answer, we need to perform multi-hop reasoning through the given knowledge. Here is the step-by-step reasoning:
1. Yao Ming's wife is named Ye Li.
2. Ye Li was educated at Shanghai University of Sport.
3. Shanghai University of Sport was established in November 1952.
Based on the above reasoning chain, we can generate a complex question and answer with *3-hop* reasoning capabilities:
<Question>: Please tell me, when was the university that Yao Ming's wife graduated from established?
<Correct answer>: Yao Ming's wife, Ye Li, graduated from Shanghai University of Sport, which was established in *November 1952*.
<Hallucinated answer>: Yao Ming's wife, Ye Li, graduated from Shanghai University of Sport, which was established in *March 1954*.
\"""

\"""
<Knowledge>: ["je marche seul", "lyrics by", "Jean-Jacques Goldman"], ["Jean-Jacques Goldman", "sibling", "robert goldman (songwriter)"], ["robert goldman (songwriter)", "father", "alter mojze goldman"], ["alter mojze goldman", "place of death", "Sport in Paris"], ["Sport in Paris", "flag", "Flag of Paris"]
<Explanation>: To answer the given question, we need to perform multi-hop reasoning through the given triplets. Here is the step-by-step reasoning:
1. The lyricist of Je Marche Seul is "Jean-Jacques Goldman".
2. Jean-Jacques Goldman's sibling is Robert Goldman (songwriter).
3. Robert Goldman's father is Alter Mojze Goldman.
4. Alter Mojze Goldman died in Sport in Paris.
5. The flag of Sport in Paris is the Flag of Paris.
Based on the above reasoning chain, we can generate a complex question and answer with *5-hop* reasoning capabilities:
<Question>: Can you please provide me with the flag of the city where the father of the sibling of the lyricist of "Je Marche Seul" died?
<Correct answer>: The flag of the city where the father of the sibling of the lyricist of "Je Marche Seul" died is the *Flag of Paris*.
<Hallucinated answer>: The flag of the city where the father of the sibling of the lyricist of "Je Marche Seul" died is the *Flag of Germany*.
\"""

*You need to thoroughly study the above example to grasp the meaning of multi-hop reasoning, the core is step-by-step reasoning*. Please generate a complex and fluent question and answer that requires multi-hop reasoning based entirely on the given knowledge below without introducing any prior knowledge or the knowledge from the example. Make sure that the generated <Question> involve multi-hop reasoning and include *all* the information from the <Knowledge>! Answers include <Correct answer> and <Hallucinated answer>. Make ensure that the <Correct answer> can be *correctly* deduced from the <Knowledge> using multi-hop reasoning without relying on unknown or insufficient information. <Hallucinated answer> sounds plausible but is factually incorrect. *To produce <Hallucinated answer>, you should choose one or more of the following strategies: fabricate information to resolve factual contradictions, misunderstand the question context and intention, provide an answer that is either too general or too specific, or employ incorrect reasoning to arrive at a hallucinated answer not supported by the knowledge*. *Follow the example format for output*:
\"""
<Knowledge>: {}
"""


with open('qid2ent.json', 'r') as file:
    # Load the qid and entity name pairs into the dictionary
    qid2ent = json.load(file)

with open('kb.json', 'r') as file:
    #Load the knowledge graph
    kb_data = json.load(file)


def dfs_kqa_pro(eid, subgraph, max_deep, dfs_entity_set, fr="", y=0):
    if len(subgraph) >= max_deep:
        return
    try:
        ent = kb_data['entities'][eid]
    except KeyError:
        return
    # Filter forward relations
    forwards = []
    add = 0
    for idx, rel_info in enumerate(ent['relations']):
        if rel_info['direction'] == 'forward':
            if rel_info['object'] in dfs_entity_set:
                continue
            if fr != "" and y == 1:
                if rel_info['relation'] == fr:
                    dfs_entity_set.add(rel_info['object'])
                    subgraph.append([ent['name'], rel_info['relation'], qid2ent[rel_info['object']]])
                    dfs_kqa_pro(rel_info['object'], subgraph, max_deep, dfs_entity_set, rel_info['relation'], y)
                    return
            forwards.append(idx)

    # Randomly choose a forwards
    if len(forwards) == 0:
        return
    idx = random.randint(0, len(forwards) - 1)
    dfs_entity_set.add(ent['relations'][forwards[idx]]['object'])
    rel_info = ent['relations'][forwards[idx]]
    subgraph.append([ent['name'], rel_info['relation'], qid2ent[rel_info['object']]])
    dfs_kqa_pro(rel_info['object'], subgraph, max_deep, dfs_entity_set, rel_info['relation'], y)

def get_attributes(subgraph):
    search_ent = subgraph[-1][2]
    ent = ""
    for eid in kb_data['entities']:
        if search_ent == kb_data['entities'][eid]['name']:
            ent = kb_data['entities'][eid]
            break
    if ent == "":
        return
    for idx, attr_info in enumerate(ent['attributes']):
        if attr_info['value']['type'] == 'string':
            continue
        if len(attr_info['qualifiers']) > 0:
            for k, v in attr_info['qualifiers'].items():
                if "time" not in k:
                    continue
                subgraph.append([[ent['name'], attr_info['key'], str(attr_info['value'])], k, str(v[0])])
                return

    for idx, attr_info in enumerate(ent['attributes']):
        if attr_info['value']['type'] == 'string':
            continue
        if len(attr_info['qualifiers']) > 0:
            continue
        subgraph.append([ent['name'], attr_info['key'], str(attr_info['value'])])
        return

    for idx, attr_info in enumerate(ent['attributes']):
        if len(attr_info['qualifiers']) > 0:
            continue
        subgraph.append([ent['name'], attr_info['key'], str(attr_info['value'])])
        return

    for idx, attr_info in enumerate(ent['attributes']):
        for k, v in attr_info['qualifiers'].items():
            subgraph.append([[ent['name'], attr_info['key'], str(attr_info['value'])], k, str(v[0])])
            return

def generate_multi_hop_reasoning_subgraph():
    # Implement a dfs algorithm
    head_set = set()
    tot = 0
    file_name = 'your subgraph path' #the file path of the extracted subgraph
    with open(os.path.join(file_name), 'w', encoding='utf-8') as fin:
        while True:
            entity_set = set()
            for eid, ent in kb_data['entities'].items():
                if eid in kb_data['concepts']:
                    continue
                if eid in entity_set or eid in head_set:
                    continue
                entity_set.add(eid)
                head_set.add(eid)
                subgraph = []
                dfs_entity_set = {eid}
                # Randomly generate the numbers
                x = random.randint(2, 5)
                y = random.randint(0, 1)
                dfs_kqa_pro(eid, subgraph, x, dfs_entity_set, "", y)
                if len(subgraph) <= 1:
                    continue

                entity_set = entity_set | dfs_entity_set
                get_attributes(subgraph)
                fin.write(json.dumps({
                    "triples": list(subgraph),
                    "size": len(subgraph),
                    "head": kb_data['entities'][eid]['name']
                }, ensure_ascii=False) + '\n')
                tot += 1
                if tot >= 1200:
                    break
            break
    pass
def reason_generation(subgraph,prompt):
    prompts = prompt.format(subgraph)
    outputs = llm.generate(prompts, sampling_params,use_tqdm=False)
    # Return the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        return generated_text

def get_data(path):
    test_objects = []
    with open(path, 'r') as file:
        for line in file:
            try:
                json_object = json.loads(line)
                test_objects.append(json_object)
            except json.JSONDecodeError:
                print(f"Failed to decode JSON on line: {line}")
    return test_objects

if __name__ == '__main__':
    generate_multi_hop_reasoning_subgraph()
    test_objects = get_data("your subgraph path")
    question_list = []
    correct_answer_list = []
    hallucinated_answer_list = []
    responses = []
    for i in tqdm(range(len(test_objects[0]))):
        evidence = test_objects[0][i]['subgraph']
        response = reason_generation(evidence, KG_QA_MULTI_HOP_REASONING_DATA)
        question_matches = re.findall(r'<Question>:(.*?)(?=<Correct answer>:|$)', response, re.DOTALL)
        if question_matches:
            question = question_matches[-1]
            print("question:", question.strip(' ').strip('\n'))
        correct_answer_match = re.findall(r'<Correct answer>:(.*?)(?=<Hallucinated answer>:|$)', response, re.DOTALL)
        if correct_answer_match:
            correct_answer = correct_answer_match[-1]
            print("correct_answer:", correct_answer.strip(' ').strip('\n'))
        hallucinated_answer_match = list(re.finditer(r'<Hallucinated answer>:', response))
        if hallucinated_answer_match:
            # Get the end position of the last tag
            last_match_end = hallucinated_answer_match[-1].end()
            # Extract all the content after the last tag
            hallucinated_answer = response[last_match_end:].strip()
            print("hallucinated_answer:", hallucinated_answer.strip(' ').split('\n')[0])
        responses.append(response)
        question_list.append(question.strip(' ').strip('\n'))
        correct_answer_list.append(correct_answer.strip(' ').strip('\n'))
        hallucinated_answer_list.append(hallucinated_answer.strip(' ').strip('\n').split('\n')[0])
    subgraph_right = []
    for i in range(len(question_list)):
        if question_list[i] == None or correct_answer_list[i] == None or hallucinated_answer_list[i] == None or \
                question_list[i] == "(You will write the question here)" or question_list[
            i] == "(Please provide the question here)":
            continue
        else:
            label = random.choice([0, 1])
            if label == 0:
                answer = hallucinated_answer_list[i]
            else:
                answer = correct_answer_list[i]
            my_dict = {}
            my_dict = {
                "question": question_list[i],
                    "answer": answer,
                    "label": label,
                    "evdence": responses[i]
            }
            subgraph_right.append(my_dict)
    with open("generated data path.json", "w", encoding="utf-8") as f:
        json.dump(subgraph_right, f) #save generate data

