import json
from tqdm import tqdm
import re
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
import torch

rel_pass = ['ISBN-13','ISBN-10','ISNI','CANTIC-ID','Libris-URI','OCLC control number','exploitation visa number','Munzinger IBA']
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
with open('kb.json', 'r') as file:
    kb_data = json.load(file)
with open('qid2ent.json', 'r') as file:    
    qid2ent = json.load(file)

def load_model(path,num_label):
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_label)
    model.load_state_dict(torch.load(path),strict=False)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    model.to(device)
    return model


def hop_predictor(input): 
    inputs = tokenizer(input, return_tensors="pt", padding=True, truncation=True, max_length=512)  
    input_ids = inputs['input_ids'].to(device)  
    attention_mask = inputs['attention_mask'].to(device)
    with torch.no_grad():  
        outputs = hop_model(input_ids, attention_mask=attention_mask)
    predicted_class_idx = torch.argmax(outputs.logits, dim=-1) 
    return predicted_class_idx.item()+1

def data_classifier(model,input):
    inputs = tokenizer(input, return_tensors="pt", padding=True, truncation=True, max_length=512)  
    input_ids = inputs['input_ids'].to(device)  
    attention_mask = inputs['attention_mask'].to(device)
    with torch.no_grad():  
        outputs = rel_model(input_ids, attention_mask=attention_mask)
    # softmax = Softmax(dim=1)
    probabilities = torch.softmax(outputs.logits,dim=-1)
    predicted_class_idx = torch.argmax(outputs.logits, dim=-1)
    return predicted_class_idx.item(),probabilities[0][1].item()

hop_predictor = classifier.load_model('hop_predictor/model_weights.pth', 4)#define hop predictor
rel_classifier = classifier.load_model('relation_classifier/model_weights.pth', 2)#define relation calssifier
ent_classifier = classifier.load_model('entity_classifier/model_weights.pth', 2) #define entity classifier
factual_classifier = classifier.load_model('factual_classifier/model_weights.pth', 2) #define factual classifier

def kg_retrieve(question,ent,ent_list1,hop,subgraph,ent_searched,deep): #Iteratively retrieve subgraphs
    if ent in ent_searched or deep >= 3:
        return 0
    else:
        ent_searched.append(ent)
        Qid_list = kb_data['entities']
        lowercase_ent = ent.lower().strip()
        for qid in Qid_list:
            evi_data = kb_data['entities'][qid]
            ent_name = evi_data['name']
            if lowercase_ent == ent_name.lower():
                rels = {}
                rel_l1 = {}
                for data in evi_data['attributes']:
                    rel = data['key']
                    if rel in rel_pass:
                        pass
                    else:
                        keys_list = list(rel_l1.keys())
                        if rel in keys_list:
                            pred = rel_l1[rel]
                        else:
                            input = question + '[sep]' + ent_name + '[sep]' + rel
                            pred,pp = rel_classifier(input)
                            rel_l1[rel] = pred
                            rels[rel] = pp
                        if pred == 1: #Filter the relationship path
                            tg_ent = data['value']['value']
                            tg_f,pp = ent_classifier(str(ent_list1),tg_ent) 
                            if tg_f == 1: # Filter entities
                                triplet = (ent_name,rel,tg_ent)
                                items = data['qualifiers'].items()
                                for key, value in items:
                                    if len(key) != 0:
                                        triplet = (triplet, key, value[0]['value'])
                                if triplet not in subgraph:
                                    subgraph.append(triplet)
                rel_l2 = {}
                for data in evi_data['relations']:
                    relation = data['relation']
                    if relation in rel_pass:
                        pass
                    else:
                        keys_list = list(rel_l2.keys())
                        if relation in keys_list:
                            pred = rel_l2[relation]
                        else:
                            input = question + '[sep]' + ent_name + '[sep]' + relation
                            pred,pp = rel_classifier(input)
                            rel_l2[relation] = pred
                            rels[relation] = pp
                        if pred == 1:
                            qid = data['object']
                            tg_ent = qid2ent[qid]
                            tg_f,pp = ent_classifier(str(ent_list1),tg_ent)
                            if tg_f == 1:
                                # print("---------------------------")
                                # print(input,tg_ent,hop)
                                triplet = (ent_name, relation, tg_ent)
                                triplet1 = (tg_ent, relation, ent_name)
                                items = data['qualifiers'].items()
                                for key, value in items:
                                    if len(key) != 0:
                                        triplet = (triplet, key, value[0]['value'])
                                        triplet1 = (triplet1, key, value[0]['value'])
                                if triplet not in subgraph and triplet1 not in subgraph:
                                    subgraph.append(triplet)
                                if hop > 1:
                                    deep += 1
                                    hop = hop_predictor(question + '[sep]' + str(tg_ent))
                                    search_wikikb(question,tg_ent,ent_list1,hop,subgraph,ent_searched,deep)
                count_ent = 0
                for value in rel_l1.values():
                    if value == 1:
                        count_ent += 1
                for value in rel_l2.values():
                    if value == 1:
                        count_ent += 1
                if count_ent <= 1:
                    sorted_items = sorted(rels.items(), key=lambda x: x[2])
                    top_two_keys = [item[0] for item in sorted_items[-5:]]
                    for k in top_two_keys:
                        if rels[k] >0.0001:
                            for data in evi_data['attributes']:
                                rel = data['key']
                                if rel in top_two_keys:
                                    tg_ent = data['value']['value']
                                    tg_f,pp = ent_classifier(str(ent_list1),tg_ent) #用微调bert筛选子图
                                    if tg_f == 1:
                                        triplet = (ent_name,rel,tg_ent)
                                        items = data['qualifiers'].items()
                                        for key, value in items:
                                            if len(key) != 0:
                                                triplet = (triplet, key, value[0]['value'])
                                        if triplet not in subgraph:
                                            subgraph.append(triplet)
                            for data in evi_data['relations']:
                                relation = data['relation']
                                if relation in top_two_keys:
                                    qid = data['object']
                                    tg_ent = qid2ent[qid]
                                    tg_f,pp = ent_classifier(str(ent_list1),tg_ent)
                                    if tg_f == 1:
                                        # print("---------------------------")
                                        # print(input,tg_ent,hop)
                                        triplet = (ent_name, relation, tg_ent)
                                        triplet1 = (tg_ent, relation, ent_name)
                                        items = data['qualifiers'].items()
                                        for key, value in items:
                                            if len(key) != 0:
                                                triplet = (triplet, key, value[0]['value'])
                                                triplet1 = (triplet1, key, value[0]['value'])
                                        if triplet not in subgraph and triplet1 not in subgraph:
                                            subgraph.append(triplet)

def reason_pred(answer,subgraph,label):
    input = answer + '[sep]' + str(subgraph)
    pred,pp = factual_classifier(input)
    return pred==label


def FactCHD():# Use KARMA on the FactCHD dataset
    ent_list = []  
    with open('ent list path', 'r') as file:  
        for line in file:  
            try:  
                json_object = json.loads(line)  
                ent_list.append(json_object)  
            except json.JSONDecodeError:  
                print(f"Failed to decode JSON on line: {line}")
    ent_list1 = []
    for i in range(len(ent_list[0])):
        ents = []
        for j in range(len(ent_list[0][i])):
            ent = ent_list[0][i][j].strip('*').strip('"')
            ents.append(ent.split('(')[0])
        ent_list1.append(ents)
    
    count_kg = 0
    count_true = 0
    count_wiki = 0
    subgraphs = []
    test_objects = []  
    with open('data/FactCHD/raw_test.json', 'r') as file:  
        for line in file:  
            try:  
                json_object = json.loads(line)  
                test_objects.append(json_object)  
            except json.JSONDecodeError:  
                print(f"Failed to decode JSON on line: {line}")
    for i in tqdm(range(len(test_objects))):
        if test_objects[i]['type'] == 'kg':
            count_kg += 1
            domain = test_objects[i]['domain']
            if domain != 'medicine':
                count_wiki += 1
                question = test_objects[i]['query']
                answer = test_objects[i]['response']
                subgraph = []
                ent_searched = []
                for ent in ent_list1[count_wiki-1]:
                    input = question + '[sep]' + ent
                    hop = hop_predictor(input)
                    search_wikikb(question,ent,ent_list1[count_wiki-1],hop,subgraph,ent_searched,1)
                subgraphs.append(subgraph)
    count_wiki = 0
    count_true = 0
    count1_1 = 0
    count_1 = 0
    count1_2 = 0
    count_2 = 0
    count1_3 = 0
    count_3 = 0
    for i in tqdm(range(len(test_objects))):
        if test_objects[i]['type'] == 'kg':
            domain = test_objects[i]['domain']
            subgraph_type = test_objects[i]['subgraph_type']
            if domain != 'medicine' or subgraph_type != "all":
                count_wiki += 1
                question = test_objects[i]['query']
                answer = test_objects[i]['response']
                evidence = subgraphs[count_wiki-1]
                if subgraph_type == "multi_hop_reasoning":
                    count1_1 += 1
                elif subgraph_type == "set_operation":
                    count1_2 += 1
                elif subgraph_type == "quantitative_comparison":
                    count1_3 += 1
                if test_objects[i]['label'] == 'FACTUAL':
                    label = 1
                else: 
                    label = 0
                pred = reason_pred(answer,evidence,label)
                if pred == 1:
                    count_true += 1
                    if subgraph_type == "multi_hop_reasoning":
                        count_1 += 1
                    elif subgraph_type == "set_operation":
                        count_2 += 1
                    elif subgraph_type == "quantitative_comparison":
                        count_3 += 1
    print("true number:" count_true, "count all:" count_wiki)
    print("multi_hop_reasoning": count_1, "count": count1_1)
    print("set_operation": count_2, "count": count1_2)
    print("quantitative_comparison": count_3, "count": count1_3)

def MutiHallu(): # Use KARMA on the MutiHallu dataset
    test_objects = []  
    with open('data/MutiHallu/MutiHallu.json', 'r') as file:  
        for line in file:  
            try:  
                json_object = json.loads(line)  
                test_objects.append(json_object)  
            except json.JSONDecodeError:  
                print(f"Failed to decode JSON on line: {line}")

    ent_list = []  
    with open('ent list path', 'r') as file:  
        for line in file:  
            try:  
                json_object = json.loads(line)  
                ent_list.append(json_object)  
            except json.JSONDecodeError:  
                print(f"Failed to decode JSON on line: {line}")

    ent_list1 = []
    for i in range(len(ent_list[0])):
        ents = []
        for j in range(len(ent_list[0][i])):
            ent = ent_list[0][i][j].strip('*').strip('"')
            ents.append(ent.split('(')[0])
        ent_list1.append(ents)
    subgraphs = []
    for i in tqdm(range(len(test_objects[0]))):
        question = test_objects[0][i]['question']
        answer = test_objects[0][i]['answer']
        label = test_objects[0][i]['label']
        subgraph = []
        ent_searched = []
        for j in range(len(ent_list1[i])):
            input = question + '[sep]' + ent_list1[i][j]
            hop = hop_classifier(input)
            search_wikikb1(question,ent_list1[i][j],ent_list1[i],hop,subgraph,ent_searched)
        subgraphs.append(subgraph)
    count_true = 0
    for i in tqdm(range(len(test_objects[0]))):
        question = test_objects[0][i]['question']
        answer = test_objects[0][i]['answer']
        label = test_objects[0][i]['label']
        subgraph = test_objects[0][i]['subgraph']
        evidence = subgraphs[i]
        pred = reason_pred(answer,evidence,label)
        if pred == 1:
            count_true += 1
    print("true number:" count_true, "count": len(test_objects[0]))




if __name__ == '__main__':
    subgraph1 = FactCHD()
    subgraph2 = MutiHallu()

