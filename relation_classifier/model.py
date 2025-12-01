import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
import jsonx

import json

with open('kb.json', 'r') as file:
    kb_data = json.load(file)



def prepare_data(data_objects):
    data_text = []
    data_label = []
    for data_object in data_objects:
        ents = []
        test_type = data_object['type']
        domain = data_object['domain']
        if text_type == 'kg' and domain != 'medicine':
            question = data_object['query']
            evidences = data_object['evidence']
            if evidences[0] == '[':
                pass
            else:
                for triplet in evidences:
                    try:
                        ent1 = triplet[0]
                        rel = triplet[1]
                        ent2 = triplet[2]
                    except IndexError:
                        continue
                    if ent1 not in ents and ent1[0] != '[':
                        ents.append(ent1)
                        input = question + '[sep]' + ent1 + '[sep]' + rel
                        if input not in data_text:
                            data_text.append(input)
                            data_label.append(1)
                    elif ent1 not in ents and ent1[0] == '[':
                        try:
                            ent_1h =  ast.literal_eval(ent1)
                        except SyntaxError:
                            continue
                        ent11 = ent_1h[0]
                        if ent11 not in ents:
                            ents.append(ent11)
                        rel1 = ent_1h[1]
                        ent12 = ent_1h[2]
                        if ent12 not in ents:
                            ents.append(ent12)
                        input1 = question + '[sep]' + ent11 + '[sep]' + rel1
                        if input1 not in data_text:
                            data_text.append(input1)
                            data_label.append(1)
                        input2 = question + '[sep]' + ent12 + '[sep]' + rel1
                        if input2 not in data_text:
                            data_text.append(input2)
                            data_label.append(1)
                    if ent2 not in ents and ent2[0] != '[':
                        ents.append(ent2)
                        input = question + '[sep]' + ent2 + '[sep]' + rel
                        if input not in data_text:
                            data_text.append(input)
                            data_label.append(1)
                    elif ent2 not in ents and ent1[0] == '[':
                        try:
                            ent_2h =  ast.literal_eval(ent2)
                        except SyntaxError:
                            continue
                        ent21 = ent_2h[0]
                        if ent21 not in ents:
                            ents.append(ent21)
                        rel2 = ent_2h[1]
                        ent22 = ent_2h[2]
                        if ent22 not in ents:
                            ents.append(ent22)
                        rel2 = ent_2h[1]
                        ent22 = ent_2h[2]
                        input3 = question + '[sep]' + ent21 + '[sep]' + rel2
                        if input3 not in data_text:
                            data_text.append(input3)
                            data_label.append(1)
                        input4 = question + '[sep]' + ent22 + '[sep]' + rel2
                        if input4 not in data_text:
                            data_text.append(input4)
                            data_label.append(1)
        Qid_list = kb_data['entities']
        for ent in ents:
            for qid in Qid_list:
                evi_data = kb_data['entities'][qid]
                ent_name = evi_data['name']
                if ent == ent_name:
                    count_n = 0
                    count_w = 0
                    rel_l = []
                    for data in evi_data['attributes']:
                        rel = data['key']
                        if rel not in rel_l:
                            rel_l.append(rel)
                    for data in evi_data['relations']:
                        rel = data['relation']
                        if rel not in rel_l:
                            rel_l.append(rel)
                    if len(rel_l) == 0:
                        continue
                    while count_n < 5:
                        if count_w >= 8:
                            break
                        count_w += 1
                        random_rel = random.choice(rel_l)
                        input = question + '[sep]' + ent + '[sep]' + random_rel
                        if input not in data_text:
                            data_text.append(input)
                            data_label.append(0)
                            count_n += 1
    return data_text,data_label

def get_data(path):
    train_objects = [] 
    with open(path, 'r') as file:  
    for line in file:  
        try:  
            json_object = json.loads(line)  
            train_objects.append(json_object)  
        except json.JSONDecodeError:  
            print(f"Failed to decode JSON on line: {line}")




if __name__ == '__main__':
    train_objects = get_data("data\FactCHD\raw_train.json") #FactCHD's train data
    test_objects = get_data("data\FactCHD\raw_test.json") #FactCHD's test data

    train_text,train_label = prepare_data(train_objects)
    test_text,test_label = prepare_data(test_objects)


    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Encode the text
    train_encodings = tokenizer(train_text, truncation=True, padding=True, return_tensors='pt', max_length=128)
    val_encodings = tokenizer(test_text, truncation=True, padding=True, return_tensors='pt', max_length=128)

    # Convert the tags to PyTorch tensors
    train_labels = torch.tensor(train_label)
    val_labels = torch.tensor(test_label)

    # Create a data loader
    train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], train_labels)
    val_dataset = TensorDataset(val_encodings['input_ids'], val_encodings['attention_mask'], val_labels)

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Load the pre-trained BERT model and add a binary classification header
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased', num_labels=2)
    class_weights = torch.tensor([3.0, 1.0])
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # Define the optimizer and the learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=5e-5, eps=5e-7)
    epochs = 10
    num_training_steps = len(train_dataloader) * epochs
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    # train the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss_fn.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_dataloader:
            # batch = {k: v.to(device) for k, v in batch.items()}
            # outputs = model(**batch, labels=batch['labels'])
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            # loss = outputs.loss
            # train_loss += loss
            # loss.backward()
            logits = outputs.logits
            logits = logits.to(device)
            labels = labels.to(device)
            loss = loss_fn(logits, labels)
            train_loss += loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        train_loss /= len(train_dataloader)
        print(len(train_dataloader))

        # evaluate the model
        model.eval()
        val_loss, val_accuracy = 0, 0
        with torch.no_grad():
            for batch in val_dataloader:
                # batch = {k: v.to(device) for k, v in batch.items()}
                # outputs = model(**batch, labels=batch['labels'])
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                labels = batch[2].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()
                val_preds = torch.argmax(outputs.logits, dim=-1)
                val_accuracy += (val_preds == labels).sum().item()
        val_loss /= len(val_dataloader)
        val_accuracy /= len(val_dataset)
        print(len(val_dataloader))
        print(
            f'Epoch {epoch + 1}/{epochs}, Train_Loss: {train_loss:}, Val_Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}')

    torch.save(model.state_dict(), 'model_weights_fchd.pth')

