import json
import random
import ast
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from torch import nn

with open('kb.json', 'r') as file:
    kb_data = json.load(file)

with open('qid2ent.json', 'r') as file:
    qid2ent = json.load(file)

def get_data(path): #Obtain the data of FactCHD
    train_objects = []
    with open('raw_train.json', 'r') as file:
        for line in file:
            try:
                json_object = json.loads(line)
                train_objects.append(json_object)
            except json.JSONDecodeError:
                print(f"Failed to decode JSON on line: {line}")
    return train_objects

def data_generate(data_objects,ent_lists):# Construction of training data
    data_text = []
    data_label = []
    count_n = -1
    for data_object in data_objects:
        ents = []
        count_n += 1
        text_type = data_object['type']
        domain = data_object['domain']
        if text_type == 'kg' and domain != 'medicine':
            evidences = data_object['evidence']
            ent_list = ent_lists[count_n]
            if evidences[0] == '[':
                pass
            else:
                for triplet in evidences:
                    try:
                        ent1 = triplet[0]
                        ent2 = triplet[2]
                    except IndexError:
                        continue
                    if ent1 not in ents and ent1[0] != '[':
                        ents.append(ent1)
                    elif ent1 not in ents and ent1[0] == '[':
                        ent_1h = ast.literal_eval(ent1)
                        ent11 = ent_1h[0]
                        if ent11 not in ents:
                            ents.append(ent11)
                        ent12 = ent_1h[2]
                        if ent12 not in ents:
                            ents.append(ent12)
                    if ent2 not in ents and ent2[0] != '[':
                        ents.append(ent2)
                    elif ent2 not in ents and ent1[0] == '[':
                        ent_2h = ast.literal_eval(ent2)
                        ent21 = ent_2h[0]
                        if ent21 not in ents:
                            ents.append(ent21)
                        ent22 = ent_2h[2]
                        if ent22 not in ents:
                            ents.append(ent22)
        for ent in ents:
            input = str(ent_list) + "[sep]" + ent
            if input not in data_text: #Positive example
                data_text.append(input)
                data_label.append(1)
        for ent in ents:
            count_n = 0
            count_w = 0
            while count_n < 3:
                if count_w >= 5:
                    break
                count_w += 1
                random_ent = random.choice(list(qid2ent.values()))
                input = str(ent_list) + '[sep]' + random_ent
                if input not in data_text: #Negative example
                    data_text.append(input)
                    data_label.append(0)
                    count_n += 1
            random_att = random.choice(list(qid2ent.values()))
            random_number = random.choice([0, 1])
            if random_number == 1:
                input = str(ent_list) + '[sep]' + str(random_att)
                if input not in data_text: #Negative example
                    data_text.append(input)
                    data_label.append(0)
    return data_text, data_label



if __name__ == '__main__':
    train_data = get_data('data/FactCHD/raw_train.json') #Obtain the training data of FactCHD
    test_data = get_data('data/FactCHD/raw_test.json') #Obtain test data of FactCHD
    train_ent_list = get_data('train ent list path') #Obtain the list of subgraphs extracted from the training data
    test_ent_list = get_data('test ent list path') #Obtain the list of subgraphs extracted from the test data
    train_text, train_label = data_generate(train_data, train_ent_list)
    test_text, test_label = data_generate(test_data, test_ent_list)

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
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    class_weights = torch.tensor([1.0, 10.0])
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # Define the optimizer and the learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=5e-5, eps=2e-8)
    epochs = 15
    num_training_steps = len(train_dataloader) * epochs
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    # Train the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

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
            loss = outputs.loss
            train_loss += loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        train_loss /= len(train_dataloader)
        print(len(train_dataloader))

        # Evaluation model
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
                val_pred = torch.argmax(outputs.logits, dim=-1)
                val_accuracy += (val_pred == labels).sum().item()
        val_loss /= len(val_dataloader)
        val_accuracy /= len(val_dataset)
        print(len(val_dataloader))
        print(
            f'Epoch {epoch + 1}/{epochs}, Train_Loss: {train_loss:}, Val_Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}')

    torch.save(model.state_dict(), 'model_weights.pth')
