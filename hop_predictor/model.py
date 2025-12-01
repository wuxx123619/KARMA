import srsly
import json
import torch  
from torch.utils.data import DataLoader, TensorDataset  
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup  
from sklearn.model_selection import train_test_split  

def hop_search(edgs,i,hops):
    for j in range(len(edgs)):
        if edgs[j]['start'] == i or edgs[j]['end'] == i:
            if edgs[j]['start'] == i:
                rel_ent = edgs[j]['end']
            else:
                rel_ent = edgs[j]['start']

            if rel_ent == 0:
                hops += 1
                return hops
            else:
                hops += 1
                return hop_search(edgs, rel_ent, hops)


def data_extract(path): #Process the data to return the input and the hop
    claim = {}
    relation = {}
    entity = {}
    hop = {}
    inputs = {}
    count = 0
    with open(path, 'r') as file:
        data_list = json.load(file)

    for data in data_list:
        funct = data['function']
        if funct == 'none':
            edgs = data['graph_query']['edges']
            nodes = data['graph_query']['nodes']
            for i in range(len(nodes)-1):
                claim[count] = data['question']
                input = data['question'] + "[sep]" + nodes[i+1]['friendly_name']
                inputs[count] = input
                entity[count] = nodes[i+1]['friendly_name']
                nei_num = 0
                relations = {}
                for j in range(len(edgs)):
                    if edgs[j]['start'] == i+1 or edgs[j]['end'] == i+1:
                        relations[nei_num] = edgs[j]['relation']
                        nei_num += 1
                hops = hop_search(edgs, i+1, 0)
                relation[count] = relations
                hop[count] = hops
                count += 1
    return inputs, hop

if __name__ == '__main__':
    train_input, train_hop = data_extract("data\GrailQA\grailqa_v1.0_train.json") #Grail_qa train data path
    dev_input, dev_hop = data_extract("data\GrailQA\grailqa_v1.0_dev.json") #Grail_qa dev data path

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  
  
    # Encode the text
    train_encodings = tokenizer(train_text, truncation=True, padding=True, return_tensors='pt', max_length=128)  
    val_encodings = tokenizer(dev_input, truncation=True, padding=True, return_tensors='pt', max_length=128)
    
    # Convert the tags to PyTorch tensors  
    train_labels = torch.tensor(train_hop)  
    val_labels = torch.tensor(dev_hop)  
    
    # Create a data loader 
    train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], train_labels)
    val_dataset = TensorDataset(val_encodings['input_ids'], val_encodings['attention_mask'], val_labels)
    
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)  
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)  

    # Load the pre-trained BERT model
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)  
    class_weights = torch.tensor([3.0, 1.0])
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # Define the optimizer and the learning rate scheduler 
    optimizer = AdamW(model.parameters(), lr=1e-6, eps=2e-8)  
    epochs = 20
    num_training_steps = len(train_dataloader) * epochs  
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)  
    
    # train the model  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    model.to(device)  
    loss_fn.to(device)
    # class_weights = class_weights.to(device)
    # train_labels = train_labels.to(device)
    # val_labels = val_labels.to(device)

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
        print(f'Epoch {epoch+1}/{epochs}, Train_Loss: {train_loss:}, Val_Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}')  

    torch.save(model.state_dict(), 'model_weights.pth')
