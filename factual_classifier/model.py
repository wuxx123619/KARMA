import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
import json

def prepare_data(): #Preprocess the data for training and testing
    train_objects = []
    test_objects = []
    train_texts = []
    train_labels = []
    val_texts = []
    val_labels = []
    with open('data/FactCHD/raw_train.json', 'r') as file: #Obtain the training data of FactCHD
        for line in file:
            try:
                json_object = json.loads(line)
                train_objects.append(json_object)
            except json.JSONDecodeError:
                print(f"Failed to decode JSON on line: {line}")
    with open('data/FactCHD/raw_test.json', 'r') as file: #Obtain the test data of FactCHD
        for line in file:
            try:
                json_object = json.loads(line)
                test_objects.append(json_object)
            except json.JSONDecodeError:
                print(f"Failed to decode JSON on line: {line}")

    for train_object in train_objects:
        if train_object['type'] == 'kg':
            evidence = train_object['evidence']
            input = train_object['response'] + '[sep]' + str(evidence)
            if train_object['label'] == 'FACTUAL':
                label = 1
            else:
                label = 0
            train_texts.append(input)
            train_labels.append(label)
    for test_object in test_objects:
        if test_object['type'] == 'kg':
            evidence = test_object['evidence']
            input = test_object['response'] + '[sep]' + str(evidence)
            if test_object['label'] == 'FACTUAL':
                label = 1
            else:
                label = 0
            val_texts.append(input)
            val_labels.append(label)
    return train_texts, train_labels, val_texts, val_labels
if __name__ == '__main__':
    train_texts, train_labels, val_texts, val_labels = prepare_data()
    # Load BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Encode the text
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors='pt', max_length=128)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, return_tensors='pt', max_length=128)

    # Convert the tags to PyTorch tensors
    train_labels = torch.tensor(train_labels)
    val_labels = torch.tensor(val_labels)

    # Create a data loader
    train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], train_labels)
    val_dataset = TensorDataset(val_encodings['input_ids'], val_encodings['attention_mask'], val_labels)

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Load the pre-trained BERT model and add a binary classification header
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased', num_labels=2)

    # Define the optimizer and the learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=5e-5, eps=1e-7)
    epochs = 30
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

        # Evaluate the model
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
        print(
            f'Epoch {epoch + 1}/{epochs}, Train_Loss: {train_loss:}, Val_Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}')

        torch.save(model.state_dict(), 'model_weights.pth')



