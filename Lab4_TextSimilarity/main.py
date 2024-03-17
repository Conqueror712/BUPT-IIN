import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import functional as F
from tqdm import tqdm

# Load data
def read_mrpc_dataset(file_path):
    df = pd.read_csv(file_path, sep='\t', quoting=3)
    return df

# Compute evaluation metrics
def load_bert_model(model_name, num_labels):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return tokenizer, model

# Preprocess data
def preprocess_data(df, tokenizer, max_length):
    input_ids = []
    attention_masks = []

    for sent1, sent2 in zip(df['#1 String'], df['#2 String']):
        encoded_dict = tokenizer.encode_plus(
                            sent1,
                            sent2,
                            add_special_tokens = True,
                            max_length = max_length,
                            padding = 'max_length',
                            truncation=True,
                            return_attention_mask = True,
                            return_tensors = 'pt',
                    )

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    return input_ids, attention_masks


def main():
    # Parameters setting
    model_name = 'bert-base-uncased'
    max_length = 128
    batch_size = 32
    num_epochs = 3
    learning_rate = 5e-5
    num_labels = 2  # Binary classification
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    df_train = read_mrpc_dataset('mrpc_data/train.tsv')
    df_dev = read_mrpc_dataset('mrpc_data/dev.tsv')

    # Load BERT model and tokenizer
    tokenizer, model = load_bert_model(model_name, num_labels)
    model.to(device)

    # Preprocess data
    input_ids_train, attention_masks_train = preprocess_data(df_train, tokenizer, max_length)
    input_ids_dev, attention_masks_dev = preprocess_data(df_dev, tokenizer, max_length)
    labels_train = torch.tensor(df_train['Quality'])
    labels_dev = torch.tensor(df_dev['Quality'])

    # Define DataLoader
    train_data = TensorDataset(input_ids_train, attention_masks_train, labels_train)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # Define optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Train the model
    model.train()
    for epoch in range(num_epochs):
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1} / {num_epochs}'):
            input_ids, attention_masks, labels = batch
            input_ids, attention_masks, labels = input_ids.to(device), attention_masks.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask=attention_masks, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        input_ids_dev, attention_masks_dev, labels_dev = input_ids_dev.to(device), attention_masks_dev.to(device), labels_dev.to(device)
        outputs = model(input_ids_dev, attention_mask=attention_masks_dev)
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=1)
        predicted_labels = torch.argmax(probabilities, dim=1).cpu().numpy()

    # Compute evaluation metrics and print the results
    accuracy, precision, recall, f1 = compute_metrics(df_dev['Quality'].values, predicted_labels)
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')

if __name__ == "__main__":
    main()
