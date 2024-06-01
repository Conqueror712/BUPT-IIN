import torch
from transformers import BertForSequenceClassification, BertTokenizer

# Load model and tokenizer
# model_path = "/root/.cache/huggingface/hub/bert-base-uncased/bert-base-uncased"  
model_path = "./bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# Save model
model.save_pretrained("model")

# Load model
loaded_model = BertForSequenceClassification.from_pretrained("model")

# Evaluate function
def evaluate_sentence_pair(sentence1, sentence2):
    encoded_input = tokenizer(sentence1, sentence2, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        output = loaded_model(**encoded_input)
        logits = output.logits
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
    return predicted_class

# Demo
sentence1 = "An apple a day keeps a docter away."
sentence2 = "A banana a day keeps a docter away."
predicted_label = evaluate_sentence_pair(sentence1, sentence2)
print("Predicted Label:", predicted_label)
