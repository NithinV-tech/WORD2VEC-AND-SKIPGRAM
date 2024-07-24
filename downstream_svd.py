import torch
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from build_svd import CoOccurrenceMatrixOptimized
import matplotlib.pyplot as plt
import seaborn as sns

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

svd_data = torch.load('svd-word-vectors_context10.pth',map_location=DEVICE)
vocab = svd_data['vocab']
word_vectors = svd_data['word_vectors']


class TextDataset(Dataset):
    def __init__(self, descriptions, encoded_labels, vocab, word_vectors):
        self.descriptions = descriptions
        self.encoded_labels = torch.tensor(encoded_labels, dtype=torch.long)
        self.vocab = vocab  
        self.word_vectors = torch.tensor(word_vectors, dtype=torch.float) 
    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, idx):
        words = self.descriptions.iloc[idx]
        vectors = [self.word_vectors[self.vocab[word]] if word in self.vocab else torch.zeros(100) for word in words]
        vectors_tensor = torch.stack(vectors)  
        return vectors_tensor, self.encoded_labels[idx], vectors_tensor.shape[0]
    

class LSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text, text_lengths):
        packed = pack_padded_sequence(text, text_lengths, batch_first=True)
        _, (hidden, _) = self.lstm(packed)
        out = self.fc(hidden[-1])
        return out
    

def custom_collate_fn(batch):
    batch.sort(key=lambda x: x[2], reverse=True)  
    sequences, labels, lengths = zip(*batch)
    sequences_padded = pad_sequence(sequences, batch_first=True)
    labels = torch.stack(labels)
    lengths = torch.tensor(lengths)
    return sequences_padded, labels, lengths

def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for texts, labels, lengths in loader:  
        texts, labels = texts.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(texts, lengths) 
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate_model(model, loader, device):
    model.eval()
    total_correct = 0
    total_count = 0
    with torch.no_grad():
        for texts, labels, lengths in loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts, lengths.cpu())  
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_count += labels.size(0)
    accuracy = total_correct / total_count
    return accuracy * 100 

def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

def predict_labels(model, loader, device):
    model.eval()
    actuals = []
    predictions = []
    with torch.no_grad():
        for texts, labels, lengths in loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts, lengths.cpu())  
            _, predicted = torch.max(outputs, 1)
            actuals.extend(labels.view(-1).cpu().numpy())
            predictions.extend(predicted.cpu().numpy())
    return actuals, predictions


def compute_performance_metrics(actuals, predictions,class_names):
    accuracy = accuracy_score(actuals, predictions)
    f1 = f1_score(actuals, predictions, average='weighted')
    precision = precision_score(actuals, predictions, average='weighted')
    recall = recall_score(actuals, predictions, average='weighted')
    conf_matrix = confusion_matrix(actuals, predictions)
    
    print(f'Accuracy: {accuracy:.2f}')
    print(f'F1 Score: {f1:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print('Confusion Matrix:')
    #print(conf_matrix)
    plot_confusion_matrix(conf_matrix, class_names)


if __name__ == "__main__":

    data = pd.read_csv("train.csv", encoding='utf-8')
    test_data = pd.read_csv("test.csv", encoding='utf-8')
    data['processed_text'] = data['Description'].apply(lambda x: word_tokenize(x.lower()) if pd.notnull(x) else [])
    #corpus = data['processed_text'].explode().tolist()
    test_data['processed_text'] = test_data['Description'].apply(lambda x: word_tokenize(x.lower()) if pd.notnull(x) else [])
    corpus = data['processed_text'].explode().tolist()
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(data['Class Index'])
    encoded_labels_test = label_encoder.fit_transform(test_data['Class Index'])

    dataset = TextDataset(data['processed_text'], encoded_labels, vocab, word_vectors)
    test_dataset = TextDataset(test_data['processed_text'], encoded_labels_test, vocab, word_vectors)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=custom_collate_fn)
    test_loader =  DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=custom_collate_fn)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMClassifier(100, 128, len(label_encoder.classes_)).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    for epoch in range(2):
        loss = train(model, train_loader, optimizer, criterion,device)
        print(f'Epoch: {epoch+1}, Loss: {loss:.4f}')
    
    torch.save(model.state_dict(), 'final_lstm_model_5.pth')
    print("Model saved as final_lstm_model.pth")
        
    test_accuracy = evaluate_model(model, test_loader, device)
    print(f'Test Accuracy: {test_accuracy:.2f}%')

    train_actuals, train_predictions = predict_labels(model, train_loader, DEVICE)
    test_actuals, test_predictions = predict_labels(model, test_loader, DEVICE)
    print("Training Set Performance:")
    compute_performance_metrics(train_actuals, train_predictions,label_encoder.classes_)

    print("\nTest Set Performance:")
    compute_performance_metrics(test_actuals, test_predictions,label_encoder.classes_)




