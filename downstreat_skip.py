
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from collections import Counter
from scipy.spatial.distance import cosine
import json
from nltk.stem import PorterStemmer
import string
import re
import nltk
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch import nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


ps = PorterStemmer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]

    words = [word for word in stripped if word.isalpha()]
   
    return words



class custom_skip:
    def __init__(self, sentences, vector_size=100, window=5, negative=5, learning_rate=0.01, epochs=5):
        self.sentences = sentences
        self.vector_size = vector_size
        self.window = window
        self.negative = negative
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.vocab = {}
        self.words = []
        self.W1 = None
        self.W2 = None

    
   

    def build_vocab(self):
        word_counts = Counter()
        for sentence in self.sentences:
            word_counts.update(sentence)
        self.words = list(word_counts.keys())
        self.vocab = {word: idx for idx, word in enumerate(self.words)}
        self.vocab_size = len(self.words)

    def initialize_weights(self):
        self.W1 = np.random.uniform(-0.5, 0.5, (self.vocab_size, self.vector_size))
        self.W2 = np.random.uniform(-0.5, 0.5, (self.vector_size, self.vocab_size))

    def train(self):
       for epoch in range(self.epochs):
          negative_samples_precomputed = {}
          for word_idx in range(self.vocab_size):
             negative_samples_for_word = []
             while len(negative_samples_for_word) < self.negative:
                 random_sample = np.random.randint(0, self.vocab_size)
                 if random_sample != word_idx:
                   negative_samples_for_word.append(random_sample)
             negative_samples_precomputed[word_idx] = negative_samples_for_word
    

          for sentence in self.sentences:
              for i, center_word in enumerate(sentence):
                  center_word_index = self.vocab[center_word]
                  context_indices = set()
                  for j in range(max(0, i - self.window), min(len(sentence), i + self.window + 1)):
                        if j != i:
                            context_word = sentence[j]
                            context_indices.add(self.vocab[context_word])

                  for context_word_index in context_indices:
                        self.update_weights(center_word_index, context_word_index, 1)

                  for negative_word_index in negative_samples_precomputed[center_word_index]:
                    if negative_word_index not in context_indices:
                        self.update_weights(center_word_index, negative_word_index, 0)
          print(f"Epoch {epoch+1} complete")

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def update_weights(self, center_word_index, context_word_index, label):
        center_word_vector = self.W1[center_word_index]
        context_word_vector = self.W2[:, context_word_index]
        score = np.dot(center_word_vector, context_word_vector)
        predicted = self.sigmoid(score)
        gradient = (predicted - label)
        gradient = gradient* self.learning_rate
        self.W1[center_word_index] -= gradient * context_word_vector
        self.W2[:, context_word_index] -= gradient * center_word_vector

    def get_embedding(self, word):
        if word in self.vocab:
          idx = self.vocab[word]
          embedding = self.W1[idx]
          return embedding 
        

    
    def most_similar(self, word, topn=5):
         word_embedding = self.get_embedding(word)
         if word_embedding is None:
           return []

         similarities = []

 
         for w in self.words:
            if w != word: 
                w_embedding = self.get_embedding(w)
                if w_embedding is not None: 
                    similarity = 1 - cosine(word_embedding, w_embedding)
                    similarities.append((w, similarity))

         similarities.sort(key=lambda x: x[1], reverse=True)
         return similarities[:topn]


   

    def save_model(self, model_path="word2vec_model.pth"):
        W1_tensor = torch.tensor(self.W1, dtype=torch.float32)
        W2_tensor = torch.tensor(self.W2, dtype=torch.float32)
        
        model_state = {
            "W1": W1_tensor,
            "W2": W2_tensor,
            "vocab": self.vocab,
            "vector_size": self.vector_size,
            "window": self.window
        }
        
        torch.save(model_state, model_path)       
        print("Model saved successfully.")


    def load_model(self, model_path="word2vec_model_context10.pth"):   
        model_state = torch.load(model_path)
        
        self.W1 = model_state["W1"].numpy()  
        self.W2 = model_state["W2"].numpy()
        self.vocab = model_state["vocab"]
        self.vector_size = model_state.get("vector_size", 100) 
        self.window = model_state.get("window", 10)   #2
        self.words = list(self.vocab.keys())  
        
        print("Model loaded successfully.")
        return self.vocab




model = custom_skip(sentences=[], vector_size=100, window=10, negative=5, epochs=5) 
vocab =model.load_model("word2vec_model.pth")  

# print(model.most_similar('massacre'))
# print(model.most_similar('government'))
# print(model.most_similar('house'))
# print(model.most_similar('world'))
# print(model.most_similar('peak'))
# print(model.most_similar('Russian'))

###################################################################333
train_data = pd.read_csv("train.csv", encoding='utf-8')
#train_data['processed_text'] = train_data['Description'].apply(lambda x: word_tokenize(x.lower()) if pd.notnull(x) else [])
train_data['processed_text'] = train_data['Description'].apply(lambda x: preprocess_text(x) if pd.notnull(x) else [])
test_data = pd.read_csv("test.csv", encoding='utf-8')
test_data['processed_text'] = test_data['Description'].apply(lambda x: preprocess_text(x) if pd.notnull(x) else [])
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(train_data['Class Index'])
word_vector_matrix = model.W1  
vocab = model.vocab


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

def custom_collate_fn(batch):
    batch.sort(key=lambda x: x[2], reverse=True)
    sequences, labels, lengths = zip(*batch)
    sequences_padded = pad_sequence(sequences, batch_first=True)
    labels = torch.stack(labels)
    lengths = torch.tensor(lengths)
    return sequences_padded, labels, lengths


train_dataset = TextDataset(train_data['processed_text'], encoded_labels, vocab, word_vector_matrix)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=custom_collate_fn)

encoded_test_labels = label_encoder.transform(test_data['Class Index'])
test_dataset = TextDataset(test_data['processed_text'], encoded_test_labels, vocab, word_vector_matrix)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=custom_collate_fn)


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

def compute_performance_metrics(actuals, predictions, class_names):
    accuracy = accuracy_score(actuals, predictions)
    f1 = f1_score(actuals, predictions, average='weighted')
    precision = precision_score(actuals, predictions, average='weighted')
    recall = recall_score(actuals, predictions, average='weighted')
    conf_matrix = confusion_matrix(actuals, predictions)
    
    print(f'Accuracy: {accuracy:.2f}')
    print(f'F1 Score: {f1:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    
  
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMClassifier(100, 128, len(label_encoder.classes_)).to(device)
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

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


for epoch in range(2):  
    loss = train(model, train_loader, optimizer, criterion, device)
    print(f'Epoch: {epoch+1}, Loss: {loss:.4f}')

torch.save(model.state_dict(), 'final_lstm_model_10.pth')   #3
print("Model saved as final_lstm_model_skip_context10.pth") #4


test_accuracy = evaluate_model(model, test_loader, device)
print(f'Test Accuracy: {test_accuracy:.2f}%')

train_actuals, train_predictions = predict_labels(model, train_loader, device)
test_actuals, test_predictions = predict_labels(model, test_loader, device)

class_names = label_encoder.classes_

print("Training Set Performance:")
compute_performance_metrics(train_actuals, train_predictions, class_names)

print("\nTest Set Performance:")
compute_performance_metrics(test_actuals, test_predictions, class_names)
