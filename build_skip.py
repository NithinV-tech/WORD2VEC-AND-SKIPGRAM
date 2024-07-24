
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


   

    def save_model(self, model_path="word2vec_model_context10.pth"):
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


    def load_model(self, model_path="word2vec_model.pth"):
        model_state = torch.load(model_path)
    
        self.W1 = model_state["W1"].numpy()  
        self.W2 = model_state["W2"].numpy()
        self.vocab = model_state["vocab"]
        self.vector_size = model_state.get("vector_size", 100) 
        self.window = model_state.get("window", 10)
        self.words = list(self.vocab.keys()) 
        
        print("Model loaded successfully")
        return self.vocab


###########################################################################################################

train_data = pd.read_csv("train.csv", encoding='utf-8')
train_data['processed_text'] = train_data['Description'].apply(lambda x: preprocess_text(x) if pd.notnull(x) else [])
train_corpus = [sentence for sentence in train_data['processed_text'] if sentence]
model = custom_skip(train_corpus, vector_size=100, window=10, negative=5, epochs=10)
model.build_vocab()
model.initialize_weights()
model.train()
model.save_model()