import pandas as pd
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import svds
from nltk.tokenize import word_tokenize
import numpy as np
import torch

class CoOccurrenceMatrixOptimized:
    def __init__(self, window_size=2, k=100):
        self.window_size = window_size
        self.vocab = {}
        self.co_occurrence_matrix = None
        self.word_vectors = None
        self.k = k 

    def build_vocab(self, corpus):
        self.vocab = {word: i for i, word in enumerate(set(corpus))}

    def build_co_occurrence_matrix(self, corpus):
        num_words = len(self.vocab)
        self.co_occurrence_matrix = lil_matrix((num_words, num_words))
        for i, target_word in enumerate(corpus):
            if target_word in self.vocab:
                target_index = self.vocab[target_word]
                context_range = range(max(0, i - self.window_size), min(len(corpus), i + self.window_size + 1))
                for j in context_range:
                    if i != j and corpus[j] in self.vocab:
                        context_word = corpus[j]
                        context_index = self.vocab[context_word]
                        self.co_occurrence_matrix[target_index, context_index] += 1

    def fit_transform(self, corpus):
        self.build_vocab(corpus)
        self.build_co_occurrence_matrix(corpus)
        co_occurrence_matrix_csr = self.co_occurrence_matrix.tocsr()
        U, Sigma, Vt = svds(co_occurrence_matrix_csr, k=self.k)
        self.word_vectors = U @ np.diag(np.sqrt(Sigma))
        return self.word_vectors

def process_text_and_build_svd():
    data = pd.read_csv("train.csv", encoding='utf-8')
    data['processed_text'] = data['Description'].apply(lambda x: word_tokenize(x.lower()) if pd.notnull(x) else [])
    corpus = data['processed_text'].explode().tolist()
    co_occurrence_model = CoOccurrenceMatrixOptimized(window_size=10)
    word_vectors = co_occurrence_model.fit_transform(corpus)
    vocab = co_occurrence_model.vocab
    torch.save({"vocab": vocab, "word_vectors": word_vectors}, "svd-word-vectors_context10.pth")

if __name__ == "__main__":
    process_text_and_build_svd()
