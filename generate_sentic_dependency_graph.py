# -*- coding: utf-8 -*-
import numpy as np
import spacy
from common import writefiles
nlp = spacy.load('en_core_web_sm')

# customized function for graph_sentic_aspect_dep_adj_matrix
def dependency_adj_matrix(text, aspect, senticNet):
    # https://spacy.io/docs/usage/processing-text
    document = nlp(text)
    seq_len = len(text.split())
    matrix = np.ones((seq_len, seq_len)).astype('float32')
    
    for token in document:
        if str(token) in senticNet:
            sentic = float(senticNet[str(token)])
        else:
            sentic = 0
        if str(token) in aspect:
            sentic += 1
        if token.i < seq_len:
            matrix[token.i][token.i] = 1 * sentic + sentic
            # https://spacy.io/docs/api/token
            if str(token) not in aspect:
              for child in token.children:
                  if str(child) in senticNet:
                      s = float(senticNet[str(child)])
                  else:
                      s = 0
                  if str(child) in aspect:
                      s += 1
                  if child.i < seq_len:
                      matrix[token.i][child.i] = 1 * sentic + s
                      matrix[child.i][token.i] = 1 * sentic + s
            else:
                for child in document:
                  if str(child) in senticNet:
                      s = float(senticNet[str(child)])
                  else:
                      s = 0
                  if str(child) in aspect:
                      s += 1
                  if child.i < seq_len:
                      matrix[token.i][child.i] = 1 * sentic + s
                      matrix[child.i][token.i] = 1 * sentic + s
    return matrix

if __name__ == '__main__':
    writefiles(dependency_adj_matrix, suffix='.graph_sdat', graph_dep_type=3)