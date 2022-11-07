# -*- coding: utf-8 -*-
import numpy as np
import spacy
from common import writefiles
nlp = spacy.load('en_core_web_sm')
aspect_w = 0.5
sentic_w = 0.5

# customized function for graph_sentic_aspect_dep_adj_matrix
def dependency_adj_matrix(text, aspect, senticNet):
    # https://spacy.io/docs/usage/processing-text
    document = nlp(text)
    seq_len = len(text.split())
    matrix = np.zeros((seq_len, seq_len)).astype('float32')
    
    for token in document:
        if str(token) in senticNet:
            sentic = float(senticNet[str(token)])
        else:
            sentic = 0
        if str(token) in aspect:
            sentic += 1 * aspect_w
        if token.i < seq_len:
            matrix[token.i][token.i] = (1 * sentic + sentic) * sentic_w + 1
            # https://spacy.io/docs/api/token
            for child in token.children:
                if str(child) in senticNet:
                    child_sentic = float(senticNet[str(child)])
                else:
                    child_sentic = 0
                if str(token) not in aspect and str(child) in aspect:
                    child_sentic += 1 * aspect_w
                if child.i < seq_len:
                    matrix[token.i][child.i] = (1 * sentic + child_sentic) * sentic_w + 1
                    matrix[child.i][token.i] = (1 * sentic + child_sentic) * sentic_w + 1
    return matrix 

if __name__ == '__main__':
    writefiles(dependency_adj_matrix, suffix='.graph_sdat', graph_dep_type=3)