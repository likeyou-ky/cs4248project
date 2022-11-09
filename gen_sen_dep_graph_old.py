# -*- coding: utf-8 -*-

import numpy as np
import stanza
from common import writefiles
nlp = stanza.Pipeline('en')

def dependency_adj_matrix(text, aspect, senticNet):
    # https://spacy.io/docs/usage/processing-text
    document = nlp(text)
    seq_len = len(text.split())
    matrix = np.zeros((seq_len, seq_len)).astype('float32')
    #print('='*20+':')
    #print(document)
    #print(senticNet)
    
    for token in document.sentences:
        #print('token:', token)
        if str(token) in senticNet:
            sentic = float(senticNet[str(token)]) + 1
        else:
            sentic = 0
        if str(token) in aspect:
            sentic += 1
        if token.i < seq_len:
            matrix[token.i][token.i] = 1 * sentic
            # https://spacy.io/docs/api/token
            for child in token.children:
                if str(child) in aspect:
                    sentic += 1
                if child.i < seq_len:
                    matrix[token.i][child.i] = 1 * sentic
                    matrix[child.i][token.i] = 1 * sentic

    return matrix

if __name__ == '__main__':
    writefiles(dependency_adj_matrix, '.graph_sdat', graph_dep_type=3)
