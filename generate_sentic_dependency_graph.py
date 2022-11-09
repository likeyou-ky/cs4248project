# -*- coding: utf-8 -*-

import numpy as np
from common import writefiles
import pickle
import stanfordnlp

stanfordnlp.download('en')
nlp = stanfordnlp.Pipeline()

# nlp = spacy.load('en_core_web_sm')


# customized function for graph_sentic_aspect_dep_adj_matrix

def load_sentic_word():
    """
    load senticNet
    """
    path = './senticNet/senticnet_word.txt'
    senticNet = {}
    fp = open(path, 'r')
    for line in fp:
        line = line.strip()
        if not line:
            continue
        word, sentic = line.split('\t')
        senticNet[word] = sentic
    fp.close()
    return senticNet


def dependency_adj_matrix(text, aspect, senticNet):
    # https://spacy.io/docs/usage/processing-text
    document = nlp(text)
    seq_len = len(text.split())
    matrix = np.ones((seq_len, seq_len)).astype('float32')
    #print('='*20+':')
    #print(document)
    #print(senticNet)
    
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

def process(filename):
    senticNet = load_sentic_word()
    fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    idx2graph = {}
    fout = open(filename+'.graph_sdat', 'wb')
    for i in range(0, len(lines), 3):
        text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
        aspect = lines[i + 1].lower().strip()
        adj_matrix = dependency_adj_matrix(text_left+' '+aspect+' '+text_right, aspect, senticNet)
        idx2graph[i] = adj_matrix
    pickle.dump(idx2graph, fout)
    print('done !!!'+filename)
    fout.close() 

if __name__ == '__main__':
    writefiles(dependency_adj_matrix, suffix='.graph_sdat', graph_dep_type=3)
