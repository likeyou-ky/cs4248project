# -*- coding: utf-8 -*-

import numpy as np
import spacy
import pickle

nlp = spacy.load('en_core_web_sm')
aspect_w = 0.5
sentic_w = 0.5


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
    matrix = np.zeros((seq_len, seq_len)).astype('float32')
    #print('='*20+':')
    #print(document)
    #print(senticNet)
    
    # for token in document:
    #     if str(token) in senticNet:
    #         sentic = float(senticNet[str(token)])
    #     else:
    #         sentic = 0
    #     if str(token) in aspect:
    #         sentic += 1 * aspect_w
    #     if token.i < seq_len:
    #         matrix[token.i][token.i] = (1 * sentic + sentic) * sentic_w + 1
    #         # https://spacy.io/docs/api/token
    #         for child in token.children:
    #             if str(child) in senticNet:
    #                 child_sentic = float(senticNet[str(child)])
    #             else:
    #                 child_sentic = 0
    #             if str(token) not in aspect and str(child) in aspect:
    #                 child_sentic += 1 * aspect_w
    #             if child.i < seq_len:
    #                 matrix[token.i][child.i] = (1 * sentic + child_sentic) * sentic_w + 1
    #                 matrix[child.i][token.i] = (1 * sentic + child_sentic) * sentic_w + 1
    for token in document:
        if str(token) in senticNet:
            sentic = float(senticNet[str(token)])
        else:
            sentic = 0
        if str(token) in aspect:
            t = 1
        else:
            t = 0
        if token.i < seq_len:
            matrix[token.i][token.i] = 1 * ((sentic + sentic) + t + 1)
            # https://spacy.io/docs/api/token
            for child in token.children:
                if str(child) in senticNet:
                    child_sentic = float(senticNet[str(child)])
                else:
                    child_sentic = 0
                if str(token) in aspect or str(child) in aspect:
                    t = 1
                else:
                    t = 0
                if child.i < seq_len:
                    matrix[token.i][child.i] = 1 * ((sentic + child_sentic) + t + 1)
                    matrix[child.i][token.i] = 1 * ((sentic + child_sentic) + t + 1)
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
    process('./datasets/acl-14-short-data/train.raw')
    process('./datasets/acl-14-short-data/test.raw')
    process('./datasets/semeval14/restaurant_train.raw')
    process('./datasets/semeval14/restaurant_test.raw')
    process('./datasets/semeval14/laptop_train.raw')
    process('./datasets/semeval14/laptop_test.raw')
    process('./datasets/semeval15/restaurant_train.raw')
    process('./datasets/semeval15/restaurant_test.raw')
    process('./datasets/semeval16/restaurant_train.raw')
    process('./datasets/semeval16/restaurant_test.raw')
