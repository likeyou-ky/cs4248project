# -*- coding: utf-8 -*-

import numpy as np
import spacy
import pickle

nlp = spacy.load('en_core_web_sm')

def dependency_adj_matrix(text):
    document = nlp(text)
    seq_len = len(text.split())
    matrix = np.zeros((seq_len, seq_len)).astype('float32')
    for token in document:
        if token.i < seq_len:
            matrix[token.i][token.i] = 1
            for child in token.children:
                if child.i < seq_len:
                    matrix[token.i][child.i] = 1
                    matrix[child.i][token.i] = 1
    return matrix

def sentic_adj_matrix(text, aspect, senticNet):
    word_list = text.split()
    seq_len = len(word_list)
    matrix = np.zeros((seq_len, seq_len)).astype('float32')
    for i in range(seq_len):
        word = word_list[i]
        if word in senticNet:
            sentic = float(senticNet[word]) + 1.0
        else:
            sentic = 0
        if word in aspect:
            sentic += 1.0
        for j in range(seq_len):
            matrix[i][j] += sentic
            matrix[j][i] += sentic
    for i in range(seq_len):
        if matrix[i][i] == 0:
            matrix[i][i] = 1
    return matrix

def sent_dep_adj_matrix(text, aspect, senticNet):
    document = nlp(text)
    seq_len = len(text.split())
    matrix = np.zeros((seq_len, seq_len)).astype('float32')
    for token in document:
        if str(token) in senticNet:
            sentic = float(senticNet[str(token)]) + 1
        else:
            sentic = 0
        if str(token) in aspect:
            sentic += 1
        if token.i < seq_len:
            matrix[token.i][token.i] = 1 * sentic
            for child in token.children:
                if str(child) in aspect:
                    sentic += 1
                if child.i < seq_len:
                    matrix[token.i][child.i] = 1 * sentic
                    matrix[child.i][token.i] = 1 * sentic
    return matrix

def load_sentic_word():
    path = './datasets/senticnet_word.txt'
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

def process(filename):
    senticNet = load_sentic_word()
    with open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore') as fin:
        lines = fin.readlines()
    idx2dgraph = {}
    idx2sgraph = {}
    idx2sdgraph = {}
    fgout = open(filename+'.graph', 'wb')
    fsout = open(filename+'.sentic', 'wb')
    fsdout = open(filename+'.graph_sdat', 'wb')
    for i in range(0, len(lines), 3):
        text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
        aspect = lines[i + 1].lower().strip()
        idx2dgraph[i] = dependency_adj_matrix(text_left+' '+aspect+' '+text_right)
        idx2sgraph[i] = sentic_adj_matrix(text_left+' '+aspect+' '+text_right, aspect, senticNet)
        idx2sdgraph[i] = sent_dep_adj_matrix(text_left+' '+aspect+' '+text_right, aspect, senticNet)
    pickle.dump(idx2dgraph, fgout)  
    pickle.dump(idx2sgraph, fsout)
    pickle.dump(idx2sdgraph, fsdout)
    print('done !!!', filename)      
    fgout.close()
    fsout.close()
    fsdout.close()

if __name__ == '__main__':
    process('./datasets/semeval14/restaurant_train.raw')
    process('./datasets/semeval14/restaurant_test.raw')
    process('./datasets/semeval14/laptop_train.raw')
    process('./datasets/semeval14/laptop_test.raw')
    process('./datasets/semeval15/restaurant_train.raw')
    process('./datasets/semeval15/restaurant_test.raw')
    process('./datasets/semeval16/restaurant_train.raw')
    process('./datasets/semeval16/restaurant_test.raw')