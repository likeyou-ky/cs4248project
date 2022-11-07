import numpy as np
import spacy
import pickle
nlp = spacy.load('en_core_web_sm')

def graph_dep_adj_mat(text):
    # https://spacy.io/docs/usage/processing-text
    document = nlp(text)
    seq_len = len(text.split())
    matrix = np.zeros((seq_len, seq_len)).astype('float32')
    
    for token in document:
        if token.i < seq_len:
            matrix[token.i][token.i] = 1
            # https://spacy.io/docs/api/token
            for child in token.children:
                if child.i < seq_len:
                    matrix[token.i][child.i] = 1
                    matrix[child.i][token.i] = 1

    return matrix

def graph_sentic_dep_adj_mat(text, senticNet):
    word_list = text.split()
    seq_len = len(word_list)
    matrix = np.zeros((seq_len, seq_len)).astype('float32')
    
    for i in range(seq_len):
        word = word_list[i]
        if word in senticNet:
            sentic = float(senticNet[word]) + 1.0
        else:
            sentic = 0.5
        for j in range(seq_len):
            matrix[i][j] += sentic
        for k in range(seq_len):
            matrix[k][i] += sentic
        matrix[i][i] = 1

    return matrix

def graph_sentic_aspect_dep_adj_mat(text, aspect, senticNet):
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

def load_sentic_word():
    """
    load senticNet
    """
    # sentic dependency graph.py uses _60 suffix
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

def process(filename, depadjmatrix_func, suffix='', graph_dep_type=1):
    if graph_dep_type==1:
        senticNet = None
    else:
        senticNet = load_sentic_word()
    fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    idx2graph = {}
    fout = open(filename+suffix, 'wb')
    for i in range(0, len(lines), 3):
        text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
        aspect = lines[i + 1].lower().strip()
        if graph_dep_type==1:
            adj_matrix = depadjmatrix_func(text_left+' '+aspect+' '+text_right)
        elif graph_dep_type==2:
            adj_matrix = depadjmatrix_func(text_left + ' ' + aspect + ' ' + text_right, senticNet)
        else:
            adj_matrix = depadjmatrix_func(text_left + ' ' + aspect + ' ' + text_right, aspect, senticNet)
        idx2graph[i] = adj_matrix
    pickle.dump(idx2graph, fout)
    print('done !!!', filename)
    fout.close() 

def writefiles(func, suffix='', graph_dep_type=1):
    process('./datasets/acl-14-short-data/train.raw', func, suffix, graph_dep_type)
    process('./datasets/acl-14-short-data/test.raw', func, suffix, graph_dep_type)
    process('./datasets/semeval14/restaurant_train.raw', func, suffix, graph_dep_type)
    process('./datasets/semeval14/restaurant_test.raw', func, suffix, graph_dep_type)
    process('./datasets/semeval14/laptop_train.raw', func, suffix, graph_dep_type)
    process('./datasets/semeval14/laptop_test.raw', func, suffix, graph_dep_type)
    process('./datasets/semeval15/restaurant_train.raw', func, suffix, graph_dep_type)
    process('./datasets/semeval15/restaurant_test.raw', func, suffix, graph_dep_type)
    process('./datasets/semeval16/restaurant_train.raw', func, suffix, graph_dep_type)
    process('./datasets/semeval16/restaurant_test.raw', func, suffix, graph_dep_type)