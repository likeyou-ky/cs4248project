# -*- coding: utf-8 -*-

import numpy as np
import spacy
import pickle
from collections import defaultdict
from allennlp.predictors.predictor import Predictor

predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/biaffine-dependency-parser-ptb-2020.04.06.tar.gz")

nlp = spacy.load('en_core_web_sm')


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


def get_idx(text, st_idx):
    return len(list(filter(lambda x: x == ' ', text[:st_idx])))
      
def generate_children(text):
    tree = predictor.predict(text)
    res = defaultdict(list)
    queue = [tree['hierplane_tree']['root']]
    while len(queue) > 0:
        cur = queue.pop()
        if 'children' in cur.keys():
            res[cur['word']] = list(map(lambda x: (x['word'], get_idx(text, x['spans'][0]['start'])), cur['children']))
            queue += cur['children']
    return res


def dependency_adj_matrix(text, aspect, senticNet):
    # https://spacy.io/docs/usage/processing-text
    document = nlp(text)
    children = generate_children(' '.join(list(map(lambda x: str(x), document))))
    seq_len = len(document)
    matrix = np.ones((seq_len, seq_len)).astype('float32')
    
    for token in document:
        if str(token) in senticNet:
            sentic = float(senticNet[str(token)]) * 0.5
        else:
            sentic = 0
        if str(token) in aspect:
            sentic += 0.5
        if token.i < seq_len:
            matrix[token.i][token.i] = 1 * sentic + sentic
            # https://spacy.io/docs/api/token
            if str(token) not in aspect:
              for child, i in children[str(token)]:
                  if str(child) in senticNet:
                      s = float(senticNet[str(child)]) * 0.5
                  else:
                      s = 0
                  if str(child) in aspect:
                      s += 1 * 0.5
                  if i < seq_len:
                      matrix[token.i][i] = 1 * sentic + s
                      matrix[i][token.i] = 1 * sentic + s
            else:
                for child in document:
                  if str(child) in senticNet:
                      s = float(senticNet[str(child)]) * 0.5
                  else:
                      s = 0
                  if str(child) in aspect:
                      s += 1 * 0.5
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
    # process('./datasets/acl-14-short-data/train.raw')
    # process('./datasets/acl-14-short-data/test.raw')
    process('./datasets/semeval14/restaurant_train.raw')
    process('./datasets/semeval14/restaurant_test.raw')
    process('./datasets/semeval14/laptop_train.raw')
    process('./datasets/semeval14/laptop_test.raw')
    # process('./datasets/semeval15/restaurant_train.raw')
    # process('./datasets/semeval15/restaurant_test.raw')
    # process('./datasets/semeval16/restaurant_train.raw')
    # process('./datasets/semeval16/restaurant_test.raw')