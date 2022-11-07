# -*- coding: utf-8 -*-
from common import graph_dep_adj_mat, writefiles

# dependency graph based on text. Unrelated to sentic net or sentiment aspects.
if __name__ == '__main__':
    writefiles(graph_dep_adj_mat, suffix='.graph', graph_dep_type=1)