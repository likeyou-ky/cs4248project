# -*- coding: utf-8 -*-
from common.fio import writefiles, graph_sentic_dep_adj_mat

# dependency graph based on text and sentic net. Unrelated to sentiment aspects.
if __name__ == '__main__':
    writefiles(graph_sentic_dep_adj_mat, suffix='.graph_s',graph_dep_type=2)