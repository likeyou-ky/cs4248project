# -*- coding: utf-8 -*-
from common.fio import graph_sentic_aspect_dep_adj_mat, writefiles

# dependency graph based on text, sentic net and sentiment aspects.
if __name__ == '__main__':
    writefiles(graph_sentic_aspect_dep_adj_mat, suffix='.sentic', graph_dep_type=3)