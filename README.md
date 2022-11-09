# Aspect-based Sentiment Analysis with Enhanced Graph Convolutional Networks
# Introduction
This repository was used in this paper:  
  
Aspect-based Sentiment Analysis with Enhanced Graph Convolutional Networks
<br>
Cao Peng, Gao Gui, Li Bozhao, Li Keyou, Mo Yunbin
  
Please kindly cite this paper if you use this code.

## Requirements

| Package Name | Version |
| --- | -- |
| sklearn | |
| Python | 3.8 |
| PyTorch | 1.12.1 |
| SpaCy | 3.4.2 |
| numpy | 1.23.1 |
| Transformers | 4.24.0 |
| ipdb | 0.13.9 |
| allennlp | 2.10.1 |
|allennlp-models | 2.10.1 |

## Usage

1. (required) Install packages with
```bash
pip install scikit-learn
pip install spacy
python -m spacy download en_core_web_sm
pip install transformers
pip install ipdb
pip install allennlp
pip install allennlp-models
```
2. (optional) Generate dependency graph with
```bash
python generate_dependency_graph.py
```
3. (optional) Generate sentic graph with
```bash
python generate_sentic_graph.py
```
4. (optional) Generate sentic & dependeny graph with
```bash
# For the baseline version
python generate_sentic_dependency_graph.py
# OR for our improved version
python improved_gen_sentic_dep_graph.py
```
p.s. Steps 2, 3 and 4 are optional because the output graph files are already stored in corresponding folders.

5. Training can be performed with
```bash
./train_model.sh
```

6. Testing can be performed with
```bash
# The sentence and aspect are hardcoded at lines 119-120.
# Please change them if needed before running this line.
python3 infer_for_bert.py
```