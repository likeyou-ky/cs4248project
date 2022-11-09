# Modification on Sentic GCN
# Introduction
This repository was used in this paper:  
  
Aspect-based Sentiment Analysis with enhanced graph convolutional networks
<br>
Cao Peng, Gao Gui, Li Bozhao, Li Keyou, Mo Yunbin
  
Please kindly cite this paper if you use this code.

## Requirements

| Package Name | Version |
| --- | -- |
| Python | 3.8 |
| PyTorch | 1.12.1 |
| SpaCy | 3.4.2 |
| numpy | 1.23.1 |
| Transformers | 4.24.0 |
| ipdb | 0.13.9 |
| allennlp | 2.10.1 |
|allennlp-models | 2.10.1 |

## Usage

* Install packages with
```bash
pip install spacy
python -m spacy download en_core_web_sm
pip install transformers
pip install ipdb
pip install allennlp
pip install allennlp-models

```
* Generate dependency graph with
```bash
python generate_dependency_graph.py
```
* Generate sentic graph with
```bash
python generate_sentic_graph.py
```
* Generate sentic & dependeny graph with
```bash
python generate_sentic_dependency_graph.py
```
p.s. The three steps above can be skipped because the output graph files are already stored in corresponding folders.

## Training
```bash
./train_model.sh
```

## Testing
```bash
./test_model.sh
```
