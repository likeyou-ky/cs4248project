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
| Python | 3.8 |
| PyTorch | 1.12.1 |
| scikit-learn | 1.1.3 |
| SpaCy | 3.4.2 |
| numpy | 1.23.1 |
| Transformers | 4.24.0 |
| ipdb | 0.13.9 |

Download the pre-trained Glove from here https://nlp.stanford.edu/data/glove.840B.300d.zip and put the unzipped file in the same directory as ```generate_graphs.py```

## Usage

1. (required) Install packages with

    ```bash
    pip install scikit-learn
    pip install spacy
    python -m spacy download en_core_web_sm
    pip install transformers
    pip install ipdb
    ```

1. (optional) Generate all graphs with

    ```bash
    python generate_graphs.py
    ```

1. Training can be performed with

    ```bash
    # seed 4248 can be replaced with any other seed
    CUDA_VISIBLE_DEVICES=1 python3 train.py --model_name baselinegcn --dataset rest14 --save True --learning_rate 1e-3 --seed 4248 --batch_size 16 --hidden_dim 300
    ```

1. Testing can be performed with

    ```bash
    # The sentence and aspect are hardcoded at lines 105-106.
    # Please change them if needed before running this line.
    python infer.py
    ```
