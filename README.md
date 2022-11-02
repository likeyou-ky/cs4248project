# Modification on Sentic GCN
# Introduction
This repository was used in this paper:  
  
[**Aspect-Based Sentiment Analysis via Affective Knowledge Enhanced Graph Convolutional Networks**](https://www.sentic.net/sentic-gcn.pdf)
<br>
Bin Liang, Hang Su, Lin Gui, Erik Cambria, Ruifeng Xu. *Knowledge-Based Systems, 2021: 107643.*
  
Please cite this paper and kindly give a star for this repository if you use this code.

## Requirements

| Package Name | Version |
| --- | --- |
| Python | 3.10 |
| PyTorch | 1.12.1 |
| SpaCy | 3.4.2 |
| numpy | 1.23.1 |
| Transformers | 4.24.0 |
| ipdb | 0.13.9 |

## Usage

* Install packages with
```bash
pip install spacy
python -m spacy download en_core_web_sm
pip install transformers
pip install ipdb
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

## Training
* Train with command, optional arguments could be found in [train.py](/train.py) \& [train_bert.py](/train_bert.py)
* Please tune the argument of *--seed* for better performance

* Run senticgcn: ```./run_senticgcn.sh```

* Run senticgcn_bert: ```./run_senticgcn_bert.sh```


## Testing
* Testing with the models saved in **state_dict**. Optional arguments could be found in [infer.py](/infer.py) \& [infer_for_bert.py](/infer_for_bert.py)
* Please run ```python infer.py``` for the testing of non-BERT models.
* Please run ```python infer_for_bert.py``` for the testing of BERT-based models.



## Citation

The BibTex of the citation is as follow:

```bibtex
@article{liang2021aspect,
  title={Aspect-based sentiment analysis via affective knowledge enhanced graph convolutional networks},
  author={Liang, Bin and Su, Hang and Gui, Lin and Cambria, Erik and Xu, Ruifeng},
  journal={Knowledge-Based Systems},
  pages={107643},
  year={2021},
  publisher={Elsevier}
}
```

## See Also
* The original knowledge base of [SenticNet](https://sentic.net/) could be found at https://sentic.net/downloads/.
* The knowledge source used in this code is [SenticNet 5](https://sentic.net/senticnet-5.pdf), which is stored at [senticnet-5.0/senticnet5.txt](/senticnet-5.0/senticnet5.txt).
* We also set several variants of our model:
    | Model        | Decription |
    | --------   | -----   |
    | [affectivegcn.py](/models/affectivegcn.py) |   Only using the affective information (i.e. Eq.2) for building graphs |
    | [attsenticgcn.py](/models/attsenticgcn.py) |   Combining our model with attention mechanism |
    | [sdgcn.py](/models/sdgcn.py) |   Interactively performing the graph convolutional operation based on dependency (i.e. Eq. 1) and affective (i.e. Eq.2) graphs |


## Credits

* The affective knowledge used in this work is from [SenticNet](https://sentic.net/).
* Here, we would like to express our heartfelt thanks to all the authors of SenticNet. 
* The code of this repository partly relies on [ASGCN](https://github.com/GeneZC/ASGCN) \& [ABSA-PyTorch](https://github.com/songyouwei/ABSA-PyTorch). 
* Here, we would like to express our gratitude to the authors of the [ASGCN](https://github.com/GeneZC/ASGCN) \& [ABSA-PyTorch](https://github.com/songyouwei/ABSA-PyTorch) repositories.