This repository provides the code implementation for the paper **"A Knowledge-Guided Machine Learning Prediction Framework for Metal Ion-Organic Compound Interactions under Data-Sparse Conditions"**. It includes the three methods proposed in the article: **Model-Base**, **Model-KG**, and **Model-KGE**. 

The framework is capable of predicting the binding free energy between metal ions and organic compounds under the guidance of prior chemical knowledge.

## Dependencies

* Python >= 3.12
* numpy == 1.26.4
* pandas == 2.2.3
* torch == 2.5.1
* torch-geometric == 2.6.1
* tensorboard == 2.18.0
* rdkit == 2024.3.2
* deepchem == 2.6.0.dev20211026183818
* scikit-learn == 1.5.2

## Usage

Running command (taking Model-KG as an example):

```bash
python main.py
```

Additionally, model interpretability can be achieved by extracting attention weights using the `get_attention` method in the `GATDeltaGModel` object (Figure 4e), and hidden layer features can be extracted using the `get_feature_map` method (Figure 4c, 4d).
