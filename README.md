## An ensemble approach to anomaly detection using high- and low-variance principal components
<p align="left"><img width="99%" src="./assests/teaser.png"/></p>

This repository provides the official PyTorch implementation of the following paper:
> **Abstract:** *With the recent proliferation of cyber physical systems (CPSs), there is a growing demand for reliable anomaly detection systems. In this paper, we propose a new ensemble learning approach for anomaly detection that utilizes the extraction of specific features tailored to anomaly detection problems. Whereas typical principal component analysis (PCA) selects principal components (PCs) associated with high variances, our proposed method also leverages PCs with low variances to account for unexpressed variations in the training data. The extracted features are then fed into conventional learning models such as support vector machines or recurrent neural networks. Since each PC can be particularly good at detecting certain types of attacks, classifiers based on different combinations of selected PCs are further combined as an ensemble. Our results show that the ensemble approach improves the overall accuracy and helps detect diverse types of unknown attacks as well. Furthermore, our simple yet effective and flexible approach can easily be deployed to various CPS environments of increasing complexity.*

### Requirement
```shell
pip3 install -r requirements.txt
```

### Train
```shell
bash ./scripts/run.sh
```

### Dataset
#### SWAT
We use the [SWaT](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/) dataset.

### Citation
```
@article{moon2022ensemble,
  title={An ensemble approach to anomaly detection using high-and low-variance principal components},
  author={Moon, Jeong-Hyeon and Yu, Jun-Hyung and Sohn, Kyung-Ah},
  journal={Computers and Electrical Engineering},
  volume={99},
  pages={107773},
  year={2022},
  publisher={Elsevier}
}
```
