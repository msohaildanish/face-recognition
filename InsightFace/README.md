# InsightFace

PyTorch implementation of Additive Angular Margin Loss for Deep Face Recognition.
[paper](https://arxiv.org/pdf/1801.07698.pdf).
```
@article{deng2018arcface,
title={ArcFace: Additive Angular Margin Loss for Deep Face Recognition},
author={Deng, Jiankang and Guo, Jia and Niannan, Xue and Zafeiriou, Stefanos},
journal={arXiv:1801.07698},
year={2018}
}
```
## Performance

Download the weights in the root dir
|Models|MegaFace|LFW|Download|
|---|---|---|---|
|SE-LResNet101E-IR|98.06%|99.80%|[Link](https://github.com/foamliu/InsightFace-v3/releases/download/v1.0/insight-face-v3.pt)|


## Usage

1. Crop your face image and save it
```bash
$ python photograph.py
```

2. Genetate face embeddings and update the database
```bash
$ python create_database.py
```

3. Test the result using webcam.
```bash
$ python webcam.py
```