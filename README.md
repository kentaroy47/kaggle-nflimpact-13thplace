# 13th place solution of team tara
Work by team tara: [@tereka](https://www.kaggle.com/tereka) [@hidehisaarai1213](https://www.kaggle.com/hidehisaarai1213) [@rishigami](https://www.kaggle.com/rishigami) [arutema47](https://www.kaggle.com/kyoshioka47)

Public: 7th 0.5503

Private: 13th 0.5017

fix.. Private: 8th 0.5337

## Overview
[Solution writeup](https://www.kaggle.com/c/nfl-impact-detection/discussion/208801)

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F3508221%2Fa5abcefa4a1968535ed22d82b18569ee%2Fimage%20(1).png?generation=1609822336590711&alt=media)


# Preparation
`pip install odach`

`pip install timm=0.1.26`

1. `train-prepare-labels.ipynb`でラベルと画像データを書き出し

Write out training images with `train-prepare-labels.ipynb`

2.`prepare_classification_images.ipynb`でclassification用データを書き出し

Write out classification images with `prepare_classification_images.ipynb`

3. pretrainフォルダにeffdetの事前学習モデルを入れておく

Place effdet pretrained models inside `pretrain` folder.

# train
for end model:
`python train_1ststage.py --enum 15 --modeltype Endzone --cutmix --strech 3 --imsize 1024 --bs 3 --all --effdet effdet4 --lr 1e-4`

for side model:
`python train_1ststage.py --enum 15 --modeltype Sideline --cutmix --strech 3 --imsize 1024 --bs 3 --all --effdet effdet4 --lr 1e-4`

classification
`python train_2ndstage.py --all`

# Inferece
1. 1ststage-inferenceでdetection結果を取得

Get inference results with `1st-stage-Inference.ipynb`

2.2nd stage-inferenceでclassification。

CLassify with `2nd-stage-Inference.ipynb`

