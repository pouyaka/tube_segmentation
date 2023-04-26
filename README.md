# tube_segmentation
This is PyTorch implementation of "Deep learning-based method for segmenting epithelial layer of tubules in histopathological images of testicular tissue".
https://arxiv.org/abs/2301.09887

abstract:
> There is growing concern that male reproduction is affected by chemicals in the environment. One way to determine the adverse effect of environmental pollutants is to use wild animals as monitors and evaluate testicular toxicity using histopathology. However, the complexity of testicular tissue makes histopathological evaluation difficult and time-consuming. Automated methods are necessary tools in the quantitative evaluation of histopathology to overcome the subjectivity of manual evaluation and accelerate the process. Testicular tissue consists of seminiferous tubules and interstitial tissue with the epithelium of seminiferous tubules containing cells that differentiate from primitive germ cells to spermatozoa in several steps.  Segmenting the epithelial layer of the seminiferous tubule is a prerequisite for the development of automated methods for detecting abnormalities in testicular tissue. We propose a  new fully connected neural network to segment the epithelium of testicular tissue.  We applied the proposed method for the 2-class problem where the epithelial layer of the tubule is the target class. The f-score and IOU of the proposed method are $\textbf{0.85}$ and $\textbf{0.92}$.  Although the proposed method is trained on a limited training set, it performs well on an independent dataset and outperforms other state-of-the-art methods.  The pretrained ResNet-34 in the encoder and attention block suggested in the decoder result in better segmentation and generalization. The proposed method can be applied to testicular tissue images from any mammalian species and can be used as the first part of a fully automated testicular tissue processing pipeline.

![Method Overview](/assets/fig1.png "method overview")

## Install
this repo is organized in this way:

```
.
├── tube_segmentation
├── data
├── outputs
└── requirements.txt
```
change the directory to path of this project and install required libraries with pip:

```sh
pip install -r requirements.txt
```

## Download
### Dataset
The dataset used for training is available for download: [Google drive](https://drive.google.com/file/d/1235SJ8eMAia7rXO3NmGS1fwYUdV5J009/view?usp=share_link)
### Pretrained Networks
pretrained networks can be downloaded and put in `outputs` folder: [Google drive]()

## Train
Download the dataset first. Use `train.py` for training different networks. Find more details on different options for train with command line interfaces(CLI) help:

```sh
python -m tube_segmentation.train --help
```
For example you can train the proposed model with:

```sh
python -m tube_segmentation.train --num-class 2 --long-edge 1280 --model-name proposedscse --pretrained --loss-fn tversky --epochs 60 --train-batch-size 4 --lr 1e-4
```

## Predict
Use downloadable trained models or train a network for getting an output of segmentation with `predict.py`. You should choose a directory of image or images for prediction. output for every image will be an image containing input image, ground truth (optional), predicted mask, and overlaid boundaries. use CLI help for more details:

```sh
python -m tube_segmentation.predict --help
```
For example you can predict one of the images in the dataset with this command(using downloadable trained model):

```sh
python -m tube_segmentation.predict --image-dir data/imgs/ --images 'M02G4x20 (10).TIF' --mat-dir data/mats/ --imagenet-norm --checkpoint outputs/proposedscse-tversky-2class-221129-173619.pt
```

## Evaluation
Get segmentation metrics like F-score, intersection over union (IoU), and mean Aggregated Jaccard Index (mAJI) with `eval.py`. use the help of CLI for different options:

```sh
python -m tube_segmentation.eval --help
```

## Acknowledgment
* main structure of the code is inspired by [openpifpaf](https://github.com/openpifpaf)
* code of K-fold cross validation in trainer.py is adapted from:
https://www.machinecurve.com/index.php/2021/02/03/how-to-use-k-fold-cross-validation-with-pytorch
