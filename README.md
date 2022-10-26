# tube_segmentation
> There is growing concern that male reproduction is affected by chemicals in the environment. One way to determine the adverse effect of environmental pollutants is to use wild animals as monitors and evaluate testicular toxicity using histopathology. However, the complexity of testicular tissue makes histopathological evaluation difficult and time-consuming. Automated methods are necessary tools in the quantitative evaluation of histopathology to overcome the subjectivity of manual evaluation and accelerate the process. Testicular tissue consists of seminiferous tubules and interstitial tissue with the epithelium of seminiferous tubules containing cells that differentiate from primitive germ cells to spermatozoa in several steps.  Segmenting the epithelial layer of the seminiferous tubule is a prerequisite for the development of automated methods for detecting abnormalities in testicular tissue. We propose a  new fully connected neural network to segment the epithelium of testicular tissue.  We applied the proposed method for the 2-class problem where the epithelial layer of the tubule is the target class. The f-score and IOU of the proposed method are $\textbf{0.85\%}$ and $\textbf{0.92\%}$.  Although the proposed method is trained on a limited training set, it performs well on an independent dataset and outperforms other state-of-the-art methods.  The pretrained ResNet-34 in the encoder and attention block suggested in the decoder result in better segmentation and generalization. The proposed method can be applied to testicular tissue images from any mammalian species and can be used as the first part of a fully automated testicular tissue processing pipeline.

## contents
this repo is organized in this way:

```
.
├── tube_segmentation
├── data
├── outputs
└── requirements.txt
```

## Install
change the directory to path of this project and install required libraries with pip:

```sh
pip install -r requirements.txt
```

## Dataset
The dataset used for training is available for download: [Google drive]()

## Pretrained Networks
pretrained networks can be downloaded and put in `ouputs` folder: [Google drive]()

## Train

## Predict

## Evaluation

## Acknowledgment
