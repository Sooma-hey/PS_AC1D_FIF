# PS_AC1D_FIF: Efficient SER Model  
*A Deep Neural Network achieving real-time Speech Emotion Recognition (SER) with low computational cost.*
[The paper's DOI](https://doi.org/10.1007/s11042-025-21009-4)
[The paper's PDF version](https://rdcu.be/evp2h)

---

## Introduction  
PS_AC1D_FIF is a resource-efficient Deep Neural Network designed for real-time **Speech Emotion Recognition (SER)** applications. By integrating prosodic and spectral features with alternating 1D convolutional layers, batch normalization, and fixed kernel sizes, it achieves a balance between computational efficiency and recognition accuracy.  

The model’s lightweight design, achieved by omitting pooling layers, makes it suitable for **low-resource applications** while maintaining high performance.

---

## Highlights  
- **Lightweight Architecture:** Alternating 1D convolutional layers with no pooling layers.  
- **High Accuracy:**  
  - **87.85%** on EMO-DB (240 seconds, 529,207 parameters).  
  - **81.12%** on RAVDESS (572 seconds, 563,908 parameters).  
- **Real-Time Capabilities:** Optimized for low computational resources.  
- **Scalable and Efficient:** Suitable for various real-time and embedded applications.

---

## Directory Structure  


```
PS_AC1D_FIF/
├── checkpoints/       # Saved models and configurations
│   ├── PS-AC1D-FIF.h5 # Model weights
│   ├── PS_AC1D_FIF.json # Model architecture in JSON
│   └── SCALER_LIBROSA.m # Scaler file
├── configs/           # Configuration files
│   └── PS_AC1D_FIF.yaml
├── datasets/          # Datasets for training/testing
│   ├── EMODB          # EMO-DB dataset
│   └── RAVDESS        # RAVDESS dataset
├── extract_feats/     # Feature extraction scripts
│   └── librosa.py
├── features/          # Extracted features
│   ├── EMODB_427      # Extracted features for EMO-DB
│   └── RAVDESS_417    # Extracted features for RAVDESS
├── models/            # Model definition files
│   └── dnn/
│       ├── __init__.py
│       ├── dnn.py
│       └── PS_AC1D_FIF.py
│   ├── __init__.py
│   └── base.py
├── utils/             # Utility scripts
│   ├── __init__.py
│   ├── files.py
│   ├── opts.py
│   ├── plot.py
├── predict.py         # Prediction script
├── preprocess.py      # Data preprocessing script
├── train.py           # Model training script
├── README.md          # Documentation file

```


## Datasets

1. [RAVDESS](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)

   English, around 1500 audios from 24 people (12 male and 12 female) including 8 different emotions (the third number of the file name represents the emotional type): 01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised.

2. [EMO-DB](https://www.kaggle.com/datasets/piyushagni5/berlin-database-of-emotional-speech-emodb)

   German, around 500 audios from 10 people (5 male and 5 female) including 7 different emotions (the second to last letter of the file name represents the emotional type): N = neutral, W = angry, A = fear, F = happy, T = sad, E = disgust, L = boredom.

&nbsp;

## Usage

### Prepare

Clone the ripo:


```python
git clone <repo-link>
cd PS_AC1D_FIF
```

Then Install the requirements:
```python
pip install -r requirements.txt
```

&nbsp;

### Experimental Setup

* Epochs: 100

* Batch Size: 10

* Optimizer: RMSProp
	
  * Learning Rate: 0.00001

  * Decay Rate: 1e-6

* Loss Function: Categorical Cross-Entropy

* Cross-Validation: 5-Fold

* Evaluation Metrics: Precision, Recall, F1-Score, and UWA

&nbsp;
