# 🚀 ArviNet — A Custom Pretrained CNN Backbone

ArviNet is a custom ResNet-style convolutional neural network backbone trained from scratch on **CIFAR-100**, designed to act as a reusable feature extractor for downstream computer vision tasks.

It was built, trained, validated, and then packaged into a clean reusable module for transfer learning experiments.

---

## 🔍 Overview

- 📦 **Architecture:** Custom Residual CNN (ResNet-inspired)
- 🏗️ **Parameters:** 11,220,132
- 🎯 **Pretrained on:** CIFAR-100
- 📏 **Output Feature Dimension:** 512
- 🧠 **Designed for:** Transfer learning & feature extraction

The backbone outputs:
```bash 
(None, 512)
```


after global average pooling.

---

## 🏋️ Pretraining Details (CIFAR-100)

- **Batch Size:** 32  
- **Callbacks Used:**
  - ReduceLROnPlateau
  - ModelCheckpoint
  - EarlyStopping
- **Planned Epochs:** 100  
- Early convergence observed  

### Final Training Metrics:

- **Training Accuracy:** 95.87%
- **Validation Accuracy:** 66.22%

> Note: Validation accuracy reflects CIFAR-100 complexity (100 fine-grained classes).

---

## 🔁 Transfer Learning Evaluation (CIFAR-10)

To evaluate transferability, ArviNet was tested on CIFAR-10 under three conditions:

| Model Type | Training Acc | Validation Acc | Epochs |
|------------|-------------|----------------|--------|
| Random Init | 99.52% | 83.72% | 30 |
| Frozen Backbone | 76.95% | 76.14% | 20 |
| Fine-Tuned Backbone | 91.52% | **85.97%** | 20 |

### 🔥 Key Observations

- Fine-tuning improved performance over random initialization.
- Pretrained features accelerated convergence.
- Frozen backbone underperformed due to domain shift between CIFAR-100 and CIFAR-10.
- Fine-tuning last layers yielded best results.

---

## 📊 Training & Transfer Graphs

Training and validation curves for all experiments are included in the repository.

These graphs demonstrate:

- Overfitting behavior in scratch training
- Plateau in frozen backbone
- Faster convergence in fine-tuned backbone

See `/notebooks` for experiment details and plotted graphs.

---

## 🏗️ Architecture Summary

ArviNet follows a staged residual design:

- Stem Convolution
- 4 Residual Stages:
  - 64 filters
  - 128 filters
  - 256 filters
  - 512 filters
- Global Average Pooling
- 512-D Feature Vector Output

It is inspired by ResNet-style design principles while being implemented from scratch using TensorFlow/Keras Functional API.

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/arvindmatharoo/ArviNet.git
cd ArviNet
pip install -r requirements.txt

```
## Usage Example 
```python
import tensorflow as tf
from arvinet import ArviNet

# Load pretrained backbone
backbone = ArviNet(pretrained=True)

# Freeze if desired
backbone.trainable = False

# Attach your own classifier
inputs = tf.keras.Input(shape=(32, 32, 3))
x = backbone(inputs)
outputs = tf.keras.layers.Dense(10, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)
```

## Suggested Use Case 
- Small image classification tasks
- Feature extraction for custom datasets
- Transfer learning experiments
- Educational purposes (understanding CNN backbones)


## 📁 Repository Structure

arvinet/
    model.py
    __init__.py
    weights/
        arvinet_pretrained.weights.h5

notebooks/
    main.ipynb
    test.ipynb


📌 Why This Project?

This project demonstrates:
- Designing CNN architecture from scratch
- Training with advanced callbacks
- Evaluating transfer learning properly
- Benchmarking against scratch models
- Packaging a model into a reusable Python module
- It bridges experimentation and engineering.

## 🔮 Future Improvements

- Train on larger datasets (e.g., ImageNet subset)
- Add model scaling variants (ArviNet-S, M, L)
- Provide HuggingFace model card
- Add Grad-CAM visualizations
- Benchmark on handwriting datasets


## Author 
*Arvind Singh* 
- Computer Science Student | Machine Learning Enthusiast 

