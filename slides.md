---
marp: true
theme: gaia
_class: lead
paginate: true
backgroundColor: #fff
backgroundImage: url('https://marp.app/assets/hero-background.svg')
---

![bg left:40% 80%](https://uploads-ssl.webflow.com/5f0c5c0bb18a279f0a62919e/5f0d7ed550b49600837c5467_privacy-image.svg)
# **Federated Learning by elimination of Downstream Redundancy**

---

## Problem of Central Model

- Central Model 🢩 Communication & Computation Bottleneck
- FL 🢩 Privacy Concerns


---

## Solutions

- Homomorphic Encryption 🢩 Reduce Communication Costs
- Only Download a Subset of the weights

---

# Paillier Encryption

 - Paillier is a type of public-key cryptosystem that supports additive homomorphic encryption.
 - Homomorphic encryption is a type of encryption that allows computations to be performed on encrypted data without having to decrypt it first
 - Additive homomorphic encryption specifically allows the addition of two encrypted values to yield an encrypted result that, when decrypted, is equal to the sum of the original values.

--- 

 
## Paillier Keys

### public key 🢩 encryption | private key 🢩 decryption 
---

<!--_class: lead -->
# Federated Learning with Heterogeneous Models
for On-device Malware Detection in IoT Networks

---

# Introduction

---

# Methods

- Evaluated the proposed framework platform on two aspects: algorithmic performance in the global model and comprehensive analysis of the system efficiency, including computational speed, communication cost, and memory cost.
The IoT node gathers the data it encounters in real-time in the form of executable application binaries.
We obtained more than 60K samples of executable files from VirusTotal [23].
These samples comprised six different classes, namely.
The 60K samples are split into 75% training data and 25% test data.
The 75% training data is split into three categories.
We implement the min-max normalization over training samples of all devices.
Training data is distributed in an imbalanced manner among the nodes.
The sample data and validation data reside on the FL server
Results
The end-to-end training time is analyzed when the bandwidth is 7.65 MB/s, and the uplink latency is 0.189 seconds (Table III).
Proposed FL 2881 seconds 2.7 milliseconds 54.51 seconds training time of the proposed FL framework with heterogeneous models is just 1.12× higher than the traditional FL.
The execution time per round is 2.21× higher than traditional FL.
The proposed technique yields an average testing latency of 2.7 milliseconds per sample of malware or benign, which is 1.63× faster than traditional FL.
The trade-off between end-to-end training time and time per round is acceptable since IoT devices with non-similar computational capabilities and resources can provide heterogeneous models for performing ondevice malware detection

---

# Conclusion
* Novel architecture of the FL paradigm that enables FL with heterogeneous ML models.
* Proposed framework has 1.63× faster-testing latency in comparison to traditional FL.
* The performance metrics reported clearly illustrate that the proposed FL framework attains a performance elevation of 7% to 13% than traditional FL in terms of accuracy, TPR, and TNR.
* The paper evaluates and further investigates the impact of heterogeneity in the distribution of ML models on the proposed FL framework.
* The results show that despite including a variable number of heterogeneous models, our technique achieves significantly high malware detection performance with 90% to 96% accuracy and 3% to 5% FPR
---
<!--_class: lead -->
# Distributed Machine Learning
# vs
# Federated Machine Learning
---
## Quick Takeaways
* Distributed machine learning (DL) is a multinode approach that improves performance and accuracy on large data sets.
* Federated learning (FL) is a decentralized approach that preserves privacy and relevance on local data sets.
* FL does not share raw data, uses encryption, and works across different locations or businesses. DL does not have these features.

---
## Distributed Machine Learning
![bg left:30% 70%](https://www.machinelearningpro.org/wp-content/uploads/2022/11/Distributed-Machine-Learning.jpg)
The distributed machine learning algorithm creates training models using independent training on various nodes. The training on enormous amounts of data is accelerated by using a distributed training system. Scalability and online re-training are necessary because training time grows exponentially when using big data.

---
## Federated Machine Learning 

![bg right:30% 90%](https://www.machinelearningpro.org/wp-content/uploads/2022/11/Distributed-Machine-Learning2.jpg)

Federated learning utilizes methods from numerous fields of study, including distributed systems, machine learning, and privacy. It works best when on-device data is more pertinent than data stored on servers. Federated learning offers cutting-edge ML to edge devices without by default centralized data and privacy. It manages the unbalanced, non-Independent, and Identically Distributed (IID) data of the features in mobile devices.

---
## Significant Differences

* FL prohibits the transmission of direct raw data. Such a limitation does not apply to DL.
* FL typically makes use of encryption or other defensive strategies to guarantee privacy. FL assures that the raw data’s security and confidentiality will be maintained. In DL, safety is not as heavily emphasized.
* Federated learning makes use of methods from a variety of fields of study, including distributed systems, machine learning, and privacy.

---
# Websites
- [Building Multilayer Perceptron Models in PyTorch](https://machinelearningmastery.com/building-multilayer-perceptron-models-in-pytorch/)
- [Three Ways to Build a Neural Network in PyTorch](https://towardsdatascience.com/three-ways-to-build-a-neural-network-in-pytorch-8cea49f9a61a)
- [Example: Walk-Through PyTorch & MNIST](https://flower.dev/docs/example-walkthrough-pytorch-mnist.html)
- [Example: PyTorch - From Centralized To Federated](https://flower.dev/docs/example-pytorch-from-centralized-to-federated.html)
- [DEEP LEARNING -> FEDERATED LEARNING IN 10 LINES OF PYTORCH + PYSYFT](https://blog.openmined.org/upgrade-to-federated-learning-in-10-lines/)
- [Federated Learning: Collaborative Machine Learning with a Tutorial on How to Get Started](https://becominghuman.ai/federated-learning-collaborative-machine-learning-with-a-tutorial-on-how-to-get-started-2e7d286a204e)
- [Preserving Data Privacy in Deep Learning | Part 1](https://towardsdatascience.com/preserving-data-privacy-in-deep-learning-part-1-a04894f78029)

---

# Papers

- [Deep, Big, Simple Neural Nets for Handwritten Digit Recognition](https://doi.org/10.1162/NECO_a_00052)
- [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://doi.org/10.48550/arXiv.1602.05629)
- [A PyTorch Implementation of Federated Learning](https://doi.org/10.5281/zenodo.4321561)

---

# Repositories & Notebooks

- [Federated learning: basic concepts](https://developers.sherpa.ai/tutorials/flexibility-and-scalability/model/pytorch)
- [Federated Learning (PyTorch)](https://www.kaggle.com/code/puru98/federated-learning-pytorch)
- [Federated-Learning (PyTorch)](https://github.com/AshwinRJ/Federated-Learning-PyTorch)
- [Federated Learning](https://github.com/shaoxiongji/federated-learning)