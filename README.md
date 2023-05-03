![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Huggingface-white?style=for-the-badge)
![Scikit](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626.svg?&style=for-the-badge&logo=Jupyter&logoColor=white)
![Pytorch](https://img.shields.io/badge/Pytorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)

# Project Name : SCV
SCV: Smart Contracts Vulnerability.

## Research Name : Revolutionizing Smart Contract Security: Deep Learning Techniques for Vulnerability Detection and Classification
### Background:
Smart contracts are self-executing contracts with the terms of the agreement between buyer and seller being directly written into lines of code. Ethereum smart contracts, written in Solidity, enable developers to implement complex business logic solutions on the blockchain. However, Solidity also increases the chance of bugs and code vulnerabilities that can be exploited by malicious users, leading to significant losses in digital assets.

### Objective:
The primary objective of this research project is to revolutionize smart contract security by exploring deep learning techniques, particularly Convolutional Neural Networks (CNNs), for the detection and classification of vulnerabilities in smart contracts deployed on the Ethereum main net. The project aims to provide an efficient solution to the problem of spending long hours searching for potential vulnerabilities in smart contracts.

### Methodology:
To achieve the objective, we will create a large-scale dataset of more than 100k smart contracts labeled using the Slither static analyzer, which passes the code through a number of rule-based detectors and returns a JSON file containing details about where those detectors found vulnerabilities. The 38 detectors that found a match in our dataset will be mapped to the following 5 classes: access-control, arithmetic, reentrancy, unchecked-calls, and others.

We will use deep learning techniques based on CNNs to detect and classify vulnerabilities in smart contracts. A program's executable file is transformed into a grayscale image, which is then fed into a convolutional architecture to extract relevant features and patterns. Given the similarities between a program's executable file and the bytecode of a smart contract, we investigate whether similar techniques could be useful for detecting vulnerabilities in Solidity code.

### Expected Outcome:
We expect to create a robust model capable of detecting and classifying vulnerabilities in smart contracts. The model will be trained and tested on the large-scale dataset of smart contracts, and we will provide an LSTM baseline, Conv2D models, and a Conv1D model to help users detect potential vulnerabilities in their smart contracts quickly. We will also provide error analysis to help users understand where the models may be prone to errors.

### Conclusion:
We believe that our research will help developers identify potential vulnerabilities in their smart contracts and prevent significant losses in digital assets. We plan to make our dataset and models available on the HuggingFace hub, and we encourage researchers and developers to explore them further.

## How to install
In order to run this code on your machine, simply code the repository with:
```sh
git clone https://github.com/bilzkaist/SCV.git
```

then install `torch` and `torchvision` with CUDA support according to your system's requirements (see [Pytorch docs](https://pytorch.org/get-started/locally/) for more info). Finally, you can install all the other project requirements by running the following commands from the folder _smart-contracts-vulnerabilities_.
```sh
pip install -U pip
pip install -r requirements.txt
```

In order to start a training, simply create a .yaml config file according to [these](https://github.com/bilzkaist/SCV/blob/main/docs/config.md) specifications and then run

```sh
python main.py path/to/your/config/file.yaml
```

## References
<a id="1">[1]</a> Huang, T. H.-D. (2018). Hunting the Ethereum Smart Contract: Color-inspired Inspection of Potential Attacks. ArXiv:1807.01868 [Cs]. http://arxiv.org/abs/1807.01868

<a id="2">[2]</a> Hwang, S.-J., Choi, S.-H., Shin, J., & Choi, Y.-H. (2022). CodeNet: Code-Targeted Convolutional Neural Network Architecture for Smart Contract Vulneratbility Detection. IEEE Access, 1–1. https://doi.org/10.1109/ACCESS.2022.3162065

<a id="3">[3]</a> Lin, W.-C., & Yeh, Y.-R. (2022). Efficient Malware Classification by Binary Sequences with One-Dimensional Convolutional Neural Networks. Mathematics, 10(4), 608. https://doi.org/10.3390/math10040608

<a id="4">[4]</a> Yashavant, C. S., Kumar, S., & Karkare, A. (2022). ScrawlD: A Dataset of Real World Ethereum Smart Contracts Labelled with Vulnerabilities. ArXiv:2202.11409 [Cs]. http://arxiv.org/abs/2202.11409

<a id="5">[5]</a> Durieux, T., Ferreira, J. F., Abreu, R., & Cruz, P. (2020). Empirical Review of Automated Analysis Tools on 47,587 Ethereum Smart Contracts. Proceedings of the ACM/IEEE 42nd International Conference on Software Engineering, 530–541. https://doi.org/10.1145/3377811.3380364
