# Deep Learning for Deconvolution of scRNA-seq Data 
A code repo containing a potential framework for carrying out experiments on demultiplexing data from pooled experimental designs without any type of genotype information of the pooled samples.

## Introduction
 To be filled

## Installation

All models are implemented using Pytorch, along with many other smaller packages. Models can be trained on CPU but it would be beneficial to use GPU. To take care of everything at once, we recommend 
using the conda package management library. Specifically, 
[miniconda3](https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh), as it is lightweight and fast to install.
If you have an existing miniconda3 installation please start at step 3. 
If you want to  install both conda and the required packages, please run:
 1. ```wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh```
 2. Go through the installation.
 3. Activate conda
 4. conda create -n <your_environment_name> python=3.6.
 5. conda activate <your_environment_name>
 6. Then run ```bash install.sh```

The install.sh file is used to install all the packages required. You could modify it to install the pytorch version of your preference.

## Overview of code:
All models require NxM numpy arrays as inputs, where N refers to the number of cells and M refers to the number of variants.

- [Datasets.py](Datasets.py): Contains a function to create the Dataset in a way that can be used by pytorch's DataLoader.
- [utils.py](utils.py): Containing useful functions for the analysis.
- [Networks](https://github.com/IoannisStournaras/Deep-Learning-for-Deconvolution-of-scRNA-seq-Data/tree/master/Networks): Different architectures tested for demultiplexing scRNA-seq. More informations can be obtained by reading the following papers:
   1. Improved Deep Embedded Clustering - [IDEC](https://www.ijcai.org/proceedings/2017/0243.pdf)
   2. Variational Deep Embedding - [Vade](https://arxiv.org/pdf/1611.05148.pdf) 
   3. Variational Autoencoder for probabilistic NMF - [PAE-NMF](https://openreview.net/pdf?id=BJGjOi09t7)
   4. Binary Matrix Factorization - [BMF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4470263&tag=1)
   5. Adversarial Auto-Encoder - [AAE](https://arxiv.org/pdf/1511.05644.pdf)

        
