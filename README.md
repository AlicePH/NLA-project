Skoltech NLA project: Optimal Layer-wise Decomposition
===================================

## What is this project about?

In this project, we conducted experiments on the efficient compression of neural network layers. We tried to achieve efficiency on each local layer as well as the whole network with minimal loss of prediction accuracy.

Based on theory provided by article: "Compressing Neural Networks: Towards Determining the Optimal Layer-wise Decomposition" by Lucas Liebenwein,  Alaa Maalouf, Oren Gal, Dan Feldman, Daniela Rus

We provide a solution that makes efficient decomposition on each layer. For this purpose we decompose each layer by folding the weight tensor into a matrix before applying SVD. The result pair of matrices is encoded as two separate layers. Scheme of working algorithm:


![12334](https://user-images.githubusercontent.com/98256321/209015877-86a3bcf8-5889-46c3-9bc4-f97e22ccf785.jpg)

We provide experiments for series of custom networks such as:
1) CNN
2) CNNLowRank
3) CaffeBNAlexNet
4) CaffeBNLowRankAlexNet

All of them yo can find in models folder.

The results we got are summarized in the next image:
![215431534](https://user-images.githubusercontent.com/98256321/209067114-9559ebdc-e261-47eb-a990-0a993d89710b.jpg)



## Quickstart

Cloning the repository with all required submodules:

    git clone https://github.com/AlicePH/NLA-project

    cd NLA-project

To run the models it is necessary to make the changes to the `main.py` file and choose the model, this file automatically runs the training loop for the selected model and computes the training time, for simplicity and visualization, we also provide a notebook file where it is possible to visualize the results.


