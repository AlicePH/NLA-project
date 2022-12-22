Skoltech NLA project: Optimal Layer-wise Decomposition
===================================

## What is this project about?

In this project, we conducted experiments on the efficient compression of neural network layers. We tried to achieve efficiency on each local layer as well as the whole network with minimal loss of prediction accuracy.

Based on theory provided by article: "Compressing Neural Networks: Towards Determining the Optimal Layer-wise Decomposition" by Lucas Liebenwein,  Alaa Maalouf, Oren Gal, Dan Feldman, Daniela Rus

We provide a solution that makes efficient decomposition on each layer. For this purpose we decompose each layer by folding the weight tensor into a matrix before applying SVD. The result pair of matrices is encoded as two separate layers. Scheme of working algorithm:


![12334](https://user-images.githubusercontent.com/98256321/209015877-86a3bcf8-5889-46c3-9bc4-f97e22ccf785.jpg)

We provide experiments for series of custom networks such as:
1) LargeCNN
2) Net_exper
3) Net_Large_Experiments

All of them yo can find in models directory.

## Quickstart

Cloning the repository with all required submodules:

    git clone https://github.com/AlicePH/NLA-project


    cd NLA-project
    
    # Install requirements
    pip install -r requirements.txt
    
    #Train
    python main.py [--model] [--rank] [--scheme] --train
    ::Parameter model describes model and can be CaffeBNAlexNet or CaffeBNLowRankAlexNet
    ::Parameter rank take integer parameter which describes rank. You could see rank descriprion in file src/rank.py
    ::Parameter scheme take two different schemes - scheme_1 or scheme_2
    ::Example
    python main.py --model='CaffeBNAlexNet' --rank=1 --scheme='scheme_2' --train
    
    #Test
    python main.py [--model] [--rank] [--scheme] --test
    ::Parameter model describes model and can be CaffeBNAlexNet or CaffeBNLowRankAlexNet
    ::Parameter rank take integer parameter which describes rank. You could see rank descriprion in file src/rank.py
    ::Parameter scheme take two different schemes - scheme_1 or scheme_2
    ::Example
    python main.py --model='CaffeBNAlexNet' --rank=1 --scheme='scheme_2' --test



