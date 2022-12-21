# -*- coding: utf-8 -*-
"""layer_decomposer.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ERNyoyS1KpNCEbc3ge7ORDemvC-9LqIE
"""

import torch
import numpy as np

def low_rank_dec(filters, rank=7):
    """
    decomposer of layer with SVD truncated
    """

    f = filters.shape[0]


    w = []
    for filter in range(f):
      w_flat = filters[filter].flatten()
      _dim = len(w_flat)
      w.append(w_flat.reshape(1, _dim))


    W = torch.cat(tuple(w), 0)
    W = W.detach().numpy()

    ck1k2 = list(filters.shape[1:])


    u, s, vh = np.linalg.svd(W)
    

    U, S, V = u[:, :rank], s[:rank], vh[:rank, :]

    U_new = U 
    V_new = np.diag(S) @ V

    U_layer = torch.Tensor(U_new).unsqueeze(dim=-1).unsqueeze(dim=-1)
    V_layer = torch.Tensor(V_new).reshape(rank, ck1k2[0], ck1k2[1], ck1k2[2])


    return U_layer, V_layer