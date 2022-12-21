import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np

def low_rank_dec(filters, rank):
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
    u.shape, s.shape, vh.shape

    U, S, V = u[:, :rank], s[:rank], vh[:rank, :]

    U_new = U 
    V_new = np.diag(S) @ V

    assert np.allclose(U_new @ V_new, U @ np.diag(S) @ V)


    U_layer = torch.Tensor(U_new).unsqueeze(dim=-1).unsqueeze(dim=-1)
    V_layer = torch.Tensor(V_new).reshape(rank, ck1k2[0], ck1k2[1], ck1k2[2])


    return U_layer, V_layer

def generate_low_rank_conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                             bias=True, padding_mode='zeros', rank=None, scheme='scheme_1'):
  if rank is None:
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                     dilation=dilation, groups=groups, bias=bias)
  if scheme == 'scheme_1':
    conv1 = nn.Conv2d(in_channels, out_channels, kernel_size)
    filters = conv1.weight
    u, v = low_rank_dec(filters, rank)

    l1 = nn.Conv2d(in_channels=in_channels,
                   out_channels=rank,
                   kernel_size=kernel_size,
                   stride=stride,
                   padding=padding,
                   dilation=dilation,
                   groups=groups,
                   bias=False
                   )
    l1.load_state_dict({'weight': v}, strict=False)
    l2 = nn.Conv2d(in_channels=rank, out_channels=out_channels,
                   kernel_size=1,
                   bias=bias)
    
    l2.load_state_dict({'weight': u}, strict=False)
    
    return nn.Sequential(OrderedDict([('V', l1), ('U', l2)]))
  elif scheme == 'scheme_2':
    if isinstance(kernel_size, int):
      kernel_size = [kernel_size, kernel_size]

    if isinstance(padding, int):
      padding = [padding, padding]
    if isinstance(stride, int):
      stride = [stride, stride]

    l1 = nn.Conv2d(in_channels=in_channels,
                   out_channels=rank,
                   kernel_size=(1, kernel_size[1]),
                   stride=(1, stride[1]),
                   padding=(0, padding[1]),
                   dilation=dilation,
                   groups=groups,
                   bias=False
                   )
    
    l2 = nn.Conv2d(in_channels=rank,
                   out_channels=out_channels,
                   kernel_size=(kernel_size[0], 1),
                   padding=(padding[0], 0),
                   stride=(stride[0], 1),
                   bias=bias)
    

    return nn.Sequential(OrderedDict([('V', l1), ('U', l2)]))

def generate_low_rank_linear(in_features, out_features, bias=True, rank=None):
  if rank is None:
    return nn.Linear(in_features, out_features, bias=bias)
  l1 = nn.Linear(in_features=in_features, out_features=rank, bias=False)
  l2 = nn.Linear(in_features=rank, out_features=out_features, bias=bias)
  return nn.Sequential(OrderedDict([('V', l1), ('U', l2)]))
