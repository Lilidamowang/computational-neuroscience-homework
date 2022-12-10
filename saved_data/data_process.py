import numpy as np
import matplotlib.pyplot as plt
import torch

pa_his = np.load(file='/home/data/yjgroup/lyl/projects/MMR-VQA/saved_data/pa_his.npy', allow_pickle=True)
print(pa_his.shape)
print(type(pa_his[0]))
print(pa_his[0].shape)
x = range(0,7)
y = pa_his[0][1].squeeze(dim=-1)
print(x, y)