import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# モデルのレイヤー

# 画像
input_image = torch.rand(3,28,28) # 28*28の画像3つ
print(input_image.size())

# 2次元画像を配列化
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())

# 線形変換
layer1 = nn.Linear(in_features=28*28, out_features=20) # 128->20
hidden1 = layer1(flat_image)
print(hidden1.size())

# ReLU関数
print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")

# 順序付け
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10) # この順番で渡される
)
input_image = torch.rand(3,28,28)
logits = seq_modules(input_image)

# Softmax関数によるスケーリング
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)