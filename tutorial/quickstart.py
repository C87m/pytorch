import torch
from torch import nn
from torch.utils.data import DataLoader 
from torchvision import datasets #torchtext torchaudioなど
from torchvision.transforms import ToTensor

# トレーニングデータセットのダウンロード
# データセットにはtransformとtarget_transformの２つの引数が含まれる
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# テストデータセットのダウンロード
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)


batch_size = 64 # データの分割数

# データローダーの作成
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)


for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break
