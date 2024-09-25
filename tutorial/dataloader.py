import torch
from torch import nn
from torch.utils.data import DataLoader 
from torchvision import datasets #torchtext torchaudioなど
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


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
# 反復処理を楽にしてくれる
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

# データローダーを使った反復
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
# 64分割されたデータのうちの1つを表示してくれる
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")
