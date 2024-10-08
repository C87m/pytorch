import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

# トレーニングデータセットのダウンロード
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


labels_map = {
    0: "T-shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

figure = plt.figure(figsize=(8,8)) # サイズ8*8の画像
cols, rows = 3, 3 # 縦横に表示する画像の数
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item() # ランダムな整数を生成
    img, label = training_data[sample_idx] # 画像とラベルを取得
    figure.add_subplot(rows, cols, i) # 画像のプロット
    plt.title(labels_map[label]) #ラベル設定
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray") # 1次元のリストを削除 グレースケール

plt.show() # 表示
