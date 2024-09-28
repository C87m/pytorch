import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

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


figure = plt.figure(figsize=(8,8)) # サイズ8*8の画像
cols, rows = 3, 3 # 縦横に表示する画像の数

grid = ImageGrid(figure, 111,
                 nrows_ncols = (rows, cols),
                 axes_pad = 0)

for ax, i in zip(grid, range(1, cols * rows + 1)):
    sample_idx = torch.randint(len(training_data), size=(1,)).item() # ランダムな整数を生成
    img, label = training_data[sample_idx] # 画像とラベルを取得
    ax.imshow(img.squeeze(), cmap="gray")
    


plt.show() # 表示

