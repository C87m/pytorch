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
# 反復処理を楽にしてくれる
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)


for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# 使用するマシン
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# モデルの定義
# nn.Moduleを継承したクラスを作成
class NeuralNetwork(nn.Module):
    # ネットワークの層
    def __init__(self):
        super().__init__() #親クラス呼び出し
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    # ネットワークの通過方法
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)
