import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ニューラルネットワークを作る
# FMNISTの分類

#デバイス
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# クラスの定義
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
model = NeuralNetwork().to(device) # インスタンスの作成
print(model) # ネットワークの構造を出力


X = torch.rand(1, 28, 28, device=device)
logits = model(X) # モデルにデータを渡す
pred_probab = nn.Softmax(dim=1)(logits) # 予測確率を取得
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}") #確率が一番大きいものを出力