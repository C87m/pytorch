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

#ハイパーパラメータ
learning_rate = 1e-3 # 学習率
batch_size = 64 # バッチサイズ
epochs = 5 # エポック数

# モデルのトレーニング
# 損失関数
loss_fn = nn.CrossEntropyLoss()
# オプティマイザー
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # 予測誤差
        pred = model(X)
        loss = loss_fn(pred, y)

        # 誤差伝播法
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# モデルの学習の確認
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


epochs = 5 # エポック数
# 反復
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

# モデルの保存
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")

# モデルの呼び出し
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth", weights_only=True))

# ラベル
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1] # 画像とラベルを取得
with torch.no_grad():
    x = x.to(device)
    pred = model(x) # モデルに入力
    predicted, actual = classes[pred[0].argmax(0)], classes[y] # 予測と実際のラベルを比較
    print(f'Predicted: "{predicted}", Actual: "{actual}"')