import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.models as models
from torchvision import transforms
from tqdm import tqdm

from constant import HIDDEN_DIM, BATCH_SIZE, VAL_TRANSFORMS, EPOCHS, PRETRAIN, VAL_LABELS, LABELS, TRANSFORMS
from model import CRNN, OCRDatasetVal, OCRDataset, CyclicLR

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def TD_criterion(x, y):
    CE = nn.CrossEntropyLoss()
    loss = 0
    for i in range(x.size(1)):
        loss += CE(x[:, i, :], y[:, i])
    return loss


aux_criterion = nn.CrossEntropyLoss()

# In[ ]:


pretrain_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('pretrain/', train=True, download=True,
                                                                         transform=transforms.Compose(
                                                                             [transforms.Grayscale(3),
                                                                              transforms.Resize((224, 224)),
                                                                              transforms.ToTensor(),
                                                                              transforms.Normalize(
                                                                                  mean=[0.485, 0.456, 0.406],
                                                                                  std=[0.229, 0.224, 0.225])])),
                                              batch_size=32, shuffle=True)

resnet50 = models.resnet50(pretrained=True)
resnet50.fc = nn.Linear(2048, 10)
resnet50 = resnet50.to(device)
optimizer = optim.Adam(resnet50.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    for x, y in tqdm(pretrain_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = resnet50(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

torch.save(resnet50.state_dict(), 'MNIST_pretrain.pt')

resnet50 = models.resnet50(pretrained=True)
resnet50.fc = nn.Linear(2048, 10)
if PRETRAIN:
    resnet50.load_state_dict(torch.load('MNIST_pretrain.pt'))
resnet50.fc = nn.Linear(2048, HIDDEN_DIM)
model = CRNN(resnet50)
model = model.to(device)

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
step_size = int((EPOCHS * 234) / 4 / 2)
scheduler = CyclicLR(optimizer, 0.001, 0.01, step_size_up=step_size)
print(step_size)

train_df = pd.read_csv(LABELS, sep=';')
train_dataset = OCRDataset(train_df, transforms=TRANSFORMS)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

val_df = pd.read_csv(VAL_LABELS, sep=';')
val_dataset = OCRDatasetVal(val_df, transforms=VAL_TRANSFORMS)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1)

for epoch in range(EPOCHS):
    running_td_loss = 0
    running_aux_loss = 0
    accuracies = []
    sample_labels = 0
    sample_yhat = 0

    model = model.train()

    for features, target in tqdm(train_dataloader):
        features, target = features.to(device), target.to(device)

        optimizer.zero_grad()

        _, number = model(features, target)

        labels = torch.max(target, -1)[1]

        td_loss = TD_criterion(number, labels)
        loss = td_loss

        loss.backward()

        running_td_loss += td_loss.item()

        optimizer.step()

        y_hat = torch.max(number, -1)[1].cpu().numpy()
        labels = labels.cpu().numpy()

        sample_labels = labels
        sample_yhat = y_hat

        acc = []
        for j in range(y_hat.shape[0]):
            acc.append((y_hat[j, :] == labels[j, :]).all())

        accuracies.append(np.sum(acc) / BATCH_SIZE)
        scheduler.step()

    model = model.eval()
    val_acc = []

    for i, (features, label) in enumerate(val_dataloader):
        features = features.to(device)
        number = model.predict(features)

        if label[0] == number:
            val_acc.append(True)
        else:
            val_acc.append(False)

    print('[Epoch {}] TD_Loss: {:.5f} Accuracy: {:.5f} Val_Accuracy: {:.5f}'.format(epoch, running_td_loss / len(
        train_dataloader), np.mean(accuracies), np.sum(val_acc) / len(val_acc)))
    print(sample_labels, sample_yhat)
torch.save(model.state_dict(), 'pretrained2.pt')
