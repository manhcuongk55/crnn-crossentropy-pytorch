import os
from glob import glob

import PIL
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

from constant import HIDDEN_DIM, VAL_TRANSFORMS, PRETRAIN
from model import CRNN

print("start loading resnet50")
resnet50 = models.resnet50(pretrained=True)
resnet50.fc = nn.Linear(2048, 10)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if PRETRAIN:
    resnet50.load_state_dict(torch.load('MNIST_pretrain.pt', map_location='cpu'))
resnet50.fc = nn.Linear(2048, HIDDEN_DIM)
model = CRNN(resnet50)
model = model.to(device)

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
val_path = 'mnist_processed_images/validation_set/'
image_files = []
for file in glob(os.path.join(val_path, '*')):
    image_files.append(file)
path_img = image_files[12]
print("Path to image {}".format(path_img))
img = PIL.Image.open(path_img)
plt.imshow(img)

result = model.predict(VAL_TRANSFORMS(img).unsqueeze(0).to(device))
print(result)
