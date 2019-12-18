import os
from glob import glob

import PIL
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import transforms

from model import CRNN

print("start loading resnet50")
resnet50 = models.resnet50(pretrained=True)
resnet50.fc = nn.Linear(2048, 10)
PRETRAIN = True
HIDDEN_DIM = 512
EPOCHS = 42

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

VAL_TRANSFORMS = transforms.Compose([
    transforms.Pad((0, 26)),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

img = PIL.Image.open(image_files[12])
plt.imshow(img)

result = model.predict(VAL_TRANSFORMS(img).unsqueeze(0).to(device))
print(result)
