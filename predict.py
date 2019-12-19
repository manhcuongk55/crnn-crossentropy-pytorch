import PIL
import torch
import torch.nn as nn
import torchvision.models as models

from constant import HIDDEN_DIM, VAL_TRANSFORMS
from model import CRNN

print("start loading resnet50")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

resnet50 = models.resnet50(pretrained=True)
resnet50.fc = nn.Linear(2048, HIDDEN_DIM)
model = CRNN(resnet50)
model.load_state_dict(torch.load('pretrained2.pt', map_location='cpu'))
model = model.to(device)

img = PIL.Image.open('mnist_images/validation_set/test.jpg')
result = model.predict(VAL_TRANSFORMS(img).unsqueeze(0).to(device))
print(result)
img.show()
