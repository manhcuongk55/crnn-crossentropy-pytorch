#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[14]:


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.models as models
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# # Hyperparameters

# In[15]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# In[16]:


IMG_PATH = 'mnist_processed_images/training_set/'
VAL_IMG_PATH = 'mnist_processed_images/validation_set/'
LABELS = 'mnist_processed_images/training_set_values.txt'
VAL_LABELS = 'mnist_processed_images/validation_set_values.txt'

# In[17]:


HIDDEN_DIM = 512
OUTPUT_DIM = 13
MAX_LEN = 10

EPOCHS = 42
BATCH_SIZE = 32
PRETRAIN = True

# In[18]:


TRANSFORMS = transforms.Compose([
    transforms.Pad((0, 26)),
    transforms.Resize((224, 224)),
    transforms.RandomAffine(10, translate=(0.1, 0.1), scale=(0.95, 1.1), shear=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

VAL_TRANSFORMS = transforms.Compose([
    transforms.Pad((0, 26)),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# # Helper functions

# In[19]:


def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes, dtype=torch.long)
    return y[labels]


# In[20]:


def one_hot_digits(digits):
    cleaned_digits = []
    for digit in digits:
        digit = digit.replace('.', '10')
        digit = digit.replace('-', '11')
        cleaned_digits.append(int(digit))

    cleaned_digits.append(12)
    cleaned_digits += [0] * (MAX_LEN - len(cleaned_digits))

    return one_hot_embedding(cleaned_digits, 13)


# In[21]:


import math


class _LRScheduler(object):
    def __init__(self, optimizer, last_epoch=-1):
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
            last_epoch = 0
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.step(last_epoch)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class CyclicLR(_LRScheduler):
    """Sets the learning rate of each parameter group according to
    cyclical learning rate policy (CLR). The policy cycles the learning
    rate between two boundaries with a constant frequency, as detailed in
    the paper `Cyclical Learning Rates for Training Neural Networks`_.
    The distance between the two boundaries can be scaled on a per-iteration
    or per-cycle basis.

    Cyclical learning rate policy changes the learning rate after every batch.
    `step` should be called after a batch has been used for training.

    This class has three built-in policies, as put forth in the paper:
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each
        cycle iteration.

    This implementation was adapted from the github repo: `bckenstler/CLR`_

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        base_lr (float or list): Initial learning rate which is the
            lower boundary in the cycle for each parameter group.
        max_lr (float or list): Upper learning rate boundaries in the cycle
            for each parameter group. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function.
        step_size_up (int): Number of training iterations in the
            increasing half of a cycle. Default: 2000
        step_size_down (int): Number of training iterations in the
            decreasing half of a cycle. If step_size_down is None,
            it is set to step_size_up. Default: None
        mode (str): One of {triangular, triangular2, exp_range}.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
            Default: 'triangular'
        gamma (float): Constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
            Default: 1.0
        scale_fn (function): Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            If specified, then 'mode' is ignored.
            Default: None
        scale_mode (str): {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle).
            Default: 'cycle'
        cycle_momentum (bool): If ``True``, momentum is cycled inversely
            to learning rate between 'base_momentum' and 'max_momentum'.
            Default: True
        base_momentum (float or list): Initial momentum which is the
            lower boundary in the cycle for each parameter group.
            Default: 0.8
        max_momentum (float or list): Upper momentum boundaries in the cycle
            for each parameter group. Functionally,
            it defines the cycle amplitude (max_momentum - base_momentum).
            The momentum at any cycle is the difference of max_momentum
            and some scaling of the amplitude; therefore
            base_momentum may not actually be reached depending on
            scaling function. Default: 0.9
        last_epoch (int): The index of the last batch. This parameter is used when
            resuming a training job. Since `step()` should be invoked after each
            batch instead of after each epoch, this number represents the total
            number of *batches* computed, not the total number of epochs computed.
            When last_epoch=-1, the schedule is started from the beginning.
            Default: -1

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = torch.optim.CyclicLR(optimizer)
        >>> data_loader = torch.utils.data.DataLoader(...)
        >>> for epoch in range(10):
        >>>     for batch in data_loader:
        >>>         train_batch(...)
        >>>         scheduler.step()


    .. _Cyclical Learning Rates for Training Neural Networks: https://arxiv.org/abs/1506.01186
    .. _bckenstler/CLR: https://github.com/bckenstler/CLR
    """

    def __init__(self,
                 optimizer,
                 base_lr,
                 max_lr,
                 step_size_up=2000,
                 step_size_down=None,
                 mode='triangular',
                 gamma=1.,
                 scale_fn=None,
                 scale_mode='cycle',
                 cycle_momentum=True,
                 base_momentum=0.8,
                 max_momentum=0.9,
                 last_epoch=-1):

        self.optimizer = optimizer

        base_lrs = self._format_param('base_lr', optimizer, base_lr)
        if last_epoch == -1:
            for lr, group in zip(base_lrs, optimizer.param_groups):
                group['lr'] = lr

        self.max_lrs = self._format_param('max_lr', optimizer, max_lr)

        step_size_up = float(step_size_up)
        step_size_down = float(step_size_down) if step_size_down is not None else step_size_up
        self.total_size = step_size_up + step_size_down
        self.step_ratio = step_size_up / self.total_size

        if mode not in ['triangular', 'triangular2', 'exp_range'] and scale_fn is None:
            raise ValueError('mode is invalid and scale_fn is None')

        self.mode = mode
        self.gamma = gamma

        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = self._triangular_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = self._triangular2_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = self._exp_range_scale_fn
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode

        self.cycle_momentum = cycle_momentum
        if cycle_momentum:
            if 'momentum' not in optimizer.defaults:
                raise ValueError('optimizer must support momentum with `cycle_momentum` option enabled')

            base_momentums = self._format_param('base_momentum', optimizer, base_momentum)
            if last_epoch == -1:
                for momentum, group in zip(base_momentums, optimizer.param_groups):
                    group['momentum'] = momentum
        self.base_momentums = list(map(lambda group: group['momentum'], optimizer.param_groups))
        self.max_momentums = self._format_param('max_momentum', optimizer, max_momentum)

        super(CyclicLR, self).__init__(optimizer, last_epoch)

    def _format_param(self, name, optimizer, param):
        """Return correctly formatted lr/momentum for each param group."""
        if isinstance(param, (list, tuple)):
            if len(param) != len(optimizer.param_groups):
                raise ValueError("expected {} values for {}, got {}".format(
                    len(optimizer.param_groups), name, len(param)))
            return param
        else:
            return [param] * len(optimizer.param_groups)

    def _triangular_scale_fn(self, x):
        return 1.

    def _triangular2_scale_fn(self, x):
        return 1 / (2. ** (x - 1))

    def _exp_range_scale_fn(self, x):
        return self.gamma ** (x)

    def get_lr(self):
        """Calculates the learning rate at batch index. This function treats
        `self.last_epoch` as the last batch index.

        If `self.cycle_momentum` is ``True``, this function has a side effect of
        updating the optimizer's momentum.
        """
        cycle = math.floor(1 + self.last_epoch / self.total_size)
        x = 1. + self.last_epoch / self.total_size - cycle
        if x <= self.step_ratio:
            scale_factor = x / self.step_ratio
        else:
            scale_factor = (x - 1) / (self.step_ratio - 1)

        lrs = []
        for base_lr, max_lr in zip(self.base_lrs, self.max_lrs):
            base_height = (max_lr - base_lr) * scale_factor
            if self.scale_mode == 'cycle':
                lr = base_lr + base_height * self.scale_fn(cycle)
            else:
                lr = base_lr + base_height * self.scale_fn(self.last_epoch)
            lrs.append(lr)

        if self.cycle_momentum:
            momentums = []
            for base_momentum, max_momentum in zip(self.base_momentums, self.max_momentums):
                base_height = (max_momentum - base_momentum) * scale_factor
                if self.scale_mode == 'cycle':
                    momentum = max_momentum - base_height * self.scale_fn(cycle)
                else:
                    momentum = max_momentum - base_height * self.scale_fn(self.last_epoch)
                momentums.append(momentum)
            for param_group, momentum in zip(self.optimizer.param_groups, momentums):
                param_group['momentum'] = momentum

        return lrs


# # Dataset

# In[ ]:


class OCRDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, df, transforms=None):
        self.df = df
        self.images = self.df.values[:, 0]
        self.labels = self.df.values[:, 1]
        self.length = len(self.df.index)
        self.transforms = transforms

    def __getitem__(self, index):
        image_path = self.images[index] + '.jpg'
        image = Image.open(f'{IMG_PATH}{image_path}')
        if self.transforms:
            image = self.transforms(image)

        label = str(self.labels[index])
        label = one_hot_digits(label)

        return (image, label)

    def __len__(self):
        return self.length


# In[ ]:


class OCRDatasetVal(torch.utils.data.dataset.Dataset):
    def __init__(self, df, transforms=None):
        self.df = df
        self.images = self.df.values[:, 0]
        self.labels = self.df.values[:, 1]
        self.length = len(self.df.index)
        self.transforms = transforms

    def __getitem__(self, index):
        image_path = self.images[index]
        image = Image.open(f'{VAL_IMG_PATH}{image_path}')
        if self.transforms:
            image = self.transforms(image)

        label = str(self.labels[index])

        return (image, label)

    def __len__(self):
        return self.length


# # Network

# In[ ]:


class CRNN(nn.Module):
    def __init__(self, backbone):
        super(CRNN, self).__init__()
        self.backbone = backbone
        self.dropout1 = nn.Dropout(0.2)
        self.linear2 = nn.Linear(HIDDEN_DIM, MAX_LEN)
        self.lstm = nn.LSTM(OUTPUT_DIM, HIDDEN_DIM, batch_first=True)
        self.out = nn.Linear(HIDDEN_DIM, OUTPUT_DIM)

    def forward(self, x, target):
        target = target.float()
        # Activation function?
        latent = F.relu(self.backbone(x))
        length = self.linear2(self.dropout1(latent))

        inputs = torch.zeros(BATCH_SIZE, 1, OUTPUT_DIM).to(device)
        hidden = (latent.unsqueeze(0), torch.zeros(1, BATCH_SIZE, HIDDEN_DIM).to(device))
        number = []

        for i in range(MAX_LEN):
            output, hidden = self.lstm(inputs, hidden)
            # Residual Connection?
            # hidden = (hidden[0]+latent.unsqueeze(0), hidden[1])
            digit = self.out(output[:, -1, :])
            number.append(digit.unsqueeze(0))
            inputs = target[:, i, :].unsqueeze(1)

        return length, torch.cat(number, 0).transpose(0, 1)

    def to_num(self, number):
        clean_number = []
        for index in number:
            if index == 10:
                char = '.'
            elif index == 11:
                char = '-'
            else:
                char = str(index)
            clean_number.append(char)
        return ''.join(clean_number)

    def predict(self, x):
        latent = F.relu(self.backbone(x))

        inputs = torch.zeros(1, 1, OUTPUT_DIM).to(device)
        hidden = (latent.unsqueeze(0), torch.zeros(1, 1, HIDDEN_DIM).to(device))
        number = []

        for i in range(MAX_LEN):
            output, hidden = self.lstm(inputs, hidden)
            # hidden = (hidden[0]+latent.unsqueeze(0), hidden[1])
            digit_prob = self.out(output[:, -1, :])
            index = torch.max(digit_prob, -1)[1][0]
            if index == 12:
                break

            inputs = torch.zeros((1, 1, OUTPUT_DIM)).to(device)
            inputs[0, 0, index] = 1
            number.append(index.item())

        return self.to_num(number)

    # # Training


# In[ ]:


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

# In[ ]:


torch.save(resnet50.state_dict(), 'MNIST_pretrain.pt')

# In[ ]:


resnet50 = models.resnet50(pretrained=True)
resnet50.fc = nn.Linear(2048, 10)
if PRETRAIN:
    resnet50.load_state_dict(torch.load('MNIST_pretrain.pt'))
resnet50.fc = nn.Linear(2048, HIDDEN_DIM)
model = CRNN(resnet50)
model = model.to(device)

# In[ ]:


optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
step_size = int((EPOCHS * 234) / 4 / 2)
scheduler = CyclicLR(optimizer, 0.001, 0.01, step_size_up=step_size)
print(step_size)

# In[ ]:


# optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)


# In[ ]:


train_df = pd.read_csv(LABELS, sep=';')
train_dataset = OCRDataset(train_df, transforms=TRANSFORMS)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# In[ ]:


val_df = pd.read_csv(VAL_LABELS, sep=';')
val_dataset = OCRDatasetVal(val_df, transforms=VAL_TRANSFORMS)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1)

# In[ ]:


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

# In[ ]:


torch.save(model.state_dict(), 'pretrained2.pt')

# In[ ]:


resnet50 = models.resnet50(pretrained=True)
resnet50.fc = nn.Linear(2048, HIDDEN_DIM)
model = CRNN(resnet50)
model.load_state_dict(torch.load('pretrained2.pt'))

model = model.to(device)

# In[ ]:


val_df = pd.read_csv(VAL_LABELS, sep=';')
val_dataset = OCRDatasetVal(val_df, transforms=VAL_TRANSFORMS)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1)
model = model.eval()

# In[ ]:


val_df = pd.read_csv(VAL_LABELS, sep=';')
val_dataset = OCRDatasetVal(val_df, transforms=VAL_TRANSFORMS)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1)
model = model.eval()

# In[ ]:


numbers = []
model = model.eval()
for i, features in enumerate(val_dataloader):
    features = features.to(device)
    number = model.predict(features)

    numbers.append(number)
    print(i, number)

# In[ ]:


val_df['value'] = numbers
val_df.to_csv('output.txt', index=None, sep=';')

# In[ ]:


import matplotlib.pyplot as plt
import os
from glob import glob
import PIL

val_path = 'mnist_processed_images/validation_set/'
image_files = []
for file in glob(os.path.join(val_path, '*')):
    image_files.append(file)

# In[ ]:


img = PIL.Image.open(image_files[12])
# print(image_files[3])
plt.imshow(img)

# In[ ]:


model.predict(VAL_TRANSFORMS(img).unsqueeze(0).to(device))
