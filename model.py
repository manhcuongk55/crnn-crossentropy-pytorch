import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.utils.data
from PIL import Image

from constant import HIDDEN_DIM, MAX_LEN, OUTPUT_DIM, BATCH_SIZE, VAL_IMG_PATH, IMG_PATH
from utils import one_hot_digits

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


class OCRDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, df, transforms=None):
        self.df = df
        self.images = self.df.values[:, 0]
        self.labels = self.df.values[:, 1]
        self.length = len(self.df.index)
        self.transforms = transforms

    def __getitem__(self, index):
        image_path = self.images[index]
        image = Image.open(f'{image_path}')
        if self.transforms:
            image = self.transforms(image)

        label = str(self.labels[index])
        label = one_hot_digits(label)

        return (image, label)

    def __len__(self):
        return self.length
