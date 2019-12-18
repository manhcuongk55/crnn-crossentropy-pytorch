import torch
import torch.utils.data
import torch.utils.data

from constant import MAX_LEN


def one_hot_embedding(labels, num_classes):
    y = torch.eye(num_classes, dtype=torch.long)
    return y[labels]


def one_hot_digits(digits):
    cleaned_digits = []
    for digit in digits:
        digit = digit.replace('.', '10')
        digit = digit.replace('-', '11')
        cleaned_digits.append(int(digit))

    cleaned_digits.append(12)
    cleaned_digits += [0] * (MAX_LEN - len(cleaned_digits))

    return one_hot_embedding(cleaned_digits, 13)
