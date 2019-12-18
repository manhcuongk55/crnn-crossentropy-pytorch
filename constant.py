from torchvision import transforms

IMG_PATH = 'mnist_images/training_set/'
VAL_IMG_PATH = 'mnist_images/validation_set/'
LABELS = 'mnist_images/training_set_values.txt'
VAL_LABELS = 'mnist_images/validation_set_values.txt'

HIDDEN_DIM = 512
OUTPUT_DIM = 13
MAX_LEN = 10

EPOCHS = 42
BATCH_SIZE = 32
PRETRAIN = True

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
