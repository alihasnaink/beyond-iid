from __future__ import annotations

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def make_transform(train: bool, randaugment=False, num_ops=2, magnitude=9):
    operations = []
    if train:
        operations += [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
        if randaugment:
            operations.append(transforms.RandAugment(num_ops=num_ops, magnitude=magnitude))
    operations += [transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)]
    return transforms.Compose(operations)


class CIFARArrayDataset(Dataset):
    def __init__(self, data, targets, indices, transform, classes=CIFAR10_CLASSES, identifier_prefix="cifar10_train"):
        self.data, self.targets, self.indices = data, targets, list(map(int, indices))
        self.transform, self.classes, self.identifier_prefix = transform, classes, identifier_prefix

    def __len__(self): return len(self.indices)

    def __getitem__(self, position):
        index = self.indices[position]
        label = int(self.targets[index])
        image = self.transform(Image.fromarray(self.data[index]))
        return image, label, f"{self.identifier_prefix}:{index}", self.classes[label]
