import torch
from torchvision import transforms


class DuplicatedCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        img1 = img.copy()
        img2 = img.copy()
        for t in self.transforms:
            img1 = t(img1)
            img2 = t(img2)
        return img1, img2


train_transform = DuplicatedCompose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.2), ratio=(0.8, 1.2)),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8),
    transforms.RandomHorizontalFlip(),
    transforms.RandomGrayscale(p=0.25),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def queue_data(data, k):
    return torch.cat([data, k], dim=0)


def dequeue_data(data, k=4096):
    if len(data) > k:
        return data[-k:]
    else:
        return data


def momentum_update(model_q, model_k, beta=0.999):
    """ model_k = beta * model_k + (1 - beta) model_q """
    for p1, p2 in zip(model_q.parameters(), model_k.parameters()):
        p2.data.mul_(beta).add_(1 - beta, p1.detach().data)
