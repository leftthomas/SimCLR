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
    transforms.RandomResizedCrop(64, scale=(0.8, 1.2), ratio=(0.8, 1.2)),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8),
    transforms.RandomHorizontalFlip(),
    transforms.RandomGrayscale(p=0.25),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

test_transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
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
    param_k = model_k.state_dict()
    param_q = model_q.named_parameters()
    for n, q in param_q:
        if n in param_k:
            param_k[n].data.copy_(beta * param_k[n].data + (1 - beta) * q.data)
    model_k.load_state_dict(param_k)
