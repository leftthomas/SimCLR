import torch
from torchvision import transforms


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(64, scale=(0.8, 1.2), ratio=(0.8, 1.2)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()])

test_transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor()])


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
