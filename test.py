import argparse
import warnings
from collections import OrderedDict

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

import utils
from model import Net

warnings.filterwarnings("ignore")


def train(model, train_loader, optimizer, epoch):
    model.train()
    total_loss, total_true, n_data, train_bar = 0, 0, 0, tqdm(train_loader)
    for data, target in train_bar:
        x, _ = data
        y = model(x.to('cuda'))
        loss = cross_entropy_loss(y, target.to('cuda'))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        n_data += len(x)
        total_loss += loss.item() * len(x)
        pred = torch.argmax(y, dim=-1)
        total_true += torch.sum((pred.cpu() == target).float()).item()
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.6f} Acc:{:.2f}%'
                                  .format(epoch, epochs, total_loss / n_data, total_true / n_data * 100))

    return total_loss / n_data, total_true / n_data * 100


def test(model, test_loader, epoch):
    model.eval()
    total_loss, total_top1, total_top5, n_data, test_bar = 0, 0, 0, 0, tqdm(test_loader)
    with torch.no_grad():
        for data, target in test_bar:
            y = model(data.to('cuda'))
            loss = cross_entropy_loss(y, target.to('cuda'))
            n_data += len(data)
            total_loss += loss.item() * len(data)
            total_top1 += torch.sum((torch.topk(y, k=1, dim=-1)[1].cpu() == target.unsqueeze(dim=-1)).any(
                dim=-1).float()).item()
            total_top5 += torch.sum((torch.topk(y, k=5, dim=-1)[1].cpu() == target.unsqueeze(dim=-1)).any(
                dim=-1).float()).item()
            test_bar.set_description('Test Epoch: [{}/{}] Loss: {:.6f} Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                     .format(epoch, epochs, total_loss / n_data, total_top1 / n_data * 100,
                                             total_top5 / n_data * 100))

    return total_loss / n_data, total_top1 / n_data * 100, total_top5 / n_data * 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test MoCo')
    parser.add_argument('--data_path', type=str, default='/home/data/imagenet/ILSVRC2012', help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=256, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', type=int, default=100, help='Number of sweeps over the dataset to train')
    parser.add_argument('--model', type=str, default='epochs/features_extractor_resnet18_128_65536.pth',
                        help='Features extractor file')

    args = parser.parse_args()
    data_path, batch_size, epochs, model_path = args.data_path, args.batch_size, args.epochs, args.model
    model_type, features_dim = model_path.split('_')[-3], int(model_path.split('_')[-2])
    train_data = datasets.CIFAR10(root='data', train=True, transform=utils.train_transform, download=True)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_data = datasets.CIFAR10(root='data', train=False, transform=utils.test_transform, download=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    features_extractor = Net(model_type, features_dim)
    features_extractor.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    for param in features_extractor.parameters():
        param.requires_grad = False
    model = nn.Sequential(OrderedDict(
        [('features_extractor', features_extractor), ('classifier', nn.Linear(features_dim, len(train_data.classes)))]))
    model = model.to('cuda')
    optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3)
    cross_entropy_loss = nn.CrossEntropyLoss()
    print("# trainable parameters:", sum(param.numel() if param.requires_grad else 0 for param in model.parameters()))
    results = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc@1': [], 'test_acc@5': []}

    best_acc = 0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train(model, train_loader, optimizer, epoch)
        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        test_loss, test_acc1, test_acc5 = test(model, test_loader, epoch)
        results['test_loss'].append(test_loss)
        results['test_acc@1'].append(test_acc1)
        results['test_acc@5'].append(test_acc5)
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('results/model_{}_{}_results.csv'.format(model_type, features_dim), index_label='epoch')
        if test_acc1 > best_acc:
            best_acc = test_acc1
            torch.save(model.state_dict(), 'epochs/model_{}_{}.pth'.format(model_type, features_dim))
