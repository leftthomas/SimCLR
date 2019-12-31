import argparse

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

import utils
from model import Model


# train for one epoch, use original class to train
def train(model, train_loader, optimizer, epoch):
    model.train()
    total_loss, total_true, n_data, train_bar = 0.0, 0.0, 0, tqdm(train_loader)
    for data, target in train_bar:
        data, target = data.to(gpu_ids[0]), target.to(gpu_ids[0])
        features = F.normalize(features_extractor(data), dim=-1).view(len(data), -1)
        optimizer.zero_grad()
        output = model(features)
        loss = cross_entropy_loss(output, target)
        loss.backward()
        optimizer.step()

        n_data += len(data)
        total_loss += loss.item() * len(data)
        total_true += torch.sum((torch.argmax(output, dim=-1) == target).float()).item()
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.6f} Acc:{:.2f}%'
                                  .format(epoch, epochs, total_loss / n_data, total_true / n_data * 100))

    return total_loss / n_data, total_true / n_data * 100


# test for on epoch, same as traditional method
def test(model, test_loader, epoch):
    model.eval()
    total_loss, total_top1, total_top5, n_data, test_bar = 0.0, 0.0, 0.0, 0, tqdm(test_loader)
    with torch.no_grad():
        for data, target in test_bar:
            data, target = data.to(gpu_ids[0]), target.to(gpu_ids[0])
            features = F.normalize(features_extractor(data), dim=-1).view(len(data), -1)
            output = model(features)
            loss = cross_entropy_loss(output, target)

            n_data += len(data)
            total_loss += loss.item() * len(data)
            total_top1 += torch.sum((torch.topk(output, k=1, dim=-1)[1] == target.unsqueeze(dim=-1)).any(
                dim=-1).float()).item()
            total_top5 += torch.sum((torch.topk(output, k=5, dim=-1)[1] == target.unsqueeze(dim=-1)).any(
                dim=-1).float()).item()
            test_bar.set_description('Test Epoch: [{}/{}] Loss: {:.4f} Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                     .format(epoch, epochs, total_loss / n_data, total_top1 / n_data * 100,
                                             total_top5 / n_data * 100))

    return total_loss / n_data, total_top1 / n_data * 100, total_top5 / n_data * 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Shadow Mode')
    parser.add_argument('--batch_size', default=128, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=200, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--model', type=str, default='epochs/cifar10_resnet18_layer1_1_128_features_extractor.pth',
                        help='Features extractor file')
    parser.add_argument('--gpu_ids', default='0,1,2,3,4,5,6,7', type=str, help='Selected gpu')

    # args parse and data prepare
    args = parser.parse_args()
    batch_size, epochs, model_path = args.batch_size, args.epochs, args.model
    _, model_type, share_type, ensemble_size, meta_class_size, _, _ = model_path.split('_')
    gpu_ids = [int(gpu) for gpu in args.gpu_ids.split(',')]
    train_data = datasets.CIFAR10(root='data', train=True, transform=utils.train_transform, download=True)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)
    test_data = datasets.CIFAR10(root='data', train=False, transform=utils.test_transform, download=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=8)

    # model setup and meta id config
    features_extractor = Model(int(ensemble_size), int(meta_class_size), model_type, share_type)
    features_extractor.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    features_extractor = nn.DataParallel(features_extractor.to(gpu_ids[0]), device_ids=gpu_ids)
    for param in features_extractor.parameters():
        param.requires_grad = False
    model = nn.DataParallel(
        nn.Linear(int(ensemble_size) * int(meta_class_size), len(train_data.classes)).to(gpu_ids[0]),
        device_ids=gpu_ids)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    cross_entropy_loss = nn.CrossEntropyLoss()
    print("# trainable parameters:", sum(param.numel() if param.requires_grad else 0 for param in model.parameters()))
    results = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc@1': [], 'test_acc@5': []}
    save_name_pre = '{}_{}_{}_{}'.format(model_type, share_type, ensemble_size, meta_class_size)

    # training loop
    best_acc = 0.0
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
        data_frame.to_csv('results/cifar10_{}_model_results.csv'.format(save_name_pre), index_label='epoch')
        if test_acc1 > best_acc:
            best_acc = test_acc1
            torch.save(model.module.state_dict(), 'epochs/cifar10_{}_model.pth'.format(save_name_pre))
