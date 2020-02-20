import argparse

import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from model import Model


# train for one epoch, each branch focus on different parts, to learn unique features
def train(net, data_loader, train_optimizer):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for data, target, pos_index in train_bar:
        data = data.to(gpu_ids[0])
        train_optimizer.zero_grad()
        features = net(data)

        # compute loss
        loss = - (p_d.sum() + p_n.sum()) / (data.size(0) * ensemble_size)
        loss.backward()
        train_optimizer.step()

        total_num += data.size(0)
        total_loss += loss.item() * data.size(0)
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return total_loss / total_num


# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def test(net, memory_data_loader, test_data_loader):
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, target, _ in tqdm(memory_data_loader, desc='Feature extracting'):
            feature_bank.append(net(data.to(gpu_ids[0])))
        # [E, D, N]
        feature_bank = torch.cat(feature_bank).permute(1, 2, 0).contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data, target, _ in test_bar:
            data, target = data.to(gpu_ids[0]), target.to(gpu_ids[0])
            output = net(data)

            total_num += data.size(0)
            # compute cos similarity between each feature vector and feature bank ---> [E, B, N]
            sim_matrix = torch.bmm(output.transpose(0, 1).contiguous(), feature_bank)
            # [E, B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
            # [E, B, K]
            sim_labels = torch.gather(feature_labels.expand(ensemble_size, data.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / temperature).exp()

            # counts for each class
            one_hot_label = torch.zeros(ensemble_size * data.size(0) * k, c, device=sim_labels.device)
            # [E*B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(ensemble_size, data.size(0), -1, c)
                                    * sim_weight.unsqueeze(dim=-1), dim=[0, 2])

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                     .format(epoch, epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))

    return total_top1 / total_num * 100, total_top5 / total_num * 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MVC')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.1, type=float, help='Temperature used in softmax')
    parser.add_argument('--batch_size', default=512, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=100, type=int, help='Number of sweeps over the dataset to train')

    # args parse
    args = parser.parse_args()
    feature_dim, temperature, batch_size, epochs = args.feature_dim, args.temperature, args.batch_size, args.epochs

    # data prepare
    train_data = utils.CIFAR10Instance(root='data', train=True, transform=utils.train_transform, download=True)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16)
    memory_data = utils.CIFAR10Instance(root='data', train=True, transform=utils.test_transform, download=True)
    memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=16)
    test_data = utils.CIFAR10Instance(root='data', train=False, transform=utils.test_transform, download=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16)

    # model setup and optimizer config
    model = Model(feature_dim).cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # training loop
    results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': []}
    save_name_pre = '{}_{}_{}_{}'.format(feature_dim, temperature, batch_size, epochs)
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer)
        results['train_loss'].append(train_loss)
        test_acc_1, test_acc_5 = test(model, memory_loader, test_loader)
        results['test_acc@1'].append(test_acc_1)
        results['test_acc@5'].append(test_acc_5)
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('results/{}_statistics.csv'.format(save_name_pre), index_label='epoch')
        if test_acc_1 > best_acc:
            best_acc = test_acc_1
            torch.save(model.state_dict(), 'results/{}_model.pth'.format(save_name_pre))
