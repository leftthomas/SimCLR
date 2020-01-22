import argparse

import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from model import Model


# train for one epoch, each branch focus on different parts, to learn unique features
def train(net, data_loader, train_optimizer):
    global z
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for data, target, pos_index in train_bar:
        data = data.to(gpu_ids[0])
        train_optimizer.zero_grad()
        features = net(data)

        # randomly generate M+1 sample indexes for each batch ---> [B, M+1]
        idx = torch.randint(high=n, size=(data.size(0), m + 1))
        # make the first sample as positive
        idx[:, 0] = pos_index
        # select memory vectors from memory bank ---> [B, 1+M, E, D]
        samples = torch.index_select(memory_bank, dim=0, index=idx.view(-1)) \
            .view(data.size(0), -1, ensemble_size, feature_dim)
        # compute cos similarity between each feature vector and memory bank ---> [B, E, 1+M]
        sim_matrix = torch.bmm(samples.to(device=features.device).permute(0, 2, 1, 3).contiguous()
                               .view(data.size(0) * ensemble_size, -1, feature_dim),
                               features.view(data.size(0) * ensemble_size, -1).unsqueeze(dim=-1)) \
            .view(data.size(0), ensemble_size, -1)
        out = torch.exp(sim_matrix / temperature)
        # Monte Carlo approximation ---> [1, E, 1], use the approximation derived from initial batches as z
        if z is None:
            z = out.detach().mean(dim=[0, 2], keepdim=True) * n
        # compute P(i|v) ---> [B, E, 1+M]
        output = out / z

        # compute loss
        # compute log(h(i|v))=log(P(i|v)/(P(i|v)+M*P_n(i))) ---> [B, E]
        p_d = (output.select(dim=-1, index=0) / (output.select(dim=-1, index=0) + m / n)).log()
        # compute log(1-h(i|v'))=log(1-P(i|v')/(P(i|v')+M*P_n(i))) ---> [B, E, M]
        p_n = ((m / n) / (output.narrow(dim=-1, start=1, length=m) + m / n)).log()
        # compute J_NCE(θ)=-E(P_d)-M*E(P_n)
        # TODO: Add branch nce loss
        loss = - (p_d.sum() + p_n.sum()) / (data.size(0) * ensemble_size)
        loss.backward()
        train_optimizer.step()

        # update memory bank ---> [B, E, D]
        pos_samples = samples.select(dim=1, index=0)
        pos_samples = features.detach().cpu() * momentum + pos_samples * (1.0 - momentum)
        pos_samples = F.normalize(pos_samples, dim=-1)
        memory_bank.index_copy_(dim=0, index=pos_index, source=pos_samples)

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
    parser.add_argument('--model_type', default='resnet18', type=str,
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnext50_32x4d'], help='Backbone type')
    parser.add_argument('--share_type', default='layer1', type=str,
                        choices=['none', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4'], help='Shared module type')
    parser.add_argument('--ensemble_size', default=8, type=int, help='Ensemble branch size')
    parser.add_argument('--feature_dim', default=16, type=int, help='Feature dim for each branch')
    parser.add_argument('--m', default=4096, type=int, help='Negative sample number')
    parser.add_argument('--temperature', default=0.1, type=float, help='Temperature used in softmax')
    parser.add_argument('--momentum', default=0.5, type=float, help='Momentum used for the update of memory bank')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--batch_size', default=128, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=200, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--with_random', action='store_true', help='With branch random weight or not')
    parser.add_argument('--gpu_ids', default='0,1,2,3,4,5,6,7', type=str, help='Selected gpu ids to use')

    # args parse
    args = parser.parse_args()
    model_type, share_type, ensemble_size = args.model_type, args.share_type, args.ensemble_size
    feature_dim, m, temperature = args.feature_dim, args.m, args.temperature
    momentum, k, batch_size, epochs = args.momentum, args.k, args.batch_size, args.epochs
    with_random, gpu_ids = args.with_random, [int(gpu) for gpu in args.gpu_ids.split(',')]

    # data prepare
    train_data = utils.CIFAR10Instance(root='data', train=True, transform=utils.train_transform, download=True)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)
    memory_data = utils.CIFAR10Instance(root='data', train=True, transform=utils.test_transform, download=True)
    memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=8)
    test_data = utils.CIFAR10Instance(root='data', train=False, transform=utils.test_transform, download=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=8)

    # model setup and optimizer config
    model = nn.DataParallel(Model(model_type, share_type, ensemble_size, feature_dim, with_random).to(gpu_ids[0]),
                            device_ids=gpu_ids)
    optimizer = optim.SGD(model.parameters(), lr=0.03, momentum=0.9, weight_decay=5e-4)
    print("# trainable model parameters:", sum(param.numel() if param.requires_grad else 0
                                               for param in model.parameters()))
    lr_scheduler = MultiStepLR(optimizer, milestones=[int(epochs * 0.6), int(epochs * 0.8)], gamma=0.1)

    # z as normalizer, init with None, c as num of train class, n as num of train data
    z, c, n = None, len(memory_data.classes), len(train_data)
    # init memory bank as unit random vector ---> [N, E, D]
    memory_bank = F.normalize(torch.randn(n, ensemble_size, feature_dim), dim=-1)

    # training loop
    results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': []}
    save_name_pre = 'cifar10_{}_{}_{}_{}_{}'.format(model_type, share_type, ensemble_size, feature_dim, with_random)
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer)
        results['train_loss'].append(train_loss)
        test_acc_1, test_acc_5 = test(model, memory_loader, test_loader)
        results['test_acc@1'].append(test_acc_1)
        results['test_acc@5'].append(test_acc_5)
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('results/{}_features_extractor_results.csv'.format(save_name_pre), index_label='epoch')
        lr_scheduler.step()
        if test_acc_1 > best_acc:
            best_acc = test_acc_1
            torch.save(model.module.state_dict(), 'epochs/{}_features_extractor.pth'.format(save_name_pre))
