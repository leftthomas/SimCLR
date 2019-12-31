import argparse
import math

import pandas as pd
import torch
import torch.optim as optim
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from model import Model


# train for one epoch, each branch focus on different parts, to learn unique features
def train(net, data_loader, train_optimizer):
    net.train()
    total_loss, total_num, data_num, train_bar = 0.0, 0, len(data_loader.dataset), tqdm(data_loader)
    for data, target, pos_index in train_bar:
        data, pos_index = data.to(gpu_ids[0]), pos_index.to(gpu_ids[0])
        train_optimizer.zero_grad()
        features = net(data)

        # sample negatives ---> [B, K+1]
        idx = torch.randint(high=data_num, size=(data.size(0), negative_num + 1)).to(gpu_ids[0])
        # make the first sample as positive
        idx[:, 0] = pos_index
        # select memory vectors from memory bank ---> [B, 1+K, E, D]
        samples = torch.index_select(memory_bank, dim=0, index=idx.view(-1)) \
            .view(data.size(0), -1, ensemble_size, feature_dim)
        # compute cos similarity between each feature vector and memory bank ---> [B, E, 1+K]
        sim_matrix = torch.bmm(
            samples.permute(0, 2, 1, 3).contiguous().view(data.size(0) * ensemble_size, -1, feature_dim),
            features.view(data.size(0) * ensemble_size, -1).unsqueeze(dim=-1)).view(data.size(0), ensemble_size, -1)
        out = torch.exp(sim_matrix / temperature)
        # Monte Carlo approximation ---> [1, E, 1]
        z = out.detach().mean(dim=[0, 2], keepdim=True) * data_num
        # [B, E, 1+K]
        output = out / z

        # compute loss
        # TODO: Add branch nce loss
        # log(p_t/(p_t+K*p_i)) ---> [B, E]
        p_d = (output.select(dim=-1, index=0) / (output.select(dim=-1, index=0) + negative_num / data_num)).log()
        # log(1-p_f/(p_f+K*p_i)) ---> [B, E, K]
        p_n = ((negative_num / data_num) / (output.narrow(dim=-1, start=1, length=negative_num)
                                            + negative_num / data_num)).log()
        # -E(P_d)-K*E(P_n)
        loss = - (p_d.sum() + p_n.sum()) / (data.size(0) * ensemble_size)
        loss.backward()
        train_optimizer.step()

        # update memory bank ---> [B, E, D]
        pos_samples = samples.select(dim=1, index=0)
        pos_samples = pos_samples * momentum + features.detach() * (1.0 - momentum)
        pos_samples = pos_samples.div(torch.norm(pos_samples, dim=-1, keepdim=True))
        memory_bank.index_copy_(dim=0, index=pos_index, source=pos_samples)

        total_num += data.size(0)
        total_loss += loss.item() * data.size(0)
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return total_loss / total_num


# test for one epoch, use weighted knn (k=top_k) to find the most similar images' label to assign the test image
def test(net, memory_data_loader, test_data_loader):
    net.eval()
    total_top1, total_top5, total_num, feature_bank, c = 0.0, 0.0, 0, [], len(memory_data_loader.dataset.classes)
    with torch.no_grad():
        # generate feature bank
        for data, target, _ in tqdm(memory_data_loader, desc='Feature extracting'):
            feature_bank.append(net(data.to(gpu_ids[0])))
        # [E, D, N]
        feature_bank = torch.cat(feature_bank).permute(1, 2, 0).contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets).to(gpu_ids[0])
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data, target, _ in test_bar:
            data, target = data.to(gpu_ids[0]), target.to(gpu_ids[0])
            output = net(data)

            total_num += data.size(0)
            # compute cos similarity between each feature vector and feature bank ---> [E, B, N]
            sim_matrix = torch.bmm(output.transpose(0, 1).contiguous(), feature_bank)
            # [E, B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=top_k, dim=-1)
            # [E, B, K]
            sim_labels = torch.gather(feature_labels.expand(ensemble_size, data.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / temperature).exp()

            # counts for each class
            one_hot_label = torch.zeros(ensemble_size * data.size(0) * top_k, c).to(gpu_ids[0])
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
    parser.add_argument('--ensemble_size', default=8, type=int, help='Ensemble model size')
    parser.add_argument('--feature_dim', default=16, type=int, help='Feature dim for each branch')
    parser.add_argument('--negative_num', default=4096, type=int, help='Negative sample number')
    parser.add_argument('--temperature', default=0.1, type=float, help='Temperature used in softmax')
    parser.add_argument('--momentum', default=0.5, type=float, help='Momentum used for the update of memory bank')
    parser.add_argument('--top_k', default=200, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--batch_size', default=128, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=200, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--with_random', action='store_true', help='With branch random weight or not')
    parser.add_argument('--gpu_ids', default='0,1,2,3,4,5,6,7', type=str, help='Selected gpu ids to use')

    # args parse
    args = parser.parse_args()
    model_type, share_type, ensemble_size = args.model_type, args.share_type, args.ensemble_size
    feature_dim, negative_num, temperature = args.feature_dim, args.negative_num, args.temperature
    momentum, top_k, batch_size, epochs = args.momentum, args.top_k, args.batch_size, args.epochs
    with_random, gpu_ids = args.with_random, [int(gpu) for gpu in args.gpu_ids.split(',')]

    # data prepare
    train_data = utils.CIFAR10Instance(root='data', train=True, transform=utils.train_transform, download=True)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    memory_data = utils.CIFAR10Instance(root='data', train=True, transform=utils.test_transform, download=True)
    memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    test_data = utils.CIFAR10Instance(root='data', train=False, transform=utils.test_transform, download=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # model setup and optimizer config
    model = nn.DataParallel(Model(model_type, share_type, ensemble_size, feature_dim, with_random).to(gpu_ids[0]),
                            device_ids=gpu_ids)
    optimizer = optim.SGD(model.parameters(), lr=0.03, momentum=0.9, weight_decay=5e-4)
    print("# trainable model parameters:", sum(param.numel() if param.requires_grad else 0
                                               for param in model.parameters()))
    lr_scheduler = MultiStepLR(optimizer, milestones=[int(epochs * 0.6), int(epochs * 0.8)], gamma=0.1)

    # init memory bank as random unit vector
    # TODO: Figure out why init memory bank like this
    std = 1. / math.sqrt(feature_dim / 3)
    # [N, E, D]
    memory_bank = ((torch.rand(len(train_data), ensemble_size, feature_dim) * 2 * std) - std).to(gpu_ids[0])

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
        lr_scheduler.step(epoch)
        if test_acc_1 > best_acc:
            best_acc = test_acc_1
            torch.save(model.module.state_dict(), 'epochs/{}_features_extractor.pth'.format(save_name_pre))
