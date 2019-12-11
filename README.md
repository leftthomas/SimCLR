# MoCo
A PyTorch implementation of MoCo based on the paper [Momentum Contrast for Unsupervised Visual Representation Learning](https://arxiv.org/abs/1911.05722).

## Requirements
- [Anaconda](https://www.anaconda.com/download/)
- [PyTorch](https://pytorch.org)
```
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
```

## Dataset
`ILSVRC2012` dataset is used in this repo, the dataset should be downloaded from [ImageNet](http://image-net.org/challenges/LSVRC/2012/)
and extracted as following:
```
{data_root}/
  train/
    n01693334/
      # JPEG files 
    n02018207/
    ......
    n04286575/
    n04596742/
  val/
    n01693334/
      # JPEG files 
    n02018207/
    ......
    n04286575/
    n04596742/
```

## Usage
### Train Features Extractor
```
python train.py --epochs 50 --dictionary_size 4096
optional arguments:
--data_path                   Path to dataset [default value is '/home/data/imagenet/ILSVRC2012']
--model_type                  Backbone type [default value is 'resnet18'] (choices=['resnet18', 'resnet50'])
--batch_size                  Number of images in each mini-batch [default value is 256]
--epochs                      Number of sweeps over the dataset to train [default value is 200]
--features_dim                Dim of features for each image [default value is 128]
--dictionary_size             Size of dictionary [default value is 65536]
```

### Train Model
```
python test.py --epochs 100 --batch_size 512
optional arguments:
--data_path                   Path to dataset [default value is '/home/data/imagenet/ILSVRC2012']
--batch_size                  Number of images in each mini-batch [default value is 256]
--epochs                      Number of sweeps over the dataset to train [default value is 100]
--model                       Features extractor file [default value is 'epochs/features_extractor_resnet18_128_65536.pth']
```

## Results
There are some difference between this implementation and official implementation:
1. The training epoch is `50`;
2. The `batch size` is `256` for `resnet18` backbone, `128` for `resnet50` backbone.

<table>
	<tbody>
		<!-- START TABLE -->
		<!-- TABLE HEADER -->
		<th>Name</th>
		<th>train time (s/iter)^*</th>
		<th>train mem (GB)^*</th>
		<th>train time (s/iter)</th>
		<th>train mem (GB)</th>
		<th>Top1 Acc %</th>
		<th>Top5 Acc %</th>
		<th>download link</th>
		<!-- TABLE BODY -->
		<!-- ROW: r18 -->
		<tr>
			<td align="center">ResNet18</td>
			<td align="center">0.11</td>
			<td align="center">11.14</td>
			<td align="center">80.49</td>
			<td align="center">53.92</td>
			<td align="center">42.71</td>
			<td align="center">68.69</td>
			<td align="center"><a href="https://pan.baidu.com/s/1jP7zWezVPBZWx_9LjJCgWg">model</a>&nbsp;|&nbsp;xxi8</td>
		</tr>
		<!-- ROW: r50 -->
		<tr>
			<td align="center">ResNet50</td>
			<td align="center">1.55</td>
			<td align="center">17.92</td>
			<td align="center">81.16</td>
			<td align="center">54.54</td>
			<td align="center">43.61</td>
			<td align="center">69.50</td>
			<td align="center"><a href="https://pan.baidu.com/s/1BeGS7gckGAczd1euB55EFA">model</a>&nbsp;|&nbsp;1jhd</td>
		</tr>
	</tbody>
</table>
