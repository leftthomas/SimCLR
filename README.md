# SimCLR
A PyTorch implementation of SimCLR based on the paper [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709).

## Requirements
- [Anaconda](https://www.anaconda.com/download/)
- [PyTorch](https://pytorch.org)
```
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
```
- thop
```
pip install thop
```

## Dataset
`CIFAR10` dataset is used in this repo, the dataset will be downloaded into `data` directory by `PyTorch` automatically.

## Usage
### Train SimCLR
```
python main.py --batch_size 1024 --epochs 1000 
optional arguments:
--feature_dim                 Feature dim for latent vector [default value is 128]
--temperature                 Temperature used in softmax [default value is 0.5]
--k                           Top k most similar images used to predict the label [default value is 200]
--batch_size                  Number of images in each mini-batch [default value is 512]
--epochs                      Number of sweeps over the dataset to train [default value is 500]
```

### Linear Evaluation
```
python linear.py --batch_size 1024 --epochs 200 
optional arguments:
--model_path                  The pretrained model path [default value is 'results/128_0.5_200_512_500_model.pth']
--batch_size                  Number of images in each mini-batch [default value is 512]
--epochs                      Number of sweeps over the dataset to train [default value is 100]
```

## Results
There are some difference between this implementation and official implementation, the model (`ResNet50`) is trained on 
one NVIDIA TESLA V100(32G) GPU:
1. No `Gaussian blur` used;
2. `Adam` optimizer with learning rate `1e-3` is used to replace `LARS` optimizer;
3. No `Linear learning rate scaling` used;
4. No `Linear Warmup` and `CosineLR Schedule` used.

<table>
	<tbody>
		<!-- START TABLE -->
		<!-- TABLE HEADER -->
		<th>Evaluation</th>
		<th>Params (M)</th>
		<th>FLOPs (G)</th>
		<th>Feature Dim</th>
		<th>Batch Size</th>
		<th>Epoch Num</th>
		<th>τ</th>
		<th>K</th>
		<th>Top1 Acc %</th>
		<th>Top5 Acc %</th>
		<th>Download</th>
		<!-- TABLE BODY -->
		<tr>
			<td align="center">KNN</td>
			<td align="center">24.62</td>
			<td align="center">1.31</td>
			<td align="center">128</td>
			<td align="center">512</td>
			<td align="center">500</td>
			<td align="center">0.5</td>
			<td align="center">200</td>
			<td align="center">93.5</td>
			<td align="center">-</td>
			<td align="center"><a href="https://pan.baidu.com/s/1jP7zWezVPBZWx_9LjJCgWg">model</a>&nbsp;|&nbsp;xxi8</td>
		</tr>
		<tr>
			<td align="center">Linear</td>
			<td align="center">24.62</td>
			<td align="center">1.31</td>
			<td align="center">-</td>
			<td align="center">512</td>
			<td align="center">100</td>
			<td align="center">-</td>
			<td align="center">-</td>
			<td align="center"><b>93.5</b></td>
			<td align="center"><b>-</b></td>
			<td align="center"><a href="https://pan.baidu.com/s/1jP7zWezVPBZWx_9LjJCgWg">model</a>&nbsp;|&nbsp;xxi8</td>
		</tr>
	</tbody>
</table>

