# imagenet64x64

## Introduction to the dataset

The link to the stored-in-image imagenet64x64 dataset. And a code in PyTorch with resnet/wrn for it.

The dataset is from [imagenet64x64](https://github.com/PatrykChrabaszcz/Imagenet32_Scripts). They downsampled the imagenet to 16x16, 32x32, and 64x64. Amazing work! The authors provides the data in 10 binary files like cifar10/cifar100. And it's not a natural way in PyTorch to load such big files(each binary file is about 1.6 GB and it costs about 20~30 seconds to load a file).

So I decompress the set of 64x64 into png files like the *imagenet*, and we can use ImageFolder and DataLoader in PyTorch to load the dataset.

### The link to the dataset

The size of compressed file is GB. I split them into 11 files and upload to Baidu Cloud Storage(I only know this one is free with > 10GB space). But I'm not sure the speed to download them.....

文件比较大， 10.6G， 我就分成了11份上传到百度上——我只知道这个提供10G以上的免费空间， 也懒得找了， 虽然我知道百度下载速度是个悲剧。。。

Wait a minute.

[imagenet64 baidu]()

## The training code

Then I use the code from PyTorch offical examples [imagenet](https://github.com/pytorch/examples/tree/master/imagenet) to train on the dataset.

```bash
python main.py -a resnet18 [imagenet-folder with train and val folders]
```

The results(use the default configuration) are coming:

| network   | GPU:0 | per epoch    | top1 accuracy(%) | top5 accuracy(%) |
|:---------:|:-----:|:------------:|:----------------:|:----------------:|
| resnet 18 | 1.50G | 3 min 10 sec |         -    |         -    |
| resnet 50 | 3.20G | 6 min 10 sec |         -    |         -    |


## Decompress code

I also upload the ipynb code about how to load the batch data and save to images. It requires numpy and imageio.

The main functions if from the authors. [load batch data](https://github.com/PatrykChrabaszcz/Imagenet32_Scripts/blob/master/WRNs_imagenet.py)

