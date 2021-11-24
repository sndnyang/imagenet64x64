# imagenet64x64

## Introduction to the dataset

The link to the stored-in-image imagenet64x64 dataset. And a code in PyTorch with resnet/wrn for it.

The dataset is from [imagenet64x64](https://github.com/PatrykChrabaszcz/Imagenet32_Scripts). They downsampled the imagenet to 16x16, 32x32, and 64x64. Amazing work! The authors provides the data in 10 binary files like cifar10/cifar100. And it's not a natural way in PyTorch to load such big files(each binary file is about 1.6 GB and it costs about 20~30 seconds to load a file).

So I decompress the set of 64x64 into png files like the *imagenet*, and we can use ImageFolder and DataLoader in PyTorch to load the dataset.

### The link to the dataset

[imagenet64 baidu](https://pan.baidu.com/s/1zjDMT14st8Ih4fqpIGbgXw)

or

[google drive](https://drive.google.com/file/d/1GpGEiuBjQ-pDKdXpfimfHAHT316xkLHc/view?usp=sharing)

## The training code

Then I use the code from PyTorch offical examples [imagenet](https://github.com/pytorch/examples/tree/master/imagenet) to train on the dataset.
```
pip install torch==1.3.1 torchvision==0.4.2 tqdm
```

```bash
python main.py -a resnet18 [imagenet-folder with train and val folders]
```
wideresnet
···bash
CUDA_VISIBLE_DEVICES=3 python main_64.py -a wrn -depth 28 --k 2 -drop 0 -b 64  /shared/imagenet64png --epochs 40 --decay_epochs 25 35
···

The results(use the default configuration) are coming:

| network              | GPU:0 |  per epoch    | epochs | top1 accuracy(%) | top5 accuracy(%) |
|:--------------------:|:-----:|:-------------:|:------:|:----------------:|:----------------:|
| resnet 18            | 1.50G |  3 min 10 sec |    90  |       42.96      |        67.72     |
| resnet 50            | 3.20G |  6 min 10 sec |    90  |       51.96      |        75.95     |
| WRN-28-2,drop 0      | 2.71G |  23 min 46 sec|    40  |       60.02      |        83.04     |


TITAN RTX, imagenet32

| network              | Memory  | epoch |  per epoch    | speed     | total time | top1 accuracy(%) | top5 accuracy(%) |
|:--------------------:|:-------:|:-----:|:-------------:|:---------:|:----------:|:----------------:|:----------------:|
| resnet 50            | 1717MiB | 90    |    03:52      | 21.57it/s | 6:09:35    | 38.056 | 62.386 |
| WRN-28-2,drop 0      | 2601MiB | 40    |    05:17      | 15.78it/s | 3:49:15    | 46.606 | 71.806 |

Use the default epochs from [resnet imagenet](https://github.com/pytorch/examples/tree/master/imagenet) and [wide resnet imagenet64x64](https://github.com/meliketoy/wide-resnet.pytorch).  I found it's is much faster than the authors reported in their paper.

### Training Curves

ImageNet32, WRN-28-2-drop0

![WeChat Image_20211123215246](https://user-images.githubusercontent.com/2310591/143163104-f954d468-18c5-49ef-bdc7-0fd1a0c42b6a.png)
![WeChat Image_20211123215309](https://user-images.githubusercontent.com/2310591/143163106-e148f1ce-198c-4517-8743-1572082784c5.png)
![WeChat Image_20211123215319](https://user-images.githubusercontent.com/2310591/143163113-57ce2b8d-9169-43c3-9b58-e0af5df93d46.png)

### Wide ResNet

The model is from [wide resnet](https://github.com/meliketoy/wide-resnet.pytorch) which gets more stars than the official repo. 

And I changed it a bit following the implementation of the imagenet64x64 authors [WRN imagenet](https://github.com/PatrykChrabaszcz/Imagenet32_Scripts/blob/master/WRNs_imagenet.py)

## Decompress code

I also upload the ipynb code about how to load the batch data and save to images.

上传了notebook代码， 32x32, 16x16的数据就不弄了。

```
numpy
imageio
```

The main function is from the authors' code [load batch data](https://github.com/PatrykChrabaszcz/Imagenet32_Scripts/blob/master/WRNs_imagenet.py)

