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


The results(use the default configuration) are coming:

| network              | GPU:0 |  per epoch    | epochs | top1 accuracy(%) | top5 accuracy(%) |
|:--------------------:|:-----:|:-------------:|:------:|:----------------:|:----------------:|
| resnet 18            | 1.50G |  ??? |    90  |       42.96      |        67.72     |
| resnet 50            | 3.20G |  ??? |    90  |       51.96      |        75.95     |
| WRN-36-2,drop 0.3    | 8.72G |  ??? |    40  |       ...        |        ...       |
| WRN-36-0.5,drop 0.3  | 2.91G |  ??? |    40  |       W          |        W         |

Use the default epochs from [resnet imagenet](https://github.com/pytorch/examples/tree/master/imagenet) and [wide resnet imagenet64x64](https://github.com/meliketoy/wide-resnet.pytorch)

### Wide ResNet

The model is from [wide resnet](https://github.com/meliketoy/wide-resnet.pytorch) which gets more stars than the official repo. When will the PyTorch team add WideResNet into torchvision.models?

And I changed it a bit following the implementation of the imagenet64x64 authors [WRN imagenet](https://github.com/PatrykChrabaszcz/Imagenet32_Scripts/blob/master/WRNs_imagenet.py)

## Decompress code

I also upload the ipynb code about how to load the batch data and save to images.

上传了notebook代码， 32x32, 16x16的数据就不弄了。

```
numpy
imageio
```

The main functions if from the authors. [load batch data](https://github.com/PatrykChrabaszcz/Imagenet32_Scripts/blob/master/WRNs_imagenet.py)

