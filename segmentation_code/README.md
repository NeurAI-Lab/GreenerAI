# How to train the segmentation network?

```bash
cd segmentation_code
INDIR = /tmp/input/cityscapes_processed/
OUTDIR = /tmp/output/
```

## Baseline experiment

``` bash
cd master/experiments/segmentation/
python train_with_data_parallel.py --model bisenet
                                   --backbone resnet18
                                   --batch-size 16
                                   --data-folder $INDIR
                                   --lr 0.025
                                   --optimizer sgd
                                   --momentum 0.9
                                   --lr-scheduler poly
                                   --checkname cs512pre-base
                                   --epochs 1000
                                   --save-dir $OUTDIR
                                   --dataset citys
                                   --multiple-GPUs False
                                   --save-interval 10
                                   --base-size 1024
                                   --crop-size 512
                                   --pretrained
                                   --em
cd ../../..
```

## One-branch

``` bash
cd bisenet-one-branch/experiments/segmentation/
python train_with_data_parallel.py --model bisenet
                                   --backbone resnet18
                                   --batch-size 16
                                   --data-folder $INDIR
                                   --lr 0.025
                                   --optimizer sgd
                                   --momentum 0.9
                                   --lr-scheduler poly
                                   --checkname cs512pre-1branch
                                   --epochs 1000
                                   --save-dir $OUTDIR
                                   --dataset citys
                                   --multiple-GPUs False
                                   --save-interval 10
                                   --base-size 1024
                                   --crop-size 512
                                   --pretrained
                                   --em
cd ../../..
```

## Progressive resizing

``` bash
cd master/experiments/segmentation/
python train_with_data_parallel.py --model bisenet
                                   --backbone resnet18
                                   --batch-size 16
                                   --data-folder $INDIR
                                   --lr 0.025
                                   --optimizer sgd
                                   --momentum 0.9
                                   --lr-scheduler poly
                                   --checkname cs512-resize-250-epochs-1500
                                   --epochs 1000
                                   --epochs_per_resize 250
                                   --save-dir $OUTDIR
                                   --dataset citys
                                   --multiple-GPUs False
                                   --save-interval 10
                                   --base-size 1024
                                   --crop-size 512
                                   --pretrained
                                   --workers 16
                                   --em
cd ../../..
```

## Random gradient

``` bash
cd master/experiments/segmentation/
python train_with_data_parallel.py --model bisenet
                                   --backbone resnet18
                                   --batch-size 16
                                   --data-folder $INDIR
                                   --lr 0.025
                                   --optimizer sgd
                                   --momentum 0.9
                                   --lr-scheduler poly
                                   --checkname random-gradients
                                   --epochs 1000
                                   --save-dir $OUTDIR
                                   --dataset citys
                                   --multiple-GPUs False
                                   --save-interval 10
                                   --base-size 1024
                                   --crop-size 512
                                   --pretrained
                                   --em
cd ../../..
```

## Spatial bottleneck

``` bash
cd master/experiments/segmentation/
python train_with_data_parallel.py --model bisenet
                                   --backbone sp_b_resnet18
                                   --batch-size 16
                                   --data-folder $INDIR
                                   --lr 0.025
                                   --optimizer sgd
                                   --momentum 0.9
                                   --lr-scheduler poly
                                   --checkname spatial-bottleneck
                                   --epochs 1000
                                   --save-dir $OUTDIR
                                   --dataset citys
                                   --multiple-GPUs False
                                   --save-interval 10
                                   --base-size 1024
                                   --crop-size 512
                                   --em
cd ../../..
```

## Switchable normalization

``` bash
cd switchable-normalization/experiments/segmentation/
python train_with_data_parallel.py --model bisenet
                                   --backbone resnet18
                                   --batch-size 16
                                   --data-folder $INDIR
                                   --lr 0.025
                                   --optimizer sgd
                                   --momentum 0.9
                                   --lr-scheduler poly
                                   --checkname cs512-switchnorm
                                   --epochs 1000
                                   --save-dir $OUTDIR
                                   --dataset citys
                                   --multiple-GPUs False
                                   --save-interval 10
                                   --base-size 1024
                                   --crop-size 512
                                   --pretrained
                                   --em
                                   --norm-layer sn
cd ../../..
```

## Cyclical Learning Rate

``` bash
cd master/experiments/segmentation/
python train_with_data_parallel.py --model bisenet
                                   --backbone resnet18
                                   --batch-size 16
                                   --data-folder $INDIR
                                   --lr 0.0001
                                   --clr-max 0.01
                                   --clr-stepsize 50
                                   --optimizer sgd
                                   --momentum 0.9
                                   --lr-scheduler clr
                                   --checkname cs512pre-clr
                                   --epochs 1000
                                   --save-dir $OUTDIR
                                   --dataset citys
                                   --multiple-GPUs False
                                   --save-interval 10
                                   --base-size 1024
                                   --crop-size 512
                                   --pretrained
                                   --em
cd ../../..
```

## Poly LR 1/2 Epochs

``` bash
cd master/experiments/segmentation/
python train_with_data_parallel.py --model bisenet
                                   --backbone resnet18
                                   --batch-size 16
                                   --data-folder $INDIR
                                   --lr 0.025
                                   --optimizer sgd
                                   --momentum 0.9
                                   --lr-scheduler poly
                                   --checkname cs512pre-e500
                                   --epochs 500
                                   --save-dir $OUTDIR
                                   --dataset citys
                                   --multiple-GPUs False
                                   --save-interval 10
                                   --base-size 1024
                                   --crop-size 512
                                   --pretrained
                                   --em
cd ../../..
```

## Mixup

``` bash
cd master/experiments/segmentation/
python train_with_data_parallel.py --model bisenet
                                   --backbone resnet18
                                   --batch-size 16
                                   --data-folder $INDIR
                                   --lr 0.025
                                   --optimizer sgd
                                   --momentum 0.9
                                   --lr-scheduler poly
                                   --checkname cs512pre-mixup
                                   --epochs 1000
                                   --save-dir $OUTDIR
                                   --dataset citys
                                   --multiple-GPUs False
                                   --save-interval 10
                                   --base-size 1024
                                   --crop-size 512
                                   --pretrained
                                   --em
                                   --mixup-alpha 1.0
cd ../../..
```

## RAdam

``` bash
cd master/experiments/segmentation/
python train_with_data_parallel.py --model bisenet
                                   --backbone resnet18
                                   --batch-size 16
                                   --data-folder $INDIR
                                   --lr 0.025
                                   --optimizer radam
                                   --momentum 0.9
                                   --lr-scheduler poly
                                   --checkname cs512-radam
                                   --epochs 1000
                                   --save-dir $OUTDIR
                                   --dataset citys
                                   --multiple-GPUs False
                                   --save-interval 10
                                   --base-size 1024
                                   --crop-size 512
                                   --pretrained
                                   --em
cd ../../..
```

## Lookahead Optimizer

``` bash
cd master/experiments/segmentation/
python train_with_data_parallel.py --model bisenet
                                   --backbone resnet18
                                   --batch-size 16
                                   --data-folder $INDIR
                                   --lr 0.025
                                   --optimizer sgd
                                   --momentum 0.9
                                   --lr-scheduler poly
                                   --checkname cs512-lookAhead
                                   --epochs 1000
                                   --save-dir $OUTDIR
                                   --dataset citys
                                   --multiple-GPUs False
                                   --save-interval 10
                                   --base-size 1024
                                   --crop-size 512
                                   --pretrained
                                   --em
                                   --lookAhead_alpha 0.5
                                   --lookAhead_steps 5
cd ../../..
```

## Dice loss

```bash
cd master/experiments/segmentation/
python train_with_data_parallel.py --model bisenet
                                   --backbone resnet18
                                   --batch-size 16
                                   --data-folder $INDIR
                                   --lr 0.025
                                   --optimizer sgd
                                   --momentum 0.9
                                   --lr-scheduler poly
                                   --checkname bisenet-dice-gamma1-512-pretrained-fix-cross
                                   --epochs 1000
                                   --save-dir $OUTDIR
                                   --dataset citys
                                   --multiple-GPUs False
                                   --save-interval 10
                                   --base-size 1024
                                   --crop-size 512
                                   --soft-dice-loss
                                   --em
                                   --pretrained
cd ../../..
```

## Focal loss

``` bash
cd master/experiments/segmentation/
python train_with_data_parallel.py --model bisenet
                                   --backbone resnet18
                                   --batch-size 16
                                   --data-folder $INDIR
                                   --lr 0.025
                                   --optimizer sgd
                                   --momentum 0.9
                                   --lr-scheduler poly
                                   --checkname cs512pre-focalfixed
                                   --epochs 1000
                                   --save-dir $OUTDIR
                                   --dataset citys
                                   --multiple-GPUs False
                                   --save-interval 50
                                   --base-size 1024
                                   --crop-size 512
                                   --pretrained
                                   --no-cross
                                   --focal-loss
                                   --em
cd ../../..
```

## Label relaxation

``` bash
cd master/experiments/segmentation/
python train_with_data_parallel.py --model bisenet
                                   --backbone resnet18
                                   --batch-size 16
                                   --data-folder $INDIR
                                   --lr 0.025
                                   --optimizer sgd
                                   --momentum 0.9
                                   --lr-scheduler poly
                                   --checkname labelrelaxpretrained
                                   --epochs 1000
                                   --save-dir $OUTDIR
                                   --dataset citys
                                   --multiple-GPUs False
                                   --save-interval 10
                                   --base-size 1024
                                   --crop-size 512
                                   --em
                                   --label-relaxed-loss
                                   --no-cross
cd ../../..
```

## Shift-invariance

``` bash
cd shift-invariant/experiments/segmentation/
python train_with_data_parallel.py --model bisenet
                                   --backbone resnet18
                                   --batch-size 16
                                   --data-folder $INDIR
                                   --lr 0.025
                                   --optimizer sgd
                                   --momentum 0.9
                                   --lr-scheduler poly
                                   --checkname cs512-shift-invariant
                                   --epochs 1000
                                   --save-dir $OUTDIR
                                   --dataset citys
                                   --multiple-GPUs False
                                   --save-interval 10
                                   --base-size 1024
                                   --crop-size 512
                                   --pretrained
                                   --em
                                   --shift-invariant 5
cd ../../..
```

## Gather-Excite

``` bash
cd master/experiments/segmentation
python train_with_data_parallel.py --model bisenet
                                   --backbone resnet18_ge
                                   --batch-size 16
                                   --data-folder /input/datasets/cityscape_processed
                                   --lr 0.025
                                   --optimizer sgd
                                   --momentum 0.9
                                   --lr-scheduler poly
                                   --checkname ge_theta_minus_pretrain
                                   --epochs 1000
                                   --save-dir $OUTDIR
                                   --dataset citys
                                   --multiple-GPUs False
                                   --save-interval 10
                                   --base-size 1024
                                   --crop-size 512
                                   --em
cd ../../..
```

## CoordConv
``` bash
cd master/experiments/segmentation
python train_with_data_parallel.py --model bisenet_cc1
                                   --backbone cc1_resnet18
                                   --batch-size 16
                                   --data-folder /input/datasets/cityscape_processed
                                   --lr 0.025
                                   --optimizer sgd
                                   --momentum 0.9
                                   --lr-scheduler poly
                                   --checkname coord-conv
                                   --epochs 1000
                                   --save-dir $OUTDIR
                                   --dataset citys
                                   --multiple-GPUs False
                                   --save-interval 10
                                   --base-size 1024
                                   --crop-size 512
                                   --pretrained
                                   --em
cd ../../..
```


# How to test a segmentation network?

To test a network use the same settings as the training for backbone, and the model. Also make sure you have checked out to the correct branch if that was a requirement for training.

```  bash
cd master/experiments/segmentation
python test_multi_fusion.py --model bisenet
                            --dataset citys
                            --data-folder $INDIR
                            --backbone resnet18
                            --batch-size 1
                            --resume $OUTDIR/path/to/model.pth.tar
                            --base-size 1024
                            --crop-size 512
                            --fusion-mode mean
                            --eval
                            --scales 1.0
```
