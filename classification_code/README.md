cd classification_code
sh download_cifar.sh

# LookAhead experiment 

``` bash
cd classifier-experiments
python train.py -b=128 -lr .1 -checkname=LookAhead -output_dir=output/classifier/ -lookAhead_steps 5 -net=resnet50
cd ..
```

# BlurPool experiment 

``` bash
cd Shift-invariance
python train.py -b=128 -lr .1 -checkname=BlurPool -output_dir=output/classifier/ -=-shiftInvariant=5 -net=resnet50
cd ..
```
# Cyclic LR experiment

``` bash
cd classifier-experiments
python train.py -b=128 -lr .1 -checkname=CyclicLR -output_dir=output/classifier/ -lr_scheduler “clr” -lr .00001 -net=resnet50
cd ..
```

# Focal Loss experiment

``` bash 
cd classifier experiments
python train.py -b=128 -lr .1 -checkname=FocalLoss -output_dir=output/classifier/ -loss focal -net=resnet50
cd ..
```
# Mixup experiment  

``` bash 
cd mixup
python train.py -b=128 -lr .1 -checkname=Mixup -output_dir=output/classifier/ -mixup_alpha 1.0 -net=resnet50
cd ..
```
# Poly LR 1/2 Epochs experiment

``` bash
cd mixup
python train.py -b=128 -lr .1 -checkname=HalfPoly -output_dir=output/classifier/ -scheduler half-poly -net=resnet50
cd ..
```

# GE-θ experiment

``` bash 
cd Gather-Excite
python train.py -b=128 -lr .1 -checkname=GatherExcite -output_dir=output/classifier/ -net=resnet50_ge -ge_trained True
cd ..
```
#  GE-θ-Minus experiment

``` bash
cd Gather-Excite
python train.py -b=128 -lr .1 -checkname=GatherExciteMinus -output_dir=output/classifier/ -net=resnet50_ge -ge_trained False
cd ..
```
​
#  Random Gradient	
​
``` bash
cd master
python train.py -b=128 -lr .1 -checkname=RandomGradient -random_gradient -output_dir=output/classifier/ -net=resnet50 
cd ..
```
​
#  Spatial Bottlenecks	
​
``` bash
cd master
python train.py -b=128 -lr .1 -checkname=SpatialBottlenecks -output_dir=output/classifier/ -net=sp_resnet50
cd ..
```
​
#  RAdam
​
``` bash
cd master
python train.py -b=128 -lr .1 -checkname=RAdam -optimizer=radam -output_dir=output/classifier/ -net=resnet50
cd ..
```
​
#  CoordConv
​
``` bash
cd master
python train.py -b=128 -lr .1 -checkname=CoordConv -output_dir=output/classifier/ -net=coord_resnet50 
cd ..
```
​
#  Switchable normalization	
​
``` bash
cd master
python train.py -b=128 -lr .1 -checkname=SwitchableNormalisation -output_dir=output/classifier/ -net=switch_resnet50 
cd ..
```
