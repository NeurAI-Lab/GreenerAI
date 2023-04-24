mkdir data
cd data
wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
tar -xvf cifar-100-python.tar.gz
rm  cifar-100-python.tar.gz
cd ..
cp -avr data master
cp -avr data classifier-experiments
cp -avr data Gather-Excite
cp -avr data mixup
cp -avr data Shift-invariance
rm -r data