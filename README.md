Cluttered Translated MNIST baseline models
=======================

Training code for two baseline convolutional neural network models for Cluttered Translated MNIST dataset. The models are reimplemented according to the description from [Dynamic Capacity Networks](http://arxiv.org/abs/1511.07838) paper.

Results for 100x100 Cluttered Translated MNIST with 8 distractors after 100 epochs of training:
* Fine model (6 conv layers): 0.92% test error
* Coarse model (3 conv layers): 1.69% test error

[DRAW model](http://arxiv.org/pdf/1502.04623.pdf) achieves 3.36% test error (with a much smaller model).

Requirements:
* GPU (due to cudnn dependency, which can be easily removed)
* torch
* optim
* cutorch
* nn
* cunn
* cudnn.torch

## How to use it?

A setup script will download MNIST and produce `mnist/*.t7` files:
```
luajit download_mnist.lua
```
Run the code!
```
th main.lua
```
Main options:
* `-GPU <zero-based ID>` sets the GPU to use
* `-coarseModel` switches to the coarse model (default is fine model)
* `-fixedTransformations` significantly reduces effective dataset size by using the same translation and clutter for every epoch. This causes overfitting.

## Results log

Fine model:
```
th main.lua
Epoch 100/100
Learning rate 0.001
train accuracy 99.416%, speed 561.18081375216 examples/s
validation accuracy 98.99%, speed 1342.6260823058 examples/s
test accuracy 99.08%, speed 1341.5407242925 examples/s
```

Fine model, fixed transformations:
```
th main.lua -fixedTransformations
Epoch 100/100
Learning rate 0.001
train accuracy 100%, speed 555.88065280935 examples/s
validation accuracy 98.49%, speed 1331.3891508727 examples/s
test accuracy 98.71%, speed 1306.1804514772 examples/s
```

Coarse model:
```
th main.lua -coarseModel
Epoch 100/100
Learning rate 0.001
train accuracy 98.428%, speed 2878.7456310446 examples/s
validation accuracy 98.33%, speed 4579.5679380543 examples/s
test accuracy 98.31%, speed 4590.1166433292 examples/s
```

Coarse model, fixed transformations:
```
th main.lua -fixedTransformations -coarseModel
Epoch 100/100
Learning rate 0.001
train accuracy 100%, speed 2722.8762624103 examples/s
validation accuracy 96.84%, speed 4407.5991842855 examples/s
test accuracy 96.63%, speed 4230.2471408819 examples/s
```
