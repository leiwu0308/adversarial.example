## pytorch implemment of  adversarial examples generating
ToDo List
 
 - [] implement iterative gradient sign method
 - [] test MNIST/CIFAR10, LeNet, ResNet, FNN, Linear Classifier 
 - [] survey the cross-model transferability within linear classifier


### Result for MNIST
Adversrial robustness of LeNet and FNN(784-500-500-300-50-10) for MNIST, where neither batch normalization nor dropout is  used. 
the stepsize is set to 5/255, and number of iteration = 10

|perturbation|0     |5   |10  |15  | 20 | 25 | 30 |35  |40  |   
|:--:        |:--:  |:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|LeNet    | 99.27|98.35|96.77|93.56|87.77|77.20|61.85|44.91|32.09|
|FNN      |98.78 |97.06|93.30|85.27|68.55|48.62|33.48|23.65|17.80|
|Linear   |92.66 |81.46|56.00|25.44|10.03|2.93 |0.83 |0.31 |0.17 |

### Result for CIFAR10
only one-step is used in this experiment.

|perturbation| 0 | 1| 2 | 3 | 4 | 5 |
|:--------:|:----:|:--:|:--:|:--:|:--:|:--:|
|ResNet-8|87.25|33.41|11.78|6.33|4.84|4.06| 
