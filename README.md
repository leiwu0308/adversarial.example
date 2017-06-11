## pytorch implemment of  adversarial examples generating
ToDo List
 
 -  implement iterative gradient sign method
 -  test MNIST/CIFAR10, LeNet, ResNet, FNN, Linear Classifier 
 -  survey the cross-model transferability within linear classifier


### Result for MNIST
Adversrial robustness for MNIST, where neither batch normalization nor dropout is  used. Fast gradient sign (FGS) adversarial perturbations
are used to assess the adversarial robustness.

|perturbation           |0     |5   |10  |15  | 20 | 25 | 30 |35  |40  |   
|:--:                   |:--:  |:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|LeNet                  | 99.27|98.35|96.77|93.56|87.77|77.20|61.85|44.91|32.09|
|Linear                 |92.66 |81.46|56.00|25.44|10.03|2.93 |0.83 |0.31 |0.17 |
|FNN                    |98.78 |97.06|93.30|85.27|68.55|48.62|33.48|23.65|17.80|
|784-500-10             |98.42 |96.12|90.64|79.00|58.65|36.68|22.43|14.78|11.21|
|784-500-300-10         |98.55 |96.53|91.48|81.47|65.35|49.37|37.95|30.49|24.99|
|784-500-300-200x1-10   |98.63 |96.75|92.71|85.18|74.84|63.48|53.71|44.91|38.02|
|784-500-300-200x2-10   |98.61 |96.88|93.11|86.02|76.10|66.20|57.51|50.85|45.31|
|784-500-300-200x4-10   |98.50 |96.02|91.70|84.17|74.50|67.99|65.54|64.29|63.37|
|784-500-300-200x8-10   |98.32 |95.72|90.31|81.92|74.55|70.01|67.10|65.54|64.26|

### Result for CIFAR10
only one-step is used in this experiment.

|perturbation| 0 | 1| 2 | 3 | 4 | 5 |
|:--------:|:----:|:--:|:--:|:--:|:--:|:--:|
|ResNet-8|87.25|33.41|11.78|6.33|4.84|4.06| 
|ResNet-14|90.42|38.71|24.38|19.74|17.10|15.59|
|ResNet-20|91.40|43.97|28.88|23.24|20.39|18.52|
|ResNet-26|92.08|46.30|30.58|23.86|20.85|18.93|
|ResNet-32|92.08|46.19|30.93|25.09|22.08|20.31|
|ResNet-38|92.09|48.35|31.37|24.36|20.76|18.95|
|ResNet-44|92.41|49.16|33.60|27.51|24.14|22.15|
|ResNet-56|91.14|52.88|33.93|24.73|19.72|16.78|
|ResNet-110|91.31|61.78|41.15|30.01|24.13|20.92|
