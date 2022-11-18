# CSPNet：2020

**A New Backbone that can Enhance Learning Capability of CNN**

#### 论文出发点和背景

通过重新设计架构，缓解网络需要大量推理计算的问题（归咎于网络优化中的重复梯度信息）

通过设计CSPNet能够在减少计算量的同时实现更为丰富的梯度组合。 

提出CSPNet主要为了处理以下问题：

1.加强CNN的学习能力 

2.消除瓶颈过高的计算瓶颈

3.降低内存成本

#### 论文创新思路

通过设计CSPNet能够在减少计算量的同时实现更为丰富的梯度组合。通过将底层的特征映射划分为两部分，然后通过crossstage hierarchy进行特征融合。

无论是残差网络还是密集网络，跳连会导致将梯度传递到前向的层，进而导致重复学习冗余信息

#### 论文方法介绍



![image-20221117202222010](C:\Users\李鑫\AppData\Roaming\Typora\typora-user-images\image-20221117202222010.png)

传统的卷积神经网络可以被描述为：

![image-20221117202547569](C:\Users\李鑫\AppData\Roaming\Typora\typora-user-images\image-20221117202547569.png)

ResNet和DenseNet分别可以被描述为：

![image-20221117202634207](C:\Users\李鑫\AppData\Roaming\Typora\typora-user-images\image-20221117202634207.png)

$x_{0}$被分为了两部分 x0'和x0'',T是transition function用来截断梯度流的函数,M是对两个部分进行融合的函数

图中的Transition Layer代表过渡层，主要包含瓶颈层（1x1卷积）和池化层（可选）

![image-20221117202921807](C:\Users\李鑫\AppData\Roaming\Typora\typora-user-images\image-20221117202921807.png)

![image-20221117203712033](C:\Users\李鑫\AppData\Roaming\Typora\typora-user-images\image-20221117203712033.png)

![image-20221117203801622](C:\Users\李鑫\AppData\Roaming\Typora\typora-user-images\image-20221117203801622.png)

目标检测检测头：

1.二级检测头优于一级检测头，提出了EFM

2.聚合特征金字塔

3.平衡计算消耗，结合Maxout操作

![image-20221117204911172](C:\Users\李鑫\AppData\Roaming\Typora\typora-user-images\image-20221117204911172.png)

#### 实际效果

将CSP应用到ResNet 、ResNext和DenseNet上后，计算量可以从10%减少到20%，而且在准确率方面还优于上述网络。

![image-20221117201135293](C:\Users\李鑫\AppData\Roaming\Typora\typora-user-images\image-20221117201135293.png)

![image-20221117204939610](C:\Users\李鑫\AppData\Roaming\Typora\typora-user-images\image-20221117204939610.png)

![image-20221117204949418](C:\Users\李鑫\AppData\Roaming\Typora\typora-user-images\image-20221117204949418.png)

![image-20221117205010023](C:\Users\李鑫\AppData\Roaming\Typora\typora-user-images\image-20221117205010023.png)

#### 个人理解

第一次见DenseNet的时候就有一种特征过分重用的感觉，记得在GhostNet里面有张图说是可视化每个通道上的feature，结果发现很多的特征图都是十分相似的。如果像DenseNet那样全部cat过去就会有很多的冗余信息。CSPNet就是通过设计模块，将输入的信息分为两部分，(以改进的DenseNet为例)通过对Dense支路进行先transition后concatenate,阻断一部分梯度信息。CSP不仅吸纳了DenseNet特征重用的优点，同时也没有对重复的梯度信息进行利用。



