# DenseNet: 2018

**Densely Connected Convolutional Networks**

code:https://github.com/liuzhuang13/DenseNet

#### 论文出发点或背景

最近的研究表明，将靠近输入和靠近输出的层通过跳连方式连接起来的网络可以变得更深、准确率更高、训练更高效。

当网络变深的时候，会发生梯度弥散的现象。最近的ResNet和Highway Net是通过跳连的方式解决的。随即深度是通过在训练过程中随机删除层来缩短ResNet以此允许更好的信息和梯度传递。FractalNet通过重复组合多个具有不同卷积块数量的并行层，进而获得更大的深度。

尽管他们的网络拓扑和训练方法上都有所不同，但是他们都通过跳连实现了从前层到后层的连接。

ResNet的优点是梯度可以通过恒等映射的支路直接进入下层，但是恒等映射与输出相加的形式可能会阻碍网络中的信息流。

#### 论文创新思路

为了确保网络层间有最大的信息流动，我们直接将所有层的特征图拼接起来。为了保证前馈特性，每一层都从所有之前的层获得额外的输入，并将自己的特征映射传递到后续所有的层。相比于ResNet，我们没有通过使用加和来组合特征，而是通过在通道上拼接。

为了改善层之间的信息流传递，我们提出了一种新的连接方式![image-20221117190110863](C:\Users\李鑫\AppData\Roaming\Typora\typora-user-images\image-20221117190110863.png)

#### 论文方法大概介绍

普通的卷积L层神经网络只有 L个连接。而我们的DenseNet有$L(L-1)/2$条连接



![image-20221117183118613](C:\Users\李鑫\AppData\Roaming\Typora\typora-user-images\image-20221117183118613.png)

复合函数：批归一化、ReLU和3×3卷积

过渡层：批归一化、1×1卷积层、2×2平均池化

瓶颈层：BN-ReLU-Conv（1×1）-BN-ReLU-Conv（3×3）

#### 实际效果

观察到Dense Connection具有正则化效应

比ResNet的参数更少

DenseNets的表现与最先进的ResNets相当，同时需要明显更少的参数和计算来实现类似的性能。

DenseNet的网络结构更为紧凑，加强了特征重用

![image-20221117185351577](C:\Users\李鑫\AppData\Roaming\Typora\typora-user-images\image-20221117185351577.png)

![image-20221117190938036](C:\Users\李鑫\AppData\Roaming\Typora\typora-user-images\image-20221117190938036.png)

![image-20221117191226496](C:\Users\李鑫\AppData\Roaming\Typora\typora-user-images\image-20221117191226496.png)

![image-20221117191308536](C:\Users\李鑫\AppData\Roaming\Typora\typora-user-images\image-20221117191308536.png)

#### 个人理解

1.相比于ResNet，DenseNet将L层之前的L-1层的特征都得到了利用，在特征重用方面做的比ResNet要好一点，而且相较于直接加和这种信息传递，直接将之前的特征和该层特征拼接起来，信息的完整度更高

2.不足的一点是：由于需要cat操作，数据需要被复制，显存占用很容易增加