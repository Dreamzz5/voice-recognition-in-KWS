# A Simple Net

#### 介绍
使用简单网络 实现语言识别

#### 数据来源
Google speech Command
#### 数据描述

[数据](http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz)为长度为1s 的语言信号:
```
label={"silence","unknown","yes","no","up","down","left","right","on","off","stop","go"}；
```
#### 参数设置与防止过拟合方法

使用[Adam](https://zhuanlan.zhihu.com/p/32626442)方法并结合<a href="https://www.codecogs.com/eqnedit.php?latex=$l_2$" target="_blank"><img src="https://latex.codecogs.com/png.latex?$l_2$" title="$l_2$" /></a>正则

![网络规模](https://images.gitee.com/uploads/images/2019/0806/171750_d378e551_4938113.png "Conv.png")

使用`Dropout`防止过拟合

<div align=center><img width="300" height="300" src="https://images.gitee.com/uploads/images/2019/0806/172058_c486eecb_4938113.png"/><img width="300" height="300" src="https://images.gitee.com/uploads/images/2019/0806/172044_0a5ccd3d_4938113.png"/></div>


#### 网络整体架构
<div align=center><img  src="https://images.gitee.com/uploads/images/2019/0806/174431_b8820d56_4938113.png"/></div>

#### 卷积加速
[MATLAB_MTIMESX](https://www.mathworks.com/matlabcentral/fileexchange/25977-mtimesx-fast-matrix-multiply-with-multi-dimensional-support) 按照[教程](https://mattwang44.github.io/en/articles/MATLAB_MTIMESX/?nsukey=gAiwvE82pomcaI894d1gIDT8S2Liz5XteIoWYwp76332xCgZYOWWcJz%2FGQQ1L6Vc2k87mGbn7htoxewlzugvNT8Lp06lO0AbVOsCc%2Fm%2B2Q3zXIsQmwakcRpxCGi1%2F3jm%2FJhHsoOZ01EZMLtSVR3a%2B5v2SJA87fC%2BSBTBXZiOYPl07kvE5NZW%2BnBGSlglN9LusE8J3jpphWS4drbotXig7w%3D%3D) 将C代码进行编译。
#### 数据集

由于数据大小问题，难以上传，需要请联系我`Dream_0319@163.com`



