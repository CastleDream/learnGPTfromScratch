- [learnGPTfromScratch](#learngptfromscratch)
  - [1. 文件结构说明](#1-文件结构说明)
  - [2. 学习进度记录](#2-学习进度记录)
  - [3. 涉及的论文](#3-涉及的论文)

# learnGPTfromScratch
Ref: [Github: karpathy/nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero)

## 1. 文件结构说明
每节课对应两个`ipynb`文件， 其中：
+ `xxx_demo.ipynb` 是我的练习文件
+ 另一个就是跟着老师课上的文件

```bash
.
├── code
│   ├── 1_micrograd_demo.ipynb             # Min-Max/hinge loss, L2正则
│   ├── 1_micrograd_from_scratch.ipynb     # ✨ 手写最简单的反向梯度和模型训练框架
│   ├── 2_build_demo.ipynb                 # bigram用于中文语料的效果
│   ├── 2_build_makemore.ipynb             # ✨ 广播机制(broadcast)
│   ├── 3_MLP_demo.ipynb                   # 手动举例验证L1,L2正则差异;嵌入层/查表的前向和反向过程
│   ├── 3_MLP_makemore.ipynb               # ✨ 词嵌入, torch.view(pytorch内部机制), 交叉熵损失函数
│   ├── 4_MLP2_demo.ipynb                  # 把4_MLP2_makemore中的内容变得更pytorch化并训练一个更深的神经网络(反向传播扩展和 3Blue1Brown视频)
│   ├── 4_MLP2_makemore.ipynb              # ✨ 随机初始化和激活函数带来的dead neuron问题; (随机初始化的问题)两个正态分布相乘结果的期望和方差计算;
│   ├──
│   ├── 5_backward_makemore.ipynb
│   ├── 
│   ├── 
│   ├── ....
│   ├── a_utils.py                         # 用于 1_micrograd_from_scratch.ipynb 的辅助函数
│   ├── micrograd
│   ├── names.txt                          # makemore用到的英文名字数据集
│   ├── names_zh.txt                       # 自己搜集的中文名字数据
│   ├── names_zh_list.txt                  # 处理后得到的中文名字数据集
│   └── test.py                            # 测试 micrograd 中的代码
└── md
    ├── 1_神经网络与反向传播详解_micrograd.md
    ├── 2_语言建模详解_makemore.md
    ├── 3_多层感知机_makemore.md
    ├── 4_激活函数与梯度~批量归一化_makemore.md
    ├── 5_深入反向传播.md
    └── a_难点查漏补缺.md                  # (●'◡'●)以前理解错误/有偏差的内容，彻底纠正
```

## 2. 学习进度记录

虽然说这个视频会讲一些科普性内容，但是并没有那么科普，最好还是提前知道一些，不然就是学了个寂寞。 还是适合NLP至少入了门的人看~

每节课`---------------`上面是课程主要内容，下面是自己练习的一些补充内容



|任务|时间|前置/补充知识|
|---|---|---|
|P1: 神经网络与反向传播详解：构建 micrograd| 2026.6.14~6.18| - 前向计算<br/>- 反向传播<br/>- python的操作符重载<br/>- pytorch<br/>---------------<br/>- Min-Max/hinge loss, L2正则|
|P2: 语言建模详解：构建 makemore|2026.6.23~6.25|- Bag-of-words(词袋模型)<br/>- n-gram<br/>- softmax<br/>- 最大似然函数<br/>- 广播机制(broadcast)|
|P3: 构建 makemore 第二部分：多层感知机 | 2026.6.26~6.29|- 词嵌入<br/>- torch.view(pytorch内部机制)<br/>- 交叉熵损失函数(数值稳定性-max)<br/>---------------<br/>- 手动举例验证L1,L2正则差异<br/>- 嵌入层/查表的前向和反向过程| 
|P4: 构建 makemore 第三部分：激活函数与梯度，批量归一化|2026.7.2~7.16|- 激活函数带来的dead neuron问题<br/>- 两个正态分布相乘结果的期望和方差计算<br/>- 归一化:<br/>1. **输入端**（数据预处理）：Standardization / Normalization。处理的是原始数据（如图像像素、表格特征）, 常用的归一化包括：Min-Max 归一化、Z-score 标准化<br/>2. **隐藏层**（网络内部归一化）：BN、LN、IN、GN、RMSNorm 等。处理的是网络中间层的激活值（Activations / Hidden states）。<br/>3. **权重端**（参数归一化）：Weight Normalization。对神经网络的权重矩阵进行归一化，常用于 RNN 或某些生成模型。<br/>4. **输出端**（概率归一化）：Softmax、Sigmoid。将输出 logits 转化为概率分布（和为1或在0-1之间），本质上也是一种归一化。<br/>- 多层感知机(输入层，单个隐藏层，输出层)<br/>- BN详解<br/>- BN作用的卷积层/线性层的偏置梯度为0的代码, 前向/反向过程中偏置无效的原理推导<br/>- resnet里BN的实际使用和pytorch中BN层的参数说明<br/>---------------<br/>- 反向传播扩展和 3Blue1Brown视频<br/>- tanh的增益为什么是 5/3<br/>- 网络不同层的`权重更新量`:`权重值`(绘图判断网络训练效率/学习率设置科学性),科学诊断网络初始化参数设置的合理性|
|P5: 构建 makemore 第四部分：成为反向传播高手| 2026.7.16~|- 多分支节点的梯度=多分支的梯度的和(多元复合函数的全微分（链式法则）)<br/>- **pytorch张量中每个元素都有一个梯度值以及学习率缩放不影响梯度方向的正确理解**<br/>- 数学中，描述函数时提到的“一维、二维、n维”指的是什么<br/>- 矩阵乘法梯度计算<br/>- BN训练时用的是有偏估计，推理时用的是无偏估计(bessel correction 贝塞尔纠正)<br/> - 估计量的有偏/无偏定义<br/>- pytorch var 和 Batchnorm1d对var计算的规定 |
|P6: 构建 makemore 第五部分：构建 WaveNet|
|P7: 从零开始，用代码详解构建 GPT|
|P8: GPT现状(BRK216HFS)|
|P9: 构建 GPT 分词器|
|P10: 复现 GPT-2 (124M 参数)|


## 3. 涉及的论文
+ MLP, following Bengio et al. 2003 [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
+ CNN, following DeepMind WaveNet 2016 (in progress...) [WAVENET: A GENERATIVE MODEL FOR RAW AUDIO](https://arxiv.org/pdf/1609.03499)
+ RNN, following Mikolov et al. 2010 [Recurrent neural network based language model](https://www.fit.vut.cz/research/group/speech/public/publi/2010/mikolov_interspeech2010_IS100722.pdf)
+ LSTM, following Graves et al. 2014 [Generating Sequences With Recurrent Neural Networks](https://arxiv.org/pdf/1308.0850)
+ GRU, following Kyunghyun Cho et al. 2014 [On the Properties of Neural Machine Translation: Encoder–Decoder Approaches](https://arxiv.org/pdf/1409.1259)
+ Transformer, following Vaswani et al. 2017 [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)

----

+ kaiming init: [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/pdf/1502.01852)
+ batch normalization: [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167)