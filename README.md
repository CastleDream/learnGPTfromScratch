- [learnGPTfromScratch](#learngptfromscratch)
  - [1. 文件结构说明](#1-文件结构说明)
  - [2. 学习进度记录](#2-学习进度记录)
  - [3. 涉及的论文](#3-涉及的论文)
  - [4. 前沿研究](#4-前沿研究)
    - [1. 机械可解释性（Mechanistic Interpretability, 简称 MI）](#1-机械可解释性mechanistic-interpretability-简称-mi)
    - [2. 知识编辑（Model Editing）](#2-知识编辑model-editing)

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
│   ├── 1_micrograd_from_scratch.ipynb     # ✨micrograd:  手写最简单的反向梯度和模型训练框架
│   ├── 2_build_demo.ipynb                 # bigram用于中文语料的效果
│   ├── 2_build_makemore.ipynb             # ✨makemore_Part1:  广播机制(broadcast)
│   ├── 3_MLP_demo.ipynb                   # 手动举例验证L1,L2正则差异;嵌入层/查表的前向和反向过程
│   ├── 3_MLP_makemore.ipynb               # ✨makemore_Part2:   词嵌入, torch.view(pytorch内部机制), 交叉熵损失函数
│   ├── 4_MLP2_demo.ipynb                  # 把4_MLP2_makemore中的内容变得更pytorch化并训练一个更深的神经网络(反向传播扩展和 3Blue1Brown视频)
│   ├── 4_MLP2_makemore.ipynb              # ✨makemore_Part3:   随机初始化和激活函数带来的dead neuron问题; (随机初始化的问题)两个正态分布相乘结果的期望和方差计算;
│   ├── 5_backward_demo.ipynb              # 全导数(数学方式)求导 vs 计算图(程序可模块化的分步)求导
│   ├── 5_backward_makemore.ipynb          # ✨makemore_Part4:   手动计算交叉熵损失和BN的反向梯度，替代loss.backward()
│   ├── 6_WaveNet_makemore.ipynb           # ✨makemore_Part5:   更pytorch化的网络表达，多维数组dim的意义  
│   ├── zero_gpt                           # ✨nanoGPT
│   │   ├── 7_start_GPT.ipynb              # 从0开始写GPT的草稿，校验过的代码在train.py里
│   │   ├── train.py 
│   │   ├── 
│   │   └──                                # 
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
    ├── ...   
    └── a_难点查漏补缺.md                  # (●'◡'●)以前理解错误/有偏差的内容，彻底纠正
```

## 2. 学习进度记录

虽然说这个视频会讲一些科普性内容，但是并没有那么科普，最好还是提前知道一些，不然就是学了个寂寞。 还是适合NLP至少入了门的人看~

每节课`---------------`上面是课程主要内容，下面是自己练习的一些补充内容



|任务|时间|前置/补充知识|
|---|---|---|
|P1: 神经网络与反向传播详解：构建 micrograd| 2026.6.14~6.18| - 前向计算<br/>- 反向传播<br/>- python的操作符重载<br/>- pytorch<br/>---------------<br/>- Min-Max/hinge loss, L2正则|
|P2: 语言建模详解：构建 makemore(part 1)|2026.6.23~6.25|- Bag-of-words(词袋模型)<br/>- n-gram<br/>- softmax<br/>- 最大似然函数<br/>- 广播机制(broadcast)|
|P3: 构建 makemore 第二部分：多层感知机(part 2) | 2026.6.26~6.29|- 词嵌入<br/>- torch.view(pytorch内部机制)<br/>- 交叉熵损失函数(数值稳定性-max)<br/>---------------<br/>- 手动举例验证L1,L2正则差异<br/>- 嵌入层/查表的前向和反向过程(可以顺带看看 `5_backward_makemore.ipynb -> dC计算过程`)| 
|P4: 构建 makemore 第三部分：激活函数与梯度，批量归一化(part 3)|2026.7.2~7.16|- 激活函数带来的dead neuron问题<br/>- 两个正态分布相乘结果的期望和方差计算<br/>- 归一化:<br/>1. **输入端**（数据预处理）：Standardization / Normalization。处理的是原始数据（如图像像素、表格特征）, 常用的归一化包括：Min-Max 归一化、Z-score 标准化<br/>2. **隐藏层**（网络内部归一化）：BN、LN、IN、GN、RMSNorm 等。处理的是网络中间层的激活值（Activations / Hidden states）。<br/>3. **权重端**（参数归一化）：Weight Normalization。对神经网络的权重矩阵进行归一化，常用于 RNN 或某些生成模型。<br/>4. **输出端**（概率归一化）：Softmax、Sigmoid。将输出 logits 转化为概率分布（和为1或在0-1之间），本质上也是一种归一化。<br/>- 多层感知机(输入层，单个隐藏层，输出层)<br/>- BN详解<br/>- BN作用的卷积层/线性层的偏置梯度为0的代码, 前向/反向过程中偏置无效的原理推导<br/>- resnet里BN的实际使用和pytorch中BN层的参数说明<br/>---------------<br/>- 反向传播扩展和 3Blue1Brown视频<br/>- tanh的增益为什么是 5/3<br/>- 网络不同层的`权重更新量`:`权重值`(绘图判断网络训练效率/学习率设置科学性),科学诊断网络初始化参数设置的合理性|
|P5: 构建 makemore 第四部分：成为反向传播高手| 2026.7.16~7.30|- 多分支节点的梯度=多分支的梯度的和(多元复合函数的全微分（链式法则）)<br/>- **pytorch张量中每个元素都有一个梯度值以及学习率缩放不影响梯度方向的正确理解**<br/>- 数学中，描述函数时提到的“一维、二维、n维”指的是什么<br/>- 矩阵乘法梯度计算<br/>- BN训练时用的是有偏估计，推理时用的是无偏估计(bessel correction 贝塞尔纠正)<br/> - 估计量的有偏/无偏定义<br/>- pytorch var 和 Batchnorm1d对var计算的规定 <br/>- **交叉熵损失求导过程推导**<br/>- 交叉熵损失的梯度和为0的性质<br/>- logit_maxes的值不影响最终结果(梯度为0)和 **Log-Sum-Exp**, 应用`Log-Sum-Exp-trick`来进行交叉熵损失的前向计算和反向传播<br/>- **计算图求导的过程其实就是在计算全导数**<br/>---------------<br/>- BN反向传播的求导过程 **基于全导数求导** 和 **计算图求导** 两种方案对比, 全导数(数学方式)求导 vs 计算图(程序可模块化的分步)求导 <br/> -[CS231n Spring 2019 Assignment 2—Batch Normalization](https://blog.csdn.net/laizi_laizi/article/details/102175105) |
|P6: 构建 makemore 第五部分：构建 WaveNet|2026.7.31~8.6|- pytorch Sequential容器类<br/>- BN层修改训练/推理模式（单个样本计算方差报错）<br/>- Image Pyramids(图像金字塔)<br/>- dilated causal convolutional layers(扩张因果卷积(空洞卷积 + 因果约束))<br/>- `多维数组维度的意义`(（4,4,10）只有10包括的维度有数据，其余其实都只是截断/分组的标识); 以及 `BN面对非二维输入`|
|P7: 从零开始，用代码构建 GPT|2026.8.6~ 9.1|- nn.Embedding层随机初始化<br/>- optimizer.zero_grad(set_to_none = True),优化器设置梯度为0或者None的区别<br/>- 自注意力机制中的数学技巧(不同时间步求均值改为矩阵乘法, 为什么B批次维度会放在第一位, 多维矩阵计算)<br/>- **自注意力机制key,query的合理性(符合语言模型的本质)**<br/>- **自注意力机制value的合理性**(有向图中节点间的信息传递机制)<br/>- 自注意力机制和位置嵌入<br/>- 缩放注意力机制的合理性, **kq乘法方差计算目的**<br/>- pytorch的`forward函数`和python的`__call__()`的区别<br/> - (md文档中)register_buffer() vs register_parameter<br/>- (md文档中)**相同输出维度，多头效果优于单头**<br/>- (md文档中) transformer的残差连接 <br/>- **LayerNorm实现，BatchNorm和LayerNorm就差个dim**|
|P8: GPT现状(BRK216HFS)| 2026.9.2|大会发言，记录见[md/8_GPT现状BRK216HFS.md](./md/8_GPT现状BRK216HFS.md)
|P9: 构建 GPT 分词器|
|P10: 复现 GPT-2 (124M 参数)|


## 3. 涉及的论文
+ MLP, following Bengio et al. 2003 [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
+ CNN, following DeepMind WaveNet 2016 (in progress...) [WAVENET: A GENERATIVE MODEL FOR RAW AUDIO](https://arxiv.org/pdf/1609.03499)
+ RNN, following Mikolov et al. 2010 [Recurrent neural network based language model](https://www.fit.vut.cz/research/group/speech/public/publi/2010/mikolov_interspeech2010_IS100722.pdf)
+ LSTM, following Graves et al. 2014 [Generating Sequences With Recurrent Neural Networks](https://arxiv.org/pdf/1308.0850)
+ GRU, following Kyunghyun Cho et al. 2014 [On the Properties of Neural Machine Translation: Encoder–Decoder Approaches](https://arxiv.org/pdf/1409.1259)
+ Transformer, following Vaswani et al. 2017 [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
+ Resnet, 2015年 [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385)
  + skip connection, 2016年 [Identity Mappings in Deep Residual Networks](https://arxiv.org/pdf/1603.05027)
  + [Residual blocks — Building blocks of ResNet](https://medium.com/data-science/residual-blocks-building-blocks-of-resnet-fd90ca15d6ec)
  + [Accurate, Large Minibatch SGD:Training ImageNet in 1 Hour](https://arxiv.org/pdf/1706.02677)
+ [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf?utm_content=buffer79b4)
+ 预训练 [Language Models are Few-Shot Learners](https://arxiv.org/pdf/2005.14165)
+ 微调 [ChatGPT: Optimizing Language Models for Dialogue](https://openai.com/zh-Hans-CN/index/chatgpt/)
+ GPT2: [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf.), 不用科学上网
+ LLaMA: [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/pdf/2302.13971)

----

+ kaiming init: [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/pdf/1502.01852)
+ batch normalization: [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167)

---

## 4. 前沿研究
### 1. 机械可解释性（Mechanistic Interpretability, 简称 MI）
直接看网站： [Anthropic’s Interpretability Research](https://transformer-circuits.pub/)
+ [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)
  + 网页版本，没有pdf版本，直接页面翻译阅读吧
+ [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)
  + 网页版本， 虽然有arxiv上的pdf版本，但是很明显，是直接html直接保存成pdf的
+ [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html)
  + 同上
+ [Towards Monosemanticity: Decomposing Language Models With Dictionary Learning](https://transformer-circuits.pub/2023/monosemantic-features)
  + 同上
+ [anthropic-Mapping the Mind of a Large Language Model ](https://www.anthropic.com/research/mapping-mind-language-model)
  + 对应的链接：[transformer-circuits.pub: Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)
  + 搜索过程中还顺带看到这个：[Mapping the Minds of LLMs: A Graph-Based Analysis of Reasoning LLM](https://arxiv.org/pdf/2505.13890)
+ [OpenAI-Multimodal neurons in artificial neural networks](https://openai.com/index/multimodal-neurons/)
  + [distill-Multimodal Neurons in Artificial Neural Networks](https://distill.pub/2021/multimodal-neurons/), 我最喜欢的已经停运的`distill`网站，
+ [A Mechanistic Interpretability Analysis of Grokking](https://www.alignmentforum.org/posts/N6WM6hs7RQMKDhYjB/a-mechanistic-interpretability-analysis-of-grokking)
  + 这个有论文，arxiv上的，[PROGRESS MEASURES FOR GROKKING VIA MECHANISTIC INTERPRETABILITY](https://arxiv.org/pdf/2301.05217)

### 2. 知识编辑（Model Editing）
+ [Locating and Editing Factual Associations in GPT](https://rome.baulab.info/)
  + ROME (Rank-One Model Editing) 技术
  + 这也是个网页版的，下面很多引用也都是代表作
  + 这个有论文，arxiv上的, [Locating and Editing Factual Associations in GPT](https://arxiv.org/pdf/2202.05262)
+ [Mass Editing Memory in a Transformer](https://memit.baulab.info/)
  + MEMIT技术，[MASS-EDITING MEMORY IN A TRANSFORMER](https://arxiv.org/pdf/2210.07229)


---

+ TransformerLens：由著名独立研究员 Neel Nanda 开发的开源 Python 库。它是目前做机械可解释性最权威、最常用的工具，专门用于提取、缓存和分析 Transformer 每一层的激活值和注意力模式。
+ Neuronpedia：一个可视化的在线数据库，展示了通过 SAE 提取出的数以万计的 LLM 内部“特征”，你可以直观地搜索“哪些神经元对‘猫’敏感”。
+ BertViz：由 Jesse Vig 开发的注意力机制可视化工具，能极其直观地展示注意力头在句子词元（Token）之间的连线关系。
+ Distill.pub：虽然已停止更新，但 Chris Olah 团队早年在这里发表的关于 CNN 和 Transformer 视觉化解析的文章（如 Feature Visualization），至今仍是该领域的“圣经”，排版和交互极其精美。