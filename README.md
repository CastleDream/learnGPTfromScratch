- [learnGPTfromScratch](#learngptfromscratch)
  - [1. 文件结构说明](#1-文件结构说明)
  - [2. 学习进度记录](#2-学习进度记录)

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
│   ├── 1_micrograd_from_scratch.ipynb     # ✨
│   ├── 2_build_demo.ipynb                 # bigram用于中文语料的效果
│   ├── 2_build_makemore.ipynb             # ✨ 广播机制(broadcast)
│   ├── 3_MLP_demo.ipynb                   # 手动举例验证L1,L2正则差异;嵌入层/查表的前向和反向过程
│   ├── 3_MLP_makemore.ipynb               # ✨ 词嵌入, torch.view(pytorch内部机制), 交叉熵损失函数
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
    ├── 
    └── 
```

## 2. 学习进度记录

虽然说这个视频会讲一些科普性内容，但是并没有那么科普，最好还是提前知道一些，不然就是学了个寂寞。 还是适合NLP至少入了门的人看~

|任务|时间|前置知识|
|---|---|---|
|P1: 神经网络与反向传播详解：构建 micrograd| 2026.6.14~6.18| 前向计算，反向传播，python的操作符重载，pytorch |
|P2: 语言建模详解：构建 makemore|2026.6.23~6.25|Bag-of-words(词袋模型), n-gram, softmax, 最大似然函数|
|P3: 构建 makemore 第二部分：多层感知机 | 2026.6.26~6.29|词嵌入, torch.view(pytorch内部机制), 交叉熵损失函数; 手动举例验证L1,L2正则差异; 嵌入层/查表的前向和反向过程| 
|P4: 构建 makemore 第三部分：激活函数与梯度，批量归一化|2026.7.2~|
|P5: 构建 makemore 第四部分：成为反向传播高手|
|P6: 构建 makemore 第五部分：构建 WaveNet|
|P7: 从零开始，用代码详解构建 GPT|
|P8: GPT现状(BRK216HFS)|
|P9: 构建 GPT 分词器|
|P10: 复现 GPT-2 (124M 参数)|