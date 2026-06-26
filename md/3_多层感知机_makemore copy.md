- [P3: 构建 makemore 第二部分：多层感知机](#p3-构建-makemore-第二部分多层感知机)
  - [链接](#链接)
  - [关键内容](#关键内容)

# P3: 构建 makemore 第二部分：多层感知机
## 链接
B站视频链接：
+ [Andrej Karpathy【中英⚡从零构建 GPT（重制版）|Neural Networks: Zero to Hero】](https://www.bilibili.com/video/BV1mqrTBvEaf/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=1019ffdc843339404e9df6ae52ff9e77)

Github项目： <https://github.com/karpathy/makemore>

## 关键内容
1. 上节课的内容里有Bigram进行字符预测，这里主要是输入单个字符，预测下一个字符，这种方案存在的问题：
   1. 模型效果一般
   2. 难以扩展，
      1. 当输入单个字符，预测下个字符时，模型的参数（P矩阵的大小）是 `27x27`; 
      2. 当输入的字符变多，预测下个字符时，模型的参数随输入字符的数量呈现指数级增长
      3. 例如： 输入2个字符，预测下个字符，则输入的组合就有 `27*27=729`, 则预测下一个字符时，整个模型的参数就来到了$27^3 = 19683$
      4. 即：当输入从1变成2时，表格的大小从 `729`激增至`19683`， 从七百到接近两万，翻了27（约等于30）倍

**MLP**, 
+ following `Bengio et al. 2003` [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)



+ CNN, following DeepMind WaveNet 2016 (in progress...) [WAVENET: A GENERATIVE MODEL FOR RAW AUDIO](https://arxiv.org/pdf/1609.03499)
+ RNN, following Mikolov et al. 2010 [Recurrent neural network based language model](https://www.fit.vut.cz/research/group/speech/public/publi/2010/mikolov_interspeech2010_IS100722.pdf)
+ LSTM, following Graves et al. 2014 [Generating Sequences With Recurrent Neural Networks](https://arxiv.org/pdf/1308.0850)
+ GRU, following Kyunghyun Cho et al. 2014 [On the Properties of Neural Machine Translation: Encoder–Decoder Approaches](https://arxiv.org/pdf/1409.1259)
+ Transformer, following Vaswani et al. 2017 [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)