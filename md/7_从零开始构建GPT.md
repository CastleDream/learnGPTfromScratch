- [P7: 从零开始，用代码构建 GPT](#p7-从零开始用代码构建-gpt)
  - [链接](#链接)
  - [关键内容](#关键内容)

# P7: 从零开始，用代码构建 GPT
## 链接
B站视频链接：
+ [Andrej Karpathy【中英⚡从零构建 GPT（重制版）|Neural Networks: Zero to Hero】](https://www.bilibili.com/video/BV1mqrTBvEaf/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=1019ffdc843339404e9df6ae52ff9e77)

Github项目：
+ <https://github.com/karpathy/makemore>
+ [Github: karpathy/nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero)

## 关键内容
+ 肯定不是复现chatGPT，而是实现一个基于Transformer的网络，同时也不是word-level的，而依然是character-level的
+ 这里老师选择的数据集是 [karpathy/tiny_shakespeare](https://huggingface.co/datasets/karpathy/tiny_shakespeare), 或者Github上的：<https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt>
+ 可以尝试下金庸？？？

完整的训练过程位于： [karpathy/nanoGPT](https://github.com/karpathy/nanogpt)

这节课用到的notebook地址(Google Colab): https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing