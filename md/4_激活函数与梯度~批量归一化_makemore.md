- [P4: 构建 makemore 第三部分：激活函数与梯度，批量归一化](#p4-构建-makemore-第三部分激活函数与梯度批量归一化)
  - [链接](#链接)
  - [关键内容](#关键内容)

# P4: 构建 makemore 第三部分：激活函数与梯度，批量归一化

## 链接

B站视频链接：
+ [Andrej Karpathy【中英⚡从零构建 GPT（重制版）|Neural Networks: Zero to Hero】](https://www.bilibili.com/video/BV1mqrTBvEaf/?p=3&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=1019ffdc843339404e9df6ae52ff9e77)

Github项目：
+ <https://github.com/karpathy/makemore>
+ [Github: karpathy/nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero)

## 关键内容

需要更进一步了解激活值和反向传播，这样才能理解为什么RNN(循环神经网络)作为一种通用近似器，理论上可以实现所有算法，但是不容易通过我们常用的一阶梯度优化方法来进行高效训练

