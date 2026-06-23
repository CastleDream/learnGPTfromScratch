- [P2: 语言建模详解：构建 makemore](#p2-语言建模详解构建-makemore)
  - [链接](#链接)
  - [关键内容](#关键内容)

# P2: 语言建模详解：构建 makemore
## 链接
B站视频链接：
+ 2026年1月9日的重制版本：[Andrej Karpathy【中英⚡从零构建 GPT（重制版）|Neural Networks: Zero to Hero】](https://www.bilibili.com/video/BV1mqrTBvEaf/?p=2&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=1019ffdc843339404e9df6ae52ff9e77)

Github项目： <https://github.com/karpathy/makemore>

## 关键内容

`makemore` 是一个自回归字符级语言模型(`autoregressive character-level language model`), 会将训练数据的每行作为一个样本，对每行的字符序列进行建模，能够预测序列中的下一个字符
+ 字符级语言模型的任务是：在给定一段序列的情况下，预测下一个字符
+ 以`isabella`这串字符为例，说明：
    1. `I`很有可能是名字的首字母
    2. `s`以及后续的都可能会在`i`之后出现
    3. `isabella`之后，字符结束，这里需要一个停止生成的符号