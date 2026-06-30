
- [P1: 神经网络与反向传播详解：构建 micrograd](#p1-神经网络与反向传播详解构建-micrograd)
  - [链接](#链接)
  - [关键内容](#关键内容)

# P1: 神经网络与反向传播详解：构建 micrograd
## 链接
B站视频链接：
+  [【精译⚡从零开始构建 GPT 系列】Andrej Karpathy](https://www.bilibili.com/video/BV11yHXeuE9d/?spm_id_from=333.337.search-card.all.click&vd_source=1019ffdc843339404e9df6ae52ff9e77)
+ 2026年1月9日的重制版本：[Andrej Karpathy【中英⚡从零构建 GPT（重制版）|Neural Networks: Zero to Hero】](https://www.bilibili.com/video/BV1mqrTBvEaf?buvid=XUA8C1BA16956A5648F049B542ACCF70B8D30&from_spmid=tm.recommend.0.0&is_story_h5=false&mid=Is1xlji2jfaisvby%2F4iEVQ%3D%3D&plat_id=116&share_from=ugc&share_medium=android&share_plat=android&share_session_id=d7996f03-32d2-49a0-942f-15848e8df474&share_source=COPY&share_tag=s_i&spmid=united.player-video-detail.0.0&timestamp=1767974026&unique_k=cz6kfKa&up_id=18053089&vd_source=1019ffdc843339404e9df6ae52ff9e77)

Github项目： 
+ <https://github.com/karpathy/makemore>
+ [Github: karpathy/nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero)

## 关键内容
只记录一些重点内容，毕竟不再是初阶教程了，更直接一点

micrograd是一个标量值自动微分引擎（scalar-value autograd engine），是在单个标量上进行工作的，而不是在传统的张量库（例如：Pytorch等）上对张量进行操作的；
+ 虽然对于神经网络来说，要把所有的张量一路分解成标量来进行计算有些繁琐
+ 但是这个主要是用于教学目的，而不是生产环境
+ 抛开n维张量，只聚焦于反向传播和链式法则，真正理解神经网络的训练


不建议只听，还是要跟着jupyter敲一遍，哪怕是巩固知识和维持代码手感~
