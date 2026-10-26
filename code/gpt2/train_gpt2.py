"""
[huggingface/src/transformers/models/gpt2/modeling_gpt2.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py)

不直接用transformers库的GPT2LMHeadModel，而是自己实现一个简单的GPT2模型，方便后续修改和调试。
1. 确保可以加载OpenAI开源的gpt2模型权重
2. 用自己训练的数据去训练新的模型权重，确保可以超过原始的gpt2模型

可以对照之前加载的gpt2的模型看看每层的变量叫什么

# 两个嵌入层
transformer.wte.weight torch.Size([50257, 768])
transformer.wpe.weight torch.Size([1024, 768])

# 这里应该有个list, 因为不是string索引,而是数字索引了
transformer.h.0.ln_1.weight torch.Size([768])
transformer.h.0.ln_1.bias torch.Size([768])
transformer.h.0.attn.c_attn.weight torch.Size([768, 2304])
transformer.h.0.attn.c_attn.bias torch.Size([2304])
transformer.h.0.attn.c_proj.weight torch.Size([768, 768])
transformer.h.0.attn.c_proj.bias torch.Size([768])
transformer.h.0.ln_2.weight torch.Size([768])
transformer.h.0.ln_2.bias torch.Size([768])
transformer.h.0.mlp.c_fc.weight torch.Size([768, 3072])
transformer.h.0.mlp.c_fc.bias torch.Size([3072])
transformer.h.0.mlp.c_proj.weight torch.Size([3072, 768])
transformer.h.0.mlp.c_proj.bias torch.Size([768])
transformer.h.1. 
...
transformer.h.11.mlp.c_proj.weight torch.Size([3072, 768])
transformer.h.11.mlp.c_proj.bias torch.Size([768])
transformer.ln_f.weight torch.Size([768])
transformer.ln_f.bias torch.Size([768])
lm_head.weight torch.Size([50257, 768])
"""
from dataclasses import dataclass
import torch.nn as nn
from torch.nn import functional as F



# @dataclass 自动生成类中那些重复的“样板代码”（Boilerplate code），让你用最简洁的方式定义数据结构
# 会在类创建时，自动根据你定义的类属性（带有类型注解的变量），自动生成 __init__、__repr__ 和 __eq__ 方法。 代码量减少了 80% 以上，而且不易出错。
@dataclass
class GPT2Config:
    block_size: int =256
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_embed: int = 384


class CasualSelfAttention(nn.Module):
    """

    """

class MLP(nn.Module):
    """
     FFN(x) = max(0, xW1 + b1)W2 + b2
     The dimensionality of input and output is dmodel = 512, and the inner-layer has dimensionality
dff = 2048.
    """
    def __init__(self, config:GPT2Config):
        """
            这里线性层维度有点问题，去看 
            10_复现 GPT-2.md## 3.2 TensorFlow vs huggingface(pytorch)历史差异
            Huggingface的实现用的是自定义的Conv1D
        """
        super().__init__()
        self.c_fc = nn.Linear(4*config.n_embed, config.n_embed)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(config.n_embed, 4*config.n_embed)
    
    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))



class Block(nn.Module):
    def __init__(self, config:GPT2Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        # config 仅作为 __init__ 方法的局部参数。初始化完成后，config 对象如果没有被其他地方引用，就会被垃圾回收，实例内部不保留对它的引用。
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)
        # 注意力机制的作用: 完成输入的序列之间的信息交换，token之间的依赖关系建模
        # 本质是一种 aggregation function(聚合函数)/pooling function/weighted sum function(加权求和函数)/reduce operation(归约操作)
        # MLP的作用：独立作用于每个token，token之间没有信息交换，主要是对每个token的特征进行非线性变换，增加模型的表达能力
        # Attention: reduce归约操作，MLP: map映射操作，都是对token的特征进行变换，Attention是token之间的特征变换，MLP是token自身的特征变换
        # 从这个角度考虑，Transformer本质就是对MapReduce这种模式的反复应用
        # MapReduce 最初是 Google 提出的一种分布式计算编程模型，其核心思想是将复杂的计算拆解为两个基本操作：
        # Map（映射）：对输入数据集里的每一个元素独立地应用一个函数，生成中间结果。核心特征是 “并行、独立、无状态交互”。
        # Reduce（归约）：将 Map 阶段产生的、具有某种关联（如相同 Key）的中间结果聚合在一起，应用一个聚合函数，生成最终结果。核心特征是 “跨元素的信息交互与汇总”。
    
    def forward(self, x):
        # 注意，这里的残差是Attention/FFN之前，到Norm层之后的
        # residual pathway has normalization inside them
        # 这样设计可以让反向的顶层的梯度直接流入输入端的词元，保持一个干净的残差流
        # 这种是 pre-norm 的设计，和 post-norm 的设计不同，post-norm 是在残差之后再做norm
        x = x + self.attn(self.ln_1(x))  
        x = x + self.mlp(self.ln_2(x))
        return x
        
class GPT2(nn.Module):
    """
        为了保证可以顺利加载gpt2的权重，GPT2模型的结构和参数需要和原始的GPT2模型保持一致。
    """
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        # 将 config 赋值给了 self.config，使其成为实例的属性。这意味着在整个 Block 实例的生命周期内，都可以随时通过 self.config 访问配置信息。
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(self.config.vocab_size, self.config.n_embed),
            wpe = nn.Embedding(self.config.block_size, self.config.n_embed),
            h = nn.ModuleList(Block(config) for _ in range(self.config.n_layer)),
            ln_f = nn.LayerNorm(self.config.n_embed) 
        ))
        # weight (torch.Tensor) – the learnable weights of the module of shape (out_features,in_features)
        # https://docs.pytorch.org/docs/2.14/generated/torch.nn.Linear.html
        self.lm_head = nn.Linear(self.config.n_embed, self.config.vocab_size, bias = False)
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py#L636
        # 这里lm_head确实用的是nn.Linear, 而mlp里用的是自定义的Conv1D

        # 对照论文的话，其实模型的基本框架很简单，就是这里的 self.transformer + self.lm_head
        # 之后就是去实现Block就可以
    

        