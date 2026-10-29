"""
[huggingface/src/transformers/models/gpt2/modeling_gpt2.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py)
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
https://github.com/karpathy/nanoGPT/blob/master/model.py

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
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" # 设置 HuggingFace 官方推荐的国内镜像源 (hf-mirror)
os.environ["HF_HOME"] = "E:/learnGPT/learnGPTfromScratch/other"  # 如果需要指定本地缓存路径（建议大模型单独指定目录，避免占满系统盘）
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math



# @dataclass 自动生成类中那些重复的“样板代码”（Boilerplate code），让你用最简洁的方式定义数据结构
# 会在类创建时，自动根据你定义的类属性（带有类型注解的变量），自动生成 __init__、__repr__ 和 __eq__ 方法。 代码量减少了 80% 以上，而且不易出错。
@dataclass
class GPT2Config:
    """ 
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/configuration_gpt2.py
        这里给了配置参数
        https://github.com/openai/gpt-2/blob/master/src/model.py
        原始的openai的实现也很精简
    """
    block_size: int = 1024
    vocab_size: int = 50257 # gpt2论文里给了
    n_layer: int = 12  # gpt2论文里给了
    n_head: int = 12  # 
    n_embed: int = 768 # gpt2论文里给了


class CasualSelfAttention(nn.Module):
    """
        这部分参考以前的 code/zero_gpt/bigram.py 里实现的多头注意力机制
        去看这部分实现就记得了
        https://github.com/karpathy/nanoGPT/blob/master/model.py
        transformer.h.0.attn.c_attn.weight torch.Size([768, 2304])
        transformer.h.0.attn.c_attn.bias torch.Size([2304])
        transformer.h.0.attn.c_proj.weight torch.Size([768, 768])
        transformer.h.0.attn.c_proj.bias torch.Size([768])
        所以attn里只有 c_attn和c_proj两个线性层
        c_attn 前缀 c 代表 Conv1D（也有解释为 Combined，因为合并了 Q/K/V 的计算），attn 代表 attention
        c_proj 前缀 c 代表 GPT-2 原始代码中使用的 Conv1D 操作，proj 代表 projection（投影）
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_head==0  # 要把这个维度均分到每个head上去
    
        # key,query,value的线性变换矩阵，注意这里是一次性生成的，后面会切分成多头
        # key,query,value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embed, 3*config.n_embed) 

        # MultiHead(Q, K, V ) = Concat(head1, ..., headh)WO
        self.c_proj = nn.Linear(config.n_embed, config.n_embed) 

        self.n_embed = config.n_embed
        self.n_head = config.n_head

        # 这里命名为bias，但是本质是mask的作用，只是为了和openai/hf的实现保持一致
        # self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1,1,config.block_size, config.block_size))
        self.register_buffer("bias", torch.tril(torch.ones(1, 1, config.block_size, config.block_size)))
        # torch.tril不是只能接受2d输入的， 
        # torch.tril 完美支持多维张量。规则：它总是把输入张量的最后两个维度视为矩阵的行和列，而将前面的所有维度视为 Batch 维度，并对每一个 Batch 独立执行下三角操作。

    def forward(self,x):
        B,T,C = x.size() # batchsize, sequence length, embedding dimensionality(n_embed)
        # calculate query, key, values for all heads in batch 
        # and move head forward to be the batch dim
        # nh is "number of heads",  hs is "head size", 
        # C(number of channels) = nh*ns =  embedding dimensionality(n_embed)
        # eg. in GPT-2(124M), n_head=12, ns = 64 → nh*hs= 768 channel in the Transformer
        qkv = self.c_attn(x) # (B,T,C)*(C, 3C) = (B,T,3C)
        q,k,v = qkv.split(self.n_embed, dim = 2) # 沿着dim=2，每隔n_embed切一次
        # 等价于之前的
        # k = self.key(x)        # k(B,T,head_size)
        # q = self.query(x)      # q(B,T,head_size)
        # v = self.value(x) 
        
        # 人工把一个qkv从1个head分成了n_head, 最后在nh上合并，就好了
        # 不像以前是定义好单头，再self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])构成多头
        # 这里交换位置是为了让pytorch可以把nh也当做和B一样的batch维度，
        # 而只对后面两个维度进行矩阵计算，这样后面的计算就是可以并行的
        k = k.view(B,T,self.n_head,C//self.n_head).transpose(1,2) # (B,nh,T,ns)
        q = q.view(B,T,self.n_head,C//self.n_head).transpose(1,2) 
        v = v.view(B,T,self.n_head,C//self.n_head).transpose(1,2) 

        # attention(materializes the large(T,T)matrix for all the queries and keys)
        # softmax(QK^T/(dk**0.5))V
        # q和k是矩阵乘法，(T,ns)@(ns,T)→(T,T)
        # att = (q@k.transpose(-2,-1))*(k.size(-1))**(-0.5)
        att = (q@k.transpose(-2,-1))*(1/math.sqrt(k.size(-1))) #(B,nh,T,T)
        att = att.masked_fill(self.bias[:,:,:T,:T]==0,float('-inf'))   ### ❗❗❗❗❗这里是 -inf 是负无穷
        # 实际输入的T肯定不会超过config.block_size，所以这里self.bias初始化用block_size也没问题
        att = F.softmax(att, dim = -1)
        y = att@v  #(B,nh,T,T)@(B,nh,T,ns)=(B,nh,T,ns)
        # re-assemble all head outputs side by side 重新组装所有head的输出
        y = y.transpose(1,2).contiguous().view(B,T,C)  # (B,nh,T,ns)→(B,T,nh,ns)→(B,T,C)

        # MultiHead(Q, K, V ) = Concat(head1, ..., headh)WO
        y = self.c_proj(y)
        return y

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
        self.c_fc = nn.Linear(config.n_embed, 4*config.n_embed)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4*config.n_embed, config.n_embed)
    
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

        https://github.com/openai/gpt-2/blob/master/src/model.py
        原始的openai的实现也很精简 没有引用乱七八糟的库 174行
        
        而hf的实现不仅引入了很多其他脚本，而且单纯模型文件就有1131行，老师讲课时间的那个原始版本有2k多行
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py#L75
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/configuration_gpt2.py
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
    
    # @classmethod 将类中的普通方法转换为了“类方法”，它绑定的是“类本身”，而不是“类的实例”。
    # 下面方法的第一个参数 cls代表 class，不需要手动传入它，Python 解释器会自动将类本身作为第一个参数传入。
    # 它可以访问和修改类属性（类变量），但不能访问实例属性（因为它没有 self，不知道具体是哪个实例）
    # 既可以通过类名调用，也可以通过实例调用。但无论哪种，传入的 cls 始终是 类名
    # @classmethod 最核心的用途是实现工厂模式和提供多种实例化方式。
    # 作为“替代构造函数”（工厂方法）
    # 在 LLM 中，我们经常需要根据不同的来源（配置文件、预训练权重路径、URL 等）来初始化模型。
    # Python 的 __init__ 只能有一种签名，但 @classmethod 可以帮我们实现多个优雅的构造函数。
    @classmethod
    def from_pretrained(cls, model_type):
        """loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'} 
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # 老师给的代码里其实是 n_embd，我这里前面全都是n_embed，所以全文一致即可
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embed=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embed=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embed=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embed=1600), # 1558M params
        }[model_type]    
        config_args["block_size"] = 1024
        config_args["vocab_size"] = 50257

        config = GPT2Config(**config_args)
        model = GPT2(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith(".attn.bias")]
        # discard this mask / buffer, not a param
        # 注意，不是丢掉 .attn.c_attn.bias 或者 .attn.c_proj.bias，而是.attn.bias

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained("gpt2") 
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        # 把标准的hf加载的权重state_dict，按照key，逐个赋值给上面我们自己构建的模型类对应的key
        sd_keys_hf = sd_hf.keys()
        # print(sd_keys_hf)
        # 下面这两句有点多余，我下载到的权重和用的transformers版本没有这两个masked
        # sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        # sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)

        # 所有nn.Linear的都要把weight转置一下 详见： 10_复现 GPT-2.md/TensorFlow vs huggingface(pytorch)历史差异
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape # 转置后的shape相同，则复制权重
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape, f"key = {k}, sd_hf shape is {sd_hf[k].shape}, sd.shape is {sd[k].shape}"    
                # 如果不是位于transposed列表，则直接复制即可
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model

    def forward(self,idx):
        # idx.shape is (B,T) batch_size, sequence length
        B, T = idx.size()  #  Tensor.shape   Alias for size  See also Tensor.size()   
        # idx.size() == idx.shape https://docs.pytorch.org/docs/2.15/generated/torch.Tensor.shape.html
        # 输入的序列长度要小于限制的block_size(上下文)
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        
        # 这部分就和 code\zero_gpt\bigram.py BigramLanguageModel的forward很像了
        # 有可能会在gpu上跑，所以这里对于新创建的张量，要额外指定下设备
        pos = torch.arange(0,T,dtype=torch.long, device=idx.device) 
        pos_emb = self.transformer.wpe(pos)  # 得到(T, n_embed)    pos.shape(T)在wpe.shape(block_size, n_embed)中索引 →(T,n_embed)
        tok_emb = self.transformer.wte(idx)  # 得到(B, T, n_embed) idx.shape(B,T)在wte.shape(vocab_size,n_embed)中索引 →(B,T,n_embed)
        x = pos_emb + tok_emb # pos_emb会自动广播，在T前面插入一个维度，大小为B
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # logits.shape (B,T, vocab_size)  lm_head这个层，输入是n_embed，输出是vocab_size
        return logits
    

def sample_test():
    """复现jupyter里 
    generator("Hello, I'm a language model,", max_length = 30, max_new_tokens=30, num_return_sequences = 5)
    这部分代码完成的内容
    """
    num_return_sequences = 5
    max_length = 30
    model = GPT2.from_pretrained("gpt2")
    model.eval() # 这句话在这里可能发挥不了什么作用，因为目前这个model中没有dropout,BN等train和inference阶段表现不同的模块。问了AI，详见: 10_复现 GPT-2.md
    model.to('cuda')

    # 构建prefix token
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode("Hello, I'm a language model,")  # https://tiktokenizer.vercel.app/?model=gpt2  长度为8
    tokens = torch.tensor(tokens, dtype = torch.long) # 没在这里写 , device='cuda'
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) 
    # (5, 8) 这里是为了方便演示，所以直接把同一个输入重复五次，这样就对应 transformers的pipeline的num_return_sequences参数了
    # unsqueeze Returns a new tensor with a dimension of size one inserted at the specified position.
    # unsqueeze(0) 在开头插入一个维度，作为batch维度
    # repeat Repeats this tensor along the specified dimensions. 重复第一维的数据num_return_sequences次
    x = tokens.to('cuda')  # x is (B,T) B=5, T=8
    # print(x)

    # 开始循环生成next token
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    # max_length 这里是prefix + 新生成的 一共的长度 
    # 接下来的循环里，会给每行数据，都在后面加上新的一列(其实只有一个数字，就是next token)
    # x.size(1) 等价于 x.size()[1]
    while x.size(1) < max_length:
        with torch.no_grad():
            logits = model(x)  # (B,T,vocab_size)   # 这里其实就对对于T中的每一个，比如T=5,也就是为 1~5这五个idx都生成了next token的vocab_size每个类的概率
            # take the logits at the last position
            assert not torch.isnan(logits).any(), "Logits contain NaN before softmax!"
            assert not torch.isinf(logits).any(), "Logits contain Inf before softmax!"
            logits = logits[:,-1,:] # (B, vocab_size)  所以只需要每个T的最后一个字符，因为只看这个的next  while循环下面有一段这个代码的解释
            # get the probabilities
            probs = F.softmax(logits, dim = -1)  # torch.softmax  Alias for torch.nn.functional.softmax()
            # do top-k sampling of 50 (huggingface pipeline default)
            # https://huggingface.co/docs/transformers/v5.21.0/en/main_classes/text_generation#transformers.GenerationConfig.top_k
            # top_k： The number of highest probability vocabulary tokens to keep for top-k-filtering. This value is set in a model’s generation_config.json file. If it isn’t set, the default value is 50.
            # 从 https://huggingface.co/docs/transformers/v5.21.0/en/main_classes/pipelines#transformers.TextGenerationPipeline 找到上面的链接
            #  [Text generation strategies]() and [Text generation]()
            # 大致的处理逻辑在： https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py  generation文件夹这里面的几个脚本中

            topk_probs, topk_indices = torch.topk(probs, 50, dim = -1) # 对概率那个维度取前50概率的结果 topk_probs.shape(B,50) (5,50)
            ix = torch.multinomial(topk_probs, 1) # 对概率最高的50个值，采样1次 (5,1)，作为batch中每个序列的next token, 这里取出来的是在50个概率里的indices
            xcol = torch.gather(topk_indices, -1, ix) # 需要把ix从50个概率里的索引，还原到probs里的索引 
            # topk_indices(5,50)  ix(5,1) → xcol(5,1)  
            # 假设topk_indices是 [80,33,99,500,....],50个索引，则ix比如是[2,3]
            # 则topk_indices[ix] = [99,500],没问题
            # 但是这里topk_indices是(5,50)这样的维度，写成topk_indices[:,ix]是对每行都取[2,3] 所以不行，要用gather函数 相当于是沿着某个dim进行slice
            x = torch.cat([x, xcol], dim = -1)  # x(B,T) xcol(B,1) 这里dim写1或者-1效果是一样的

    print(f"x.shape: {x.shape}")
    for i in range(num_return_sequences):
        tokens = x[i,:].tolist()
        decoded = enc.decode(tokens)
        print(">", decoded)
    # import numpy as np
    # a = np.array([[[1,2],[3,4],[5,6]],[[1,2],[3,4],[5,6]]])
    # a.shape  # (2, 3, 2)
    # a[:,-1,:]
    # array([[5, 6],
    #    [5, 6]])
    # a
    # array([[[1, 2],
    #         [3, 4],
    #         [5, 6]],

    #        [[1, 2],
    #         [3, 4],
    #         [5, 6]]])

# python code/gpt2/train_gpt2.py
if __name__ == "__main__":
    # 测试模型结构是否正确，是否可以正确加载gpt-2权重
    # model = GPT2.from_pretrained("gpt2")
    # print("didn't crash yay!")

    sample_test()
"""
# 输出结果
> Hello, I'm a language model, not a program.

So this morning I started studying for the interview in the lab. This was not
> Hello, I'm a language model, and one of the main things that bothers me when they create languages is how easy it becomes to create something that
> Hello, I'm a language model, and I wrote it off on the grounds that a language model would make me more fluent. But I'm not
> Hello, I'm a language model, I really like languages. I like languages because like, they're good. And the way we talk about languages
> Hello, I'm a language model, a language model I'm using for data modelling. All I did was test the results and then I wrote some
"""
# 对照jupyter中用hf加载的权重，使用上面的方式生成，结果是相同的，说明模型结构是完全没问题的！所以hf里写了那么大一串代码，其实很多都是多余的。。
# 但是和原始的hf的pipeline结果不同，说明hf在采样生成的时候还有些别的操作