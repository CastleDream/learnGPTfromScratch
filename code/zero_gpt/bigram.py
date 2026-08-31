"""
把 7_start_GPT.ipynb 的草稿代码规范化成脚本

详细说明都在 7_start_GPT.ipynb 中
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# 超参设置
batch_size = 32
block_size = 8
max_iters = 5000
eval_interval = 300
learning_rate = 1e-3  # 自注意力机制无法接受很高的学习率，所以不能是1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 32  # embedding维度

# 随机数种子固定
torch.manual_seed(1337)

# 读取数据 + 构建词表
text = open("code/zero_gpt/input.txt", 'r').read()
chars = sorted(list(set(text)))
vocab_size = len(chars)

# encode 和 decode 需要的映射和lambda函数
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for ch,i in stoi.items()}

encode = lambda s: [stoi[c] for c in s]  # lambda函数， s表示sentence/string 把一串字符编码为整数列表
decode = lambda l: "".join([itos[i] for i in l])

# 划分训练测试集
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split, batch_size = 4):
    """split: 数据集名称, 例如: train_data/val_data"""
    data = train_data if split=="train" else val_data
    ix = torch.randint(len(data)-block_size, (batch_size,))  
    x = torch.stack([data[i:i+block_size] for i in ix]) 
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) 
    x,y = x.to(device), y.to(device) # 多加了这句
    return x,y


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters) # eval_iters 上面定义的超参=200
        # 这里直接用200次的batch的损失来作为评估损失，和以前8:2 直接用20%数据上的作为验证损失，差不多
        # 肯定比只用一个batch的验证损失更稳定，更有说服力
        for k in range(eval_iters):
            X, Y = get_batch(split, batch_size=eval_iters)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # 评估结束后，恢复模型的训练模式
    # 虽然这个网络目前没有用BN等训练/评测行为不一致的层，但是养成思考mode的习惯还是很重要的
    return out

class Head(nn.Module):
    """single-head 自注意力机制(这里是解码器的自注意力机制 因为有mask 看不到未来信息 只能看到过去信息)
    对应论文里的 缩放点积注意力机制  Scaled Dot-Product Attention

    老师的实现其实是位于： https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py
    """
    def __init__(self, head_size):
        super().__init__()
        # n_embed是脚本开头定义的全局变量
        self.head_size = head_size
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        # tril不是这个nn.Module的参数，因此按照pytorch的规定，就以缓冲的形式存在了
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)        # k(B,T,head_size)
        q = self.query(x)      # q(B,T,head_size)
        v = self.value(x)      
        weight = (q@k.transpose(-2,-1))*self.head_size**(-0.5)   # q@k^T → (B,T,T) 需要乘以 head_size**(-0.5)
        weight = weight.masked_fill(self.tril[:T,:T]==0, float('-inf'))
        weight = F.softmax(weight, dim = -1)
        out = weight@v        # (B,T,T)@(B,T,head_size) → (B,T,head_size)
        return out


class MultiHeadAttention(nn.Module):
    """

    参考：
    torch/nn/modules/activation.py # MultiheadAttention
    https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/activation.py#L1090
    """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
    
    def forward(self, x):
        # 多头自注意力机制，本质上就是让多个自注意力机制头并行运行，再把结果拼接到一起
        # 根据上面的Heads可知 h(x).shape = (B,T,head_size) 所以最后在head_size上拼接，刚好就可以得到 num_heads*head_size的维度
        # 即：(B, T, num_heads*head_size)
        out = torch.cat([h(x) for h in self.heads], dim = -1)
        out = self.proj(out)
        # 这里加的这个线性层是为了让多头信息更好的融合，多头直出的结果是拼接的，并没有进行计算融合
        return out


class FeedForward(nn.Module):
    """
    不难发现，前馈层里的线性层，处理的对象是单个token，是token级别的处理
    即：
    自注意机制负责的token之间的信息交互，在token交互完数据之后(gather之后)
    需要每个token自身再整理一遍信息
    """
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, n_embed),
            nn.ReLU()
        )
    
    def forward(self,x):
        return self.net(x)

class Block(nn.Module):
    """构建 自注意力机制+前馈层的块，方便之后重复多次调用
    """
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed//n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
    
    def forward(self, x):
        # x = self.sa(x)     # 先token之间的信息交互
        # x = self.ffwd(x)   # 再单个token信息整理/计算
        # 添加残差连接，关于为什么是分开加了两次，详见：https://arxiv.org/pdf/2002.04745v1 里的Fig.1
        x = x + self.sa(x)   
        x = x + self.ffwd(x)   
        return x


class BigramLanguageModel(nn.Module):
    # vocab_size已经是全局变量了，这里没必要再传入了
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)

        # 为了把token_embedding的值转为logits值，这里加一个线性层
        # 根据 https://docs.pytorch.org/docs/2.13/generated/torch.nn.Linear.html
        # 线性层的in_features 和out_features 分别对应输入和输出的最后一个维度，即 Input:(*, in_features) -> Output:(*, out_features)
        # 所以下面 logits = self.lm_head(x)，输入是(B,T,n_embed)，输出是(B,T,vocab_size)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        # lm_head 即 language model head，语言模型的头部的简称

        # 不仅要编码词元的身份信息(identity), 即 tok_emb，还要编码位置信息(position)
        # 因此引入第二个table
        self.position_embedding_table = nn.Embedding(block_size, n_embed) # 注意，这里的位置索引是 [0, block_size-1]个n_embed维的嵌入

        # 加一个自注意力机制头 sa_head表示 self attention head
        # self.sa_head = Head(n_embed)  # 这里给的比较简单，直接用n_embed作为head_size了
        # 改为多头自注意力机制
        # self.sa_head = MultiHeadAttention(4, n_embed//4)  # 4个头，每个head的head_size是32//4=8
        # self.ffwd = FeedForward(n_embed)

        # 用block代替上面单个组件
        self.blocks = nn.Sequential(
            Block(n_embed, 4),
            Block(n_embed, 4),
            Block(n_embed, 4),
        )

    
    def forward(self, idx, targets= None):
        B,T = idx.shape # batch size, time steps/block size
        # logits = self.token_embedding_table(idx)

        # 不会再直接用词表嵌入查询得到的结果直接作为logits值了
        tok_emb = self.token_embedding_table(idx)  # token_embedding  (B,T,C，这里的C就是n_embed)
        # 还需要引入位置编码，位置嵌入不会随着batch改变，所以一个就够了
        pos_emb = self.position_embedding_table(torch.arange(T,device=device)) # (T,C) [0,T-1]个n_embed维的嵌入
        # 这里其实要确定 idx的T不能超过block_size 因为超过的话 这个索引过程就会越界，会报错

        x = tok_emb + pos_emb 
        # 直接对后者广播就可以，所以位置嵌入在一个batch里是一样的
        # (B,T,n_embed) + (T, n_embed)-> (B,T,n_embed)    
        # 这里的位置嵌入暂时没啥作用，因为还只是个简单的bigram模型，这里没有自注意力机制等，所以不管是哪个位置，在这种情况下都具有平移不变性(translation invariance)，所以位置嵌入的作用不大
        
        # 把经过词嵌入和位置嵌入的结果送到 自注意力机制里
        # 其实如果输出维度保持不变，多头和单头的关系有点像分组卷积 group convolution
        # 详见 https://stitch.blog.csdn.net/article/details/121920678 2. 标准多输出通道示意动图
        # https://zhuanlan.zhihu.com/p/28749411
        # https://docs.pytorch.org/docs/2.13/generated/torch.nn.Conv2d.html  这里其实有个groups参数
        # 虽然和分组卷积很像，但是多头注意力机制和同样输出维度的单头的参数量是一样的，并没有减少参数和计算量
        # self.sa_head = Head(n_embed)
        # self.sa_head = MultiHeadAttention(4, n_embed//4)
        # x = self.sa_head(x)    # 这里返回的结果是 (B,T,n_embed) 刚好和下面这层的输入对上了
        # x = self.ffwd(x)
        # 用block代替上面单个组件
        x = self.blocks(x)
        logits = self.lm_head(x) # (B,T,C,这里的C就是vocab_size, 线性层只对最后一个维度做线性变换) 
        
        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits.view(-1,vocab_size), targets.view(-1))
        return logits,loss
    
    def generate(self, idx, max_new_token):
        for _ in range(max_new_token):
            # 这里其实要确定 idx的T不能超过block_size 因为超过的话 这个索引过程就会越界，会报错。主要是对position_embedding_table的限制
            # 对idx进行截断(训练的时候因为输入都是处理过的，所以不会超过block_size)，但是生成的时候，上下文长度随着生成过程会越来越长。。所以需要限制
            # 每次只保留最新的block_size个上下文
            idx_cond = idx[:, -block_size:]
            logits,loss = self.forward(idx_cond) 
            logits = logits[:,-1,:] 
            probs = F.softmax(logits,dim = -1) 
            idx_next = torch.multinomial(probs, num_samples = 1)
            idx = torch.cat((idx, idx_next), dim = 1) # (B, T+1)
        return idx


def train_bigram():
    model = BigramLanguageModel()
    m = model.to(device)
    # print(f"model is m? ", model is m )  # model is m?  True     
    # model 和 m 其实指向同一个模型对象。
    # model = BigramLanguageModel(vocab_size).to(device) # 这样写其实更好
    optimizer = torch.optim.AdamW(m.parameters(), lr = 1e-3)
    # 在 PyTorch 里，nn.Module.to(device) 一般是 in-place 操作，也就是它会把 model 本身的参数、buffer 等移动到指定设备，然后 返回同一个对象 self。

    for iters in range(max_iters):
        # 每eval_interval步，评估一次训练集和验证集的损失
        if iters%eval_interval == 0:
            losses = estimate_loss(m)
            print(f"step {iters}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        xb,yb = get_batch('train', batch_size)
        logits, loss = m(xb,yb)
        optimizer.zero_grad(set_to_none = True)
        loss.backward()
        optimizer.step()
    print(loss.item())

    context = torch.zeros((1,1), dtype=torch.long, device=device)
    print(decode(m.generate(context, max_new_token=500)[0].tolist()))


def plot_head_size_diff():
    """画一下head_size=8和32的softmax结果函数曲线和方差
    """
    import numpy as np
    import matplotlib.pyplot as plt
    x = np.arange(0,1,0.05)
    title = ''
    for head_size in [8,32]:
        output = x*head_size**(-0.5)
        prob = np.exp(output)/sum(np.exp(output))  # softmax
        var = prob.var()
        title+=f"{head_size}: softmax var = {var:.6e}\n"
        print(f"head_size = {head_size}, 计算得到的 var 值为: {var}")
        plt.plot(x, output, label=f'output = x * {head_size}^(-0.5)')
    plt.title(title.strip())
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
    

# python code/zero_gpt/bigram.py
if __name__ == "__main__":
    train_bigram()
    # plot_head_size_diff()


# 不加自注意力机制的结果              2.8591 
#                                     ↓ 
# 加单个自注意力机制的结果            2.363  生成的东西看着有点正常语法的样子了
#                                     ↓
# 加相同输出维度的多头自注意力的结果   2.1640  生成的更好一些了 说明词云之间真的有很多信息需要交流
# 关于为什么相同输出维度的多头比单头效果好 详见： md\a_难点查漏补缺.md
#                                     ↓
# 多头自注意力 + feedforward         2.206  可能有一些随机偏差
#                                     ↓
# 多个block实现的sa+ffwd             2.289  由于网络开始变得很深，但是数据又比较少，就很容易出现优化上的问题，因此考虑加入skip connection(或者称为残差连接)


# 不加自注意力机制的结果
# max_iters = 3000
# step 0: train loss 4.7265, val loss 4.7260
# step 300: train loss 4.3852, val loss 4.3876
# step 600: train loss 4.0764, val loss 4.0828
# step 900: train loss 3.8123, val loss 3.8163
# ...
# step 2400: train loss 2.9648, val loss 2.9724
# step 2700: train loss 2.8758, val loss 2.8779
# 2.8591721057891846

# 加单个自注意力机制的结果
# step 0: train loss 4.2028, val loss 4.2032
# step 300: train loss 2.9120, val loss 2.9228
# step 600: train loss 2.6522, val loss 2.6631
# ...
# step 4500: train loss 2.3946, val loss 2.4124
# step 4800: train loss 2.3894, val loss 2.4132
# 2.3630290031433105

# 加相同输出维度的多头自注意力的结果
# step 0: train loss 4.1781, val loss 4.1798
# step 300: train loss 2.8335, val loss 2.8463
# step 600: train loss 2.6217, val loss 2.6263
# step 900: train loss 2.5216, val loss 2.5223
# ...
# step 4200: train loss 2.2625, val loss 2.2877
# step 4500: train loss 2.2575, val loss 2.2848
# step 4800: train loss 2.2456, val loss 2.2790
# 2.1640796661376953

# 多头自注意力 + feedforward
# step 0: train loss 4.2493, val loss 4.2469
# step 300: train loss 2.7881, val loss 2.8002
# step 600: train loss 2.5686, val loss 2.5719
# step 900: train loss 2.4917, val loss 2.4942
# ...
# step 4200: train loss 2.2480, val loss 2.2639
# step 4500: train loss 2.2416, val loss 2.2673
# step 4800: train loss 2.2276, val loss 2.2546
# 2.2067973613739014

# 多个block实现的多头自注意力 + feedforward
# step 0: train loss 4.2087, val loss 4.2076
# step 300: train loss 3.1733, val loss 3.1852
# step 600: train loss 2.8543, val loss 2.8415
# step 900: train loss 2.7089, val loss 2.6986
# ....
# step 4200: train loss 2.3312, val loss 2.3362
# step 4500: train loss 2.3278, val loss 2.3323
# step 4800: train loss 2.3057, val loss 2.3124
# 2.289051055908203