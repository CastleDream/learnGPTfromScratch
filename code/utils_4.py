import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt # for making figures

g = torch.Generator().manual_seed(2147483647) # for reproducibility

class Linear:
  """https://docs.pytorch.org/docs/2.13/generated/torch.nn.Linear.html
  """
  def __init__(self, fan_in, fan_out, bias=True):
    self.weight = torch.randn((fan_in, fan_out), generator=g) / fan_in**0.5
    self.bias = torch.zeros(fan_out) if bias else None
  
  def __call__(self, x):
    self.out = x @ self.weight
    if self.bias is not None:
      self.out += self.bias
    return self.out
  
  def parameters(self):
    return [self.weight] + ([] if self.bias is None else [self.bias])


class BatchNorm1d:
  """https://docs.pytorch.org/docs/2.13/generated/torch.nn.BatchNorm1d.html
  """
  def __init__(self, dim, eps=1e-5, momentum=0.1):
    self.eps = eps
    self.momentum = momentum
    self.training = True                                               # 并没有体现在初始化的参数中，是后续实例化后直接调用修改
    # https://github.com/pytorch/pytorch/blob/v2.13.0/torch/nn/modules/batchnorm.py#L187 pytorch里的实现，也确实有training这个bool值
    # parameters (trained with backprop)
    self.gamma = torch.ones(dim)
    self.beta = torch.zeros(dim)
    # buffers (trained with a running 'momentum update')
    self.running_mean = torch.zeros(dim)
    self.running_var = torch.ones(dim)                                 # 这里用的是方差，不是标准差了，完全和论文保持一致了       
  
  def __call__(self, x):
    # calculate the forward pass
    if self.training:
      xmean = x.mean(0, keepdim=True) # batch mean
      xvar = x.var(0, keepdim=True) # batch variance

      # update the buffers
      with torch.no_grad():
        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
    else:
      xmean = self.running_mean
      xvar = self.running_var
    xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
    self.out = self.gamma * xhat + self.beta                         
   
    return self.out
  
  def parameters(self):
    return [self.gamma, self.beta]  # 对应pytorch里BN层的weight和bias，不会返回running_mean和running_var，后两个是滑动平均算的，不需要依靠梯度下降算法和反向传播进行计算

class Tanh:
  """https://docs.pytorch.org/docs/2.13/generated/torch.nn.Tanh.html
  """
  def __call__(self, x):
    self.out = torch.tanh(x)
    return self.out
  def parameters(self):
    return []
  
def run_without_tanh(gain, vocab_size, Linear, block_size, Xtr, Ytr):
    n_embd = 10 # the dimensionality of the character embedding vectors
    n_hidden = 100 # the number of neurons in the hidden layer of the MLP
    
    C = torch.randn((vocab_size, n_embd),            generator=g)
    layers = [
      Linear(n_embd * block_size, n_hidden),
      Linear(           n_hidden, n_hidden), 
      Linear(           n_hidden, n_hidden), 
      Linear(           n_hidden, n_hidden), 
      Linear(           n_hidden, n_hidden), 
      Linear(           n_hidden, vocab_size),
    ]
    # print(layers[0].parameters()[0])
    with torch.no_grad():
      layers[-1].weight *= 0.1
      for layer in layers[:-1]:
        if isinstance(layer, Linear):
          layer.weight *= gain
        
    parameters = [C] + [p for layer in layers for p in layer.parameters()]
    print(sum(p.nelement() for p in parameters)) # number of parameters in total
    for p in parameters:
      p.requires_grad = True  # 明确告知pytorch这些参数需要求梯度
    
    max_steps = 200000
    batch_size = 32
    lossi = []
    
    for i in range(max_steps):
      # minibatch construct
      ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
      Xb, Yb = Xtr[ix], Ytr[ix] # batch X,Y
      
      # forward pass
      emb = C[Xb] # embed the characters into vectors
      x = emb.view(emb.shape[0], -1) # concatenate the vectors
      for layer in layers:
        x = layer(x)
      loss = F.cross_entropy(x, Yb) # loss function
      
      # backward pass
      for layer in layers:
        layer.out.retain_grad() # AFTER_DEBUG: would take out retain_graph
      for p in parameters:
        p.grad = None
      loss.backward()
      
      # if i == 0:
      #   print(layers[0].parameters()[0].grad)
      
      lr = 0.1 if i < 150000 else 0.01 # step learning rate decay
      for p in parameters:
        p.data += -lr * p.grad

      if i % 10000 == 0: # print every once in a while
        print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
      lossi.append(loss.log10().item())
      break
    return layers

# cd code
# python utils_4.py
if __name__ == '__main__':
    words = open('names.txt', 'r').read().splitlines()
    chars = sorted(list(set(''.join(words))))
    stoi = {s:i+1 for i,s in enumerate(chars)}
    stoi['.'] = 0
    itos = {i:s for s,i in stoi.items()}
    vocab_size = len(itos)
    block_size = 3 # context length: how many characters do we take to predict the next one?

    def build_dataset(words):  
        X, Y = [], []
        for w in words:
            context = [0] * block_size
            for ch in w + '.':
                ix = stoi[ch]
                X.append(context)
                Y.append(ix)
                context = context[1:] + [ix] # crop and append
        X = torch.tensor(X)
        Y = torch.tensor(Y)
        print(X.shape, Y.shape)
        return X, Y

    import random
    random.seed(42)
    random.shuffle(words)
    n1 = int(0.8*len(words))
    n2 = int(0.9*len(words))

    Xtr,  Ytr  = build_dataset(words[:n1])     # 80%
    Xdev, Ydev = build_dataset(words[n1:n2])   # 10%
    Xte,  Yte  = build_dataset(words[n2:])     # 10%

    layers = run_without_tanh(5/3, vocab_size, Linear, block_size, Xtr, Ytr) # gain = 1
    print(layers[0].parameters())
    
    plt.figure(figsize=(20, 4))
    legends = []
    for i, layer in enumerate(layers[:-1]): 
        if isinstance(layer, Linear):
            t = layer.out
            print('layer %d (%10s): mean %+.2f, std %.2f, saturated: %.2f%%' % (i, layer.__class__.__name__, t.mean(), t.std(), (t.abs() > 0.97).float().mean()*100))
            t = layer.out.grad
            print('layer %d (%10s): mean %+f, std %e' % (i, layer.__class__.__name__, t.mean(), t.std()))
            hy, hx = torch.histogram(t, density=True)   
            # 直接报错 RuntimeError: torch.histogramdd: dimension 0's range [-nan(ind), -nan(ind)] is not finite
            plt.plot(hx[:-1].detach(), hy.detach())
            legends.append(f'layer{i}_{layer.__class__.__name__}')
        plt.legend(legends);
        plt.title('activation distribution')
    plt.show()

