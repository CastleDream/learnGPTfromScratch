"""
基本的神经元定义，全连接层定义，以及MLP模块定义
ref:
https://github.com/karpathy/micrograd/blob/master/micrograd/nn.py
以及课程视频讲解
"""
from micrograd.engine import Value
import random

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0
    def parameters(self):
        return []

class Neuron(Module):
    def __init__(self, nin,nonlin):
        self.w = [Value(random.uniform(-1,1), label = 'w') for _ in range(nin)]
        self.b = Value(0, label = 'b')
        self.nonlin = nonlin

    def __call__(self,x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        out = act.tanh() 
        return out if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]
    
    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"
    
class Layer(Module):
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self,x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs)==1 else outs  
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"
    
class MLP(Module):
    def __init__(self,nin,nouts):
        sz = [nin] + nouts  
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]
        # 最后一层没有激活函数

    def __call__(self,x):
        for layer in self.layers:
            x = layer(x)
        return x
        
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"