import math
class Value:
    def __init__(self, data, _children = (), _op='', label=''):
        self.data = data
        self.grad = 0  
        self._backward = lambda:None 
        self._prev = set(_children)   # _prev指的是反向传播的前置节点，而children是前向传播的子节点 a+b=c  c的_children/_prev就是(a,b)
        self._op = _op
        self.label = label
    def __repr__(self):
        return f"Value(data={self.data})"
        
    def __add__(self,other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad +=1.0*out.grad
            other.grad += out.grad
        out._backward = _backward 
        return out

    def __radd__(self,other):
        return self+other
    
    def __mul__(self,other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward =  _backward
        return out

    def __rmul__(self,other):
        return self*other

    def __pow__(self,other):
        assert isinstance(other, (int,float))
        out = Value(self.data**other, (self,),f'**{other}')
        def _backward():
            self.grad += other * self.data**(other-1) * out.grad
        out._backward = _backward
        return out
        
    def __truediv__(self,other):
        return self*other**(-1)

    def __neg__(self):  # -self
        return self*(-1)

    def __sub__(self,other):
        return self + (-other)
    
    def __rsub__(self,other):
        return self + (-other)

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,),'exp')
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x)-1)/(math.exp(2*x)+1)
        out = Value(t, (self,), 'tanh')
        def _backward():
            self.grad += (1 - t**2)*out.grad
        out._backward = _backward
        return out
        
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for node in topo[::-1]:
            node._backward()

from graphviz import Digraph
def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root, format='svg', rankdir='LR'):
    """
    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    assert rankdir in ['LR', 'TB']
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir}) #, node_attr={'rankdir': 'TB'})
    
    for n in nodes:
        # print(n, str(id(n))) # Value(data=-6.0) 140638430115392
        # print(f"Node ID: {id(n)}, Label repr: {repr(n.label)}, Data: {n.data}")
        dot.node(name=str(id(n)), label = "{%s|data %.4f|grad %.4f}" % (n.label, n.data, n.grad), shape='record')
        # 创造了 * + 这样的伪节点，以及对应的连接
        if n._op:
            dot.node(name=str(id(n)) + n._op, label=n._op)
            dot.edge(str(id(n)) + n._op, str(id(n)))
    
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    
    return dot