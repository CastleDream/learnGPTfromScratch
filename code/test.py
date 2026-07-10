""" 
测试Value的包装结果是否和torch一致
"""
import torch
from micrograd.engine import Value

def test_sanity_check():
    """完整性检查
    """
    x = Value(-4.0)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xmg, ymg = x, y

    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xpt, ypt = x, y

    # forward pass went well
    assert ymg.data == ypt.data.item()
    # backward pass went well
    assert xmg.grad == xpt.grad.item()


def test_more_ops():

    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backward()
    amg, bmg, gmg = a, b, g

    a = torch.Tensor([-4.0]).double()
    b = torch.Tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()
    apt, bpt, gpt = a, b, g

    tol = 1e-6
    # forward pass went well
    assert abs(gmg.data - gpt.data.item()) < tol
    # backward pass went well
    assert abs(amg.grad - apt.grad.item()) < tol
    assert abs(bmg.grad - bpt.grad.item()) < tol

import torch
import torchvision.models as models
import torch.nn as nn

def count_bn_layers():
    # 1. 加载 ResNet50 网络结构 (weights=None 表示不下载/加载预训练权重，仅初始化网络结构)
    model = models.resnet50(weights=None)
    print(model)
    
    # 2. 全局统计：遍历所有子模块，统计 BN 和 Conv2d 的数量
    total_bn = 0
    total_conv = 0
    
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            total_bn += 1
        elif isinstance(m, nn.Conv2d):
            total_conv += 1
            
    print("="*40)
    print(f"ResNet50 总 BN 层数量: {total_bn}")
    print(f"ResNet50 总 Conv2d 数量: {total_conv}")
    print("="*40)
    
    # 3. 局部统计：按网络阶段拆解统计 BN 层，验证理论推导
    print("\n--- 各阶段 BN 层详细分布 ---")
    
    # 统计初始的 conv1
    conv1_bn = sum(1 for m in model.bn1.modules() if isinstance(m, nn.BatchNorm2d))
    print(f"Conv1 (初始层): {conv1_bn} 个 BN")
    
    # 统计 layer1 到 layer4 (对应论文中的 Conv2_x 到 Conv5_x)
    layer_bn_counts = []
    for i in range(1, 5):
        layer = getattr(model, f'layer{i}')
        layer_bn = sum(1 for m in layer.modules() if isinstance(m, nn.BatchNorm2d))
        layer_bn_counts.append(layer_bn)
        print(f"Layer{i} (Conv{i+1}_x 阶段): {layer_bn} 个 BN")
        
    # 计算总和
    total_bn_detailed = conv1_bn + sum(layer_bn_counts)
    print("-" * 40)
    print(f"各阶段 BN 数量相加总和: {conv1_bn} + {sum(layer_bn_counts)} = {total_bn_detailed}")


# python code/test.py
if __name__ == "__main__":
    # test_sanity_check()
    # test_more_ops()
    count_bn_layers()
