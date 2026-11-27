"""
Downloads and evaluates HellaSwag in Python.
https://github.com/rowanz/hellaswag

Example HellaSwag json item:

{"ind": 24, "activity_label": "Roof shingle removal", "ctx_a": "A man is sitting on a roof.", "ctx_b": "he", "ctx": "A man is sitting on a roof. he", "split": "val", "split_type": "indomain", "label": 3, "endings": ["is using wrap to wrap a pair of skis.", "is ripping level tiles off.", "is holding a rubik's cube.", "starts pulling up roofing on a roof."], "source_id": "activitynet~v_-JhWjGDPHMY"}

ind: dataset ID
activity_label: The ActivityNet or WikiHow label for this example
context: There are two formats. The full context is in ctx. When the context ends in an (incomplete) noun phrase, like for ActivityNet, this incomplete noun phrase is in ctx_b, and the context up until then is in ctx_a. This can be useful for models such as BERT that need the last sentence to be complete. However, it's never required. If ctx_b is nonempty, then ctx is the same thing as ctx_a, followed by a space, then ctx_b.
endings: a list of 4 endings. The correct index is given by label (0,1,2, or 3)
split: train, val, or test.
split_type: indomain if the activity label is seen during training, else zeroshot
source_id: Which video or WikiHow article this example came from

gpt2 (124M)
- eleuther harness reports acc 28.92%, acc_norm 31.14% (multiple choice style)
- this script: 10042 acc: 0.2859 acc_norm: 0.2955 (completion style)

我自己跑的结果(差不多, 可能是windows和linux的运行差异吧):
10042 acc_norm: 2968/10042=0.2956
注意: 随机猜测的准确率时25%, 所以原始的gpt2和随机猜测的效果差不多~

gpt2-xl (1558M)
- eleuther harness reports acc 40.04%, acc_norm 50.89% (multiple choice style)
- this script: 10042 acc: 0.3842 acc_norm: 0.4893 (completion style)

The validation set of HellaSwag has a total of 10,042 examples.

name	train	validation	test
default	39905	10042	10003
"""

import os
import json
import requests
import tiktoken
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F


# 老师写这个脚本的时候，可能还没有上传到Huggingface上，
# <https://huggingface.co/datasets/Rowan/hellaswag/tree/main/data>
# 不过老师给的github链接现在已经失效了，所以现在要换成用Huggingface的dataset来下载读取数据集了

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from transformers import GPT2LMHeadModel
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "hellaswag")
os.makedirs(DATA_CACHE_DIR, exist_ok=True)
import os
from datasets import load_dataset, Dataset

def load_hellaswag_splits():
    dataset_name = "Rowan/hellaswag"
    target_splits = ["train", "test", "validation"]
    loaded_data = {}
    for split in target_splits:
        dataset = load_dataset(dataset_name, split=split, cache_dir = DATA_CACHE_DIR)
    loaded_data[split] = dataset
    return loaded_data

def render_example(example):
    """
    Given the example as a dictionary, render it as three torch tensors:
    - tokens (the tokens of context + completion, of size 4xN, as there are always 4 candidates/choices)
    - mask (is 1 in the region of the candidate completion, where we evaluate likelihoods)
    - label (the index of the correct completion, which we hope has the highest likelihood)
    """
    enc = tiktoken.get_encoding("gpt2")

    ctx = example["ctx"]
    label = int(example["label"]) 
    # 注意，用dataset加载的这个数据 label在huggingface上显示的数据类型是string，所以这里要变成整数
    # 不然下面在计算判断正确时 num_correct_norm += int(pred_norm == label)
    # In [2]: 2=='2'
    # Out[2]: False
    endings = example["endings"]

    # data needed to reproduce this eval on the C size
    data = {
        "label": label,
        "ctx_tokens": None,
        "ending_tokens": [],
    }

    # gather up all the tokens 构建完整的prompt tokens 包括 ctx上下文 和 选项
    ctx_tokens = enc.encode(ctx)
    data["ctx_tokens"] = ctx_tokens  # 这是上下文部分
    # 由于有四个选项，每个长度不同，因此，以token最长的作为sequnce_length, 其余不够长度的需要填充，为了区分填充，所以需要一个mask
    # 最后其实是为了形成一个(batch_size, sequnce_length)的prompt
    tok_rows = []  # token_rows 就是4个选项对应的tokens的列表
    mask_rows = []
    for end in endings:
        end_tokens = enc.encode(" " + end) 
        # note: prepending " " because GPT-2 tokenizer 由于gpt2的分词特性，因此需要在每个句子的开头加上一个空格
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0]*len(ctx_tokens) + [1]*len(end_tokens)) 
        # 所以 mask_rows里的每个元素都是 [0,0,0,1,1,1]这样的,我们只需要1那个部分，即: choices部分
        data["ending_tokens"].append(end_tokens)

    # have to be careful during the collation because the number of tokens in each row can differ
    # 从列表改成torch.tensor格式
    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)
    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)

    return data, tokens, mask, label

@torch.no_grad()
def evaluate(model_type, device):

    torch.set_float32_matmul_precision('high') # use tf32
    model = GPT2LMHeadModel.from_pretrained(model_type, cache_dir = DATA_CACHE_DIR)  # 可以从Huggingface里加载现有的gpt2来测试一下
    model.to(device)
    # model = torch.compile(model) # optionally torch compile the model

    num_correct_norm = 0
    num_correct = 0
    num_total = 0

    dataset = load_dataset("Rowan/hellaswag", split="validation", cache_dir = DATA_CACHE_DIR)
    for example in dataset:
        data, tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)

        logits = model(tokens).logits  # (B,T,vocab_size)

        # evaluate the autoregressive loss at all positions
        # logits   (B,T,vocab_size)
        # shift_logits  (B, T-1,vocab_size) 这里不是只预测tokens的最后一个token的next token了，而是关注所有前 len(tokens)-1 个token的next token了
        # 这里去掉 -1是因为，最后一个token没有真实label对应 最后一个词 没有下一个词
        # x = data[:-1]
        # y = data[1:] 
        shift_logits = (logits[..., :-1, :]).contiguous()   # 用 :-1 切片了，所以变量的内存不连续了，所以用contiguous处理一下 会返回一个新tensor(内存也是新的)
        shift_tokens = (tokens[..., 1:]).contiguous() # 真实的 label（去掉第一个，因为第一个是 prompt/起始符，不需要被预测）
        # shift_logits.shape (4, T-1,vocab_size)
        # shift_tokens.shape (4, T-1)
        flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1)) # view操作要求tensor必须是内存连续的，所以上面用了contiguous
        flat_shift_tokens = shift_tokens.view(-1)
        # flat_shift_logits (4*(T-1), vocab_size)
        # flat_shift_tokens (4*(T-1))
        shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')  # input:(N,C)  target:(N)
        # https://docs.pytorch.org/docs/2.15/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss
        #  reduction='none'的时候，返回的损失的shape就是 (B/N) batch_size
        # 设置 reduction='none' 时，返回的 loss tensor 的 shape 与 target（标签）的 shape 完全一致。
        shift_losses = shift_losses.view(tokens.size(0), -1)
        # 这里又把 shift_losses从flaten(4*(T-1))状态变成 (4, T-1)的batch形状了

        # now get the average loss just for the completion region (where mask == 1), in each row
        # 只对mask有效的部分计算平均损失
        shift_mask = (mask[..., 1:]).contiguous()  # shift_mask.shape(4, T-1)
        # 这里 mask[..., 1:] 是为了和 shift_tokens 对齐

        # 对齐之后，之后乘法，shift_losses对应的shift_mask里是0的地方也就变成0了，只保留了shift_losses对应的shift_mask里是1的地方
        masked_shift_losses = shift_losses * shift_mask
        # sum and divide by the number of 1s in the mask
        sum_loss = masked_shift_losses.sum(dim=1)    # 不考虑reponse长度的损失，即：没有归一化的结果
        avg_loss = sum_loss / shift_mask.sum(dim=1)  # 考虑了response长度的损失，即：归一化之后的结果（对batch中的每个样本求和，即求1的个数）
        # now we have a loss for each of the 4 completions
        # the one with the lowest loss should be the most likely
        # 损失最低的就是可能性最高的，就是选择的答案
        # 注意，这里虽然是batch=4,但是这4个样本是同一个问题的四个答案，所以其实只对应最后一个true label(ABCD序号)

        # 下面其实只使用了归一化之后的结果，所以这个没有归一化的可以不看了
        # pred = sum_loss.argmin().item()
        pred_norm = avg_loss.argmin().item()

        # accumulate stats 计算统计结果
        num_total += 1
        # num_correct += int(pred == label)
        num_correct_norm += int(pred_norm == label)
        print(f"{num_total} acc_norm: {num_correct_norm}/{num_total}={num_correct_norm/num_total:.4f}")

        # debug: pretty print a few examples, and the losses in each case
        # 打印前10个样本，看看效果
        if num_total < 10:
            print("---")
            print(f"Context:\n {example['ctx']}")
            print(f"Endings:")
            for i, end in enumerate(example["endings"]):
                print(f"{i} (loss: {avg_loss[i].item():.4f}) {end}")
            print(f"predicted: {pred_norm}, actual: {label}")
 

def get_most_likely_row(tokens, mask, logits):
    """
    """
    shift_logits = (logits[..., :-1, :]).contiguous()   
    shift_tokens = (tokens[..., 1:]).contiguous() 

    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1)) 
    flat_shift_tokens = shift_tokens.view(-1) 

    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none') 
    shift_losses = shift_losses.view(tokens.size(0), -1)

    shift_mask = (mask[..., 1:]).contiguous()
    masked_shift_losses = shift_losses * shift_mask

    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)  
    pred_norm = avg_loss.argmin().item()
    return pred_norm

# python code/gpt2/hellaswag.py 
# 默认gpt-2的显存消耗是 2.3G/8G  毕竟只有124MB，又不是 641B的1B对1GB显存
if __name__ == "__main__":
    # loaded_data = load_hellaswag_splits()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_type", type=str, default="gpt2", help="the model type to use")
    parser.add_argument("-d", "--device", type=str, default="cuda", help="the device to use")
    args = parser.parse_args()
    evaluate(args.model_type, args.device)