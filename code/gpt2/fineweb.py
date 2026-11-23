""" 
没有找到老师本身完全开源的内容

但是有类似相关的: 
nanochat这个项目, 
根据 https://github.com/karpathy/nanochat/commit/324e69c45d3606095adb6b409078647145165454#diff-ae4e743c6699e9b5468bf7476fbfe1e36764b01a8f1ba612f940864fa2a91270
最开始数据集用的是  FineWeb-EDU dataset ，不过目前已经更新数据集到 NVIDIA ClimbMix dataset

FineWeb-EDU dataset的数据集处理记录位于这个提交：
https://github.com/karpathy/nanochat/blob/6e9ef8f5656cf93073058227b6d4c75e8d941777/nanochat/dataset.py
结合这个，和老师的视频里可见部分的代码


🤣🤣🤣其实这部分视频课程对应的代码在: https://github.com/karpathy/build-nanogpt

https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
Downloads and tokenizes the data and saves data shards to disk.

# 运行方式
python code/gpt2/fineweb.py
HF_ENDPOINT=https://hf-mirror.com python code/gpt2/fineweb.py # 添加hf镜像源

Will save shards to the local directory "edu_fineweb10B"
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" #  在导入任何 huggingface 相关库之前，设置 HF_ENDPOINT 环境变量
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset  # pip install datasets
from tqdm import tdqm   # pip install tdqm


local_dir = "edu_fineweb10B"
remote_name = "sample-10BT"
shard_size = int(1e8)  # 100M tokens per shard, total of 100 shards

# create the cache the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# download the dataset
fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")

# init the tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']  # end of text token
def tokenize(doc):
    # tokenizers a single document and returns a numpy array of uint16 tokens
    tokens = [eot]  # the special <|endoftext|> token delimits(分割) all documents
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0<=tokens_np).all() and (tokens_np<2**16).all(), "token dictionary too large for uint16"
    # gpt2的字典在[0, 50256]之间，一共是50257个id  2**16 = 65536, 同时这里限制了数据类型是 uint16
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

def write_datafile(filename, tokens_np):
    # writes a numpy array of uint16 tokens to a binary file
    # with open(filename, 'wb') as f:
    #     f.write(tokens_np.tobytes())
    np.save(filename, tokens_np)


# tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
# 用多线程处理数据
nprocs = max(1, os.cpu_count()/2)
with mp.Pool(nprocs) as pool:
    shard_index = 0
    # preallocate buffer to hold current shard
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None
    for tokens in pool.imap(tokenize, fw, chunksize=16):
        # is there enough space in the current shard for the new tokens?
        if token_count + len(tokens) <shard_size:
            # simply append tokens to current shard
            all_tokens_np[token_count: token_count + len(tokens)] = tokens
            token_count+=len(tokens)
            # update progress bar
            if progress_bar is None:
                progress_bar = tdqm(total = shard_size, unit = "tokens", desc = f"Shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            # write the current shard and start a new one
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
            # split the document into whatever fits in this shard; the remainder goes to next one
            remainder = shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count: token_count + remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np)
            shard_index+=1
            progress_bar = None
            # populate the next shard with the leftovers of the current doc
            all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
            token_count = len(tokens)-remainder            
    # write any remaining tokens as the last shard
    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
        write_datafile(filename, all_tokens_np[:token_count])



