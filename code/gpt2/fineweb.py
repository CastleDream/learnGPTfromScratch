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
from tqdm import tqdm   # pip install tqdm


local_dir = "edu_fineweb10B"
remote_name = "sample-10BT"  # 
shard_size = int(1e8)  # 100M tokens per shard, total of 100 shards

# create the cache the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# download the dataset
fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train", cache_dir="/home/dev/huangS/cache")
# 注意，这个执行的话，其实主要是个下载 + 加载的过程，由于网络原因，这里建议直接科学上网下载,然后修改data_dir即可
# 只流式加载该子集的前 1000 条数据用于调试代码逻辑
# fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train[:1000]", streaming=True)

# init the tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']  # end of text token
def tokenize(doc):
    # tokenizers a single document and returns a numpy array of uint16 tokens
    # '<|endoftext|>' 这个token就是gpt2分词器额外添加的那个 50256 ID的内容，
    # 虽然称为 文本结束符号 但是实际上，会被放在每个文档的开头，作为划分每个文档开端的标记
    # 即：每个文档的开头其实都是 50256 这个ID
    tokens = [eot]  # the special <|endoftext|> token delimits(分割) all documents
    tokens.extend(enc.encode_ordinary(doc["text"]))
    # encode_ordinary： Encodes a string into tokens, ignoring special tokens. 
    #                   This is equivalent to encode(text, disallowed_special=()) (but slightly faster).
    tokens_np = np.array(tokens)
    assert (0<=tokens_np).all() and (tokens_np<2**16).all(), "token dictionary too large for uint16"
    # gpt2的字典在[0, 50256]之间，一共是50257个id  2**16 -1 = 65535, 远超过50257，所以可以用 uint16来存储这些数字，节省一些存储空间
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

def write_datafile(filename, tokens_np):
    # writes a numpy array of uint16 tokens to a binary file
    # with open(filename, 'wb') as f:
    #     f.write(tokens_np.tobytes())
    np.save(filename, tokens_np)
    # Save an array to a binary file in NumPy .npy format.


# tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
# 用多线程处理数据
nprocs = max(1, os.cpu_count()//2) # 用一半 CPU 核做分词，剩下一半留给主进程写盘
print(f"系统包含的逻辑cpu个数是: {os.cpu_count()}, 并发用的线程数是: {nprocs}")
with mp.Pool(nprocs) as pool:
    shard_index = 0
    # preallocate buffer to hold current shard
    # shard_size = int(1e8)  # 100M tokens per shard, total of 100 shards 每个切片放1000w tokens
    # 预先分配一个缓存来装当前1000w的tokens
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)  # all_tokens_np.shape (shard_size,)
    token_count = 0
    progress_bar = None
    # 一共是10B tokens = 100亿 = 10**21 则chunk_size = 16, 每16个文档被打包成一个任务
    # chunksize 对主进程的 for 循环是完全透明的。imap 严格保证一一映射（1-to-1 mapping）
    # 这里虽然chunksize=16，但是对于上层的for循环，每次还是读到一个doc的处理结果，而每个doc处理后的tokens数量是不固定的
    for tokens in pool.imap(tokenize, fw, chunksize=16):
        # imap 按原始顺序返回结果（这对 token 连续性至关重要），chunksize=16 表示一次打包 16 篇文档发过去，减少进程间通信次数。
        # (method) def imap(
        #     func: (_S@imap) -> _T@imap,
        #     iterable: Iterable[_S@imap],
        #     chunksize: int | None = 1
        # ) -> IMapIterator[_T@imap]
        # 对 fw 这个迭代对象，每次迭代16个，然后对于每个对象，用tokenize函数处理
        # map()是返回list (完整列表)， imap是返回iterator (迭代器)
        # is there enough space in the current shard for the new tokens?
        # 现有的token数量是 token_count, 新从fw里解析的token是tokens，如果还有多余的空间，则直接赋值
        # 说明还可以继续向当前切片写入
        if token_count + len(tokens) <shard_size:
            # simply append tokens to current shard
            all_tokens_np[token_count: token_count + len(tokens)] = tokens
            token_count+=len(tokens)
            # update progress bar
            if progress_bar is None:
                progress_bar = tqdm(total = shard_size, unit = "tokens", desc = f"Shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            # 否则，说明当前切片已经满了，够 shard_size，需要落盘，开始写下一个shard了
            # write the current shard and start a new one
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
            # 06d 表示补零到 6 位，如 ..._000000、..._000001。
            # split the document into whatever fits in this shard; the remainder goes to next one
            # 由于当前for循环的tokens(某个文档解析出的tokens)数量并不一定规律，所以可能会导致刚好出现 10**8 - 800这种情况，
            # 比如 当前 token_count = 9999 9995， 然后当前len(tokens)=800, 则当前这个shards还差5个就满了，所以要补上缺的这几个，即：remainder
            remainder = shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count: token_count + remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np)

            # 准备下一个shard，更新token_count，满足if token_count + len(tokens) <shard_size:即可
            shard_index+=1
            progress_bar = None
            # populate the next shard with the leftovers of the current doc
            # 上面那个shard已经写过了，所以all_tokens_np这个作为临时缓存区的就可以直接覆盖了
            all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
            token_count = len(tokens)-remainder            
    # write any remaining tokens as the last shard
    # for循环结束了，也就是所有的tokens都遍历完了，即最后一个for循环执行了， token_count = len(tokens)-remainder这部分导致的 token_count!=0
    # 当处理完最后一个文档时，当前正在填充的 Shard 通常“没有满”，而循环已经结束了，没有新的文档来触发“写满并保存”的逻辑。所以必须在循环外，手动把这最后剩下的一部分数据保存下来。
    # 这部分其实就只有: [0:token_count]这一小部分
    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
        write_datafile(filename, all_tokens_np[:token_count])


# 执行后，一共得到100个shards， 10B的数据 每个shard 1亿，所以就是100个左右
