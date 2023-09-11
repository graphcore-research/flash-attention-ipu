# FlashAttention (IPU)
*Poplar implementation of FlashAttention for IPU* 

**Quickstart**
```bash
# Tested on Poplar SDK 3.3.0+7857, Ubuntu 20.04, Python 3.8, torch 2.0.1
python -m pip install git+ssh://git@github.com/graphcore-research/flash-attention-ipu.git
```

**Usage**
```python
from flash_attention_ipu import flash_attention_qkv_packed

# For user-controlled chunking on IPU
class ChunkedAttention(torch.nn.Module):
  def __init__(self, num_chunks_q, num_chunks_kv):
    super().__init__()
    self.num_chunks_q = num_chunks_q
    self.num_chunks_kv = num_chunks_kv

  def forward(self, qkv):
    return flash_attention_qkv_packed(
      self.qkv.reshape(3, -1, *self.qkv.shape[-2:]),
      num_chunks_q=self.num_chunks_q,
      num_chunks_kv=self.num_chunks_kv
    )

# For automated chunking on IPU
import flash_attention_ipu.auto
import torch.nn.functional as F

# flash_attention_ipu.auto overrides F.scaled_dot_product_attention
class SDPAttention(torch.nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, q, k, v):
    return F.scaled_dot_product_attention(
      q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True
    )
```

**Development**
```bash
git clone git@github.com:graphcore-research/flash-attention-ipu.git
cd flash-attention-ipu
make

#Optional
./build/tests
```

**Background**

FlashAttention solves a key bottleneck of dot product attention on GPUs, namely the reading and writing of the attention matrix between HBM and L2 Cache. 

This becomes particularly problematic for training large models on long sequences as backpropagation requires either 1) storing the attention matrix for each layer, which can quickly exceed GPU maximum memory, or 2) recomputing the attention matrix, which dominates FLOPs when the sequence is long enough.

FlashAttention overcomes this bottleneck by chunking the query, key, and value tensors along the sequence dimension and computing the attention matrix in chunks using an online softmax algorithm. 

For small enough chunks, it is not necessary to read and write the attention matrix from HBM, and all intermediate tensors can fit in SRAM. As a result, this is both a memory-efficient and IO-efficient algorithm for computing dot-product attention.

**What relevance does this have for IPUs where the whole model is in SRAM?**

A Graphcore IPU chip has about 900 MB of SRAM split between 1472 tiles. Each tile can communicate with the others via an all-to-all exchange. This all-to-all exchange makes it possible for operations such as large matrix multiplications to get close to [peak FLOPs](https://github.com/graphcore-research/tessellate-ipu/blob/main/notebooks/IPU%20Peak%20Flops.ipynb) for an IPU (around 350 TFLOPs).

Assuming every tile is being used for computation when the entire model fits in IPU SRAM, performance is limited by how much data needs to be exchanged across tiles. As such, a good FlashAttention implementation for the IPU  minimises both memory usage and data exchange across tiles.

This initial attempt aims to keep memory consumption low using dynamic slicing and outlined graphs. We also aim to keep exchange reasonably small using off-the-shelf tile mappings of tensors. We leave more customised tile mappings and further improvements to memory usage in future releases.
