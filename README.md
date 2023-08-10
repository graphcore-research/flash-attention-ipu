# FlashAttention (IPU)
*Exchange-based FlashAttention for IPU* 

**Quickstart**
```bash
git clone git@github.com:graphcore-research/flash-attention-ipu.git
cd flash-attention-ipu
ninja

#Optional
./build/tests
```


**Background**

FlashAttention solves a key bottleneck of dot product attention on GPUs, namely the reading and writing of the attention matrix between HBM and SRAM. 

This becomes particularly problematic for training large models on long sequences as backpropagation requires either 1) storing the attention matrix for each layer, which can quickly exceed GPU maximum memory, or 2) recomputing the attention matrix, which dominates FLOPs when the sequence is long enough.

FlashAttention overcomes this bottleneck by chunking the query, key, and value tensors along the sequence dimension and computing the attention matrix in chunks using an online softmax algorithm. 

For small enough chunks, it is not necessary to read and write the attention matrix from HBM, and all intermediate tensors can fit in SRAM. As a result, this is both a memory-efficient and IO-efficient algorithm for computing dot-product attention.

**What relevance does this have for IPUs where the whole model is in SRAM?**

A Graphcore IPU chip has about 900 MB of SRAM split between 1472 tiles. Each tile can communicate with the others via an all-to-all exchange. This all-to-all exchange makes it possible for operations such as large matrix multiplications to get close to [peak FLOPs](https://github.com/graphcore-research/tessellate-ipu/blob/main/notebooks/IPU%20Peak%20Flops.ipynb) (around 350 TFLOPs).

Assuming every tile is being used for computation, when the entire model fits in IPU SRAM it is mainly bound by exchange bandwidth (11 TB/s). As such, a good FlashAttention implementation for the IPU should do the job of minimising both memory and exchange.

Here is an initial attempt.