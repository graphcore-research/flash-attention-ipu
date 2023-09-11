// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#ifndef FLASH_ATTENTION_QKV_PACKED_HPP
#define FLASH_ATTENTION_QKV_PACKED_HPP

#include <poplar/DebugContext.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>

struct AttentionOutputWithStash {
    poplar::Tensor output;
    poplar::Tensor logSumExp;
};

AttentionOutputWithStash flashAttentionQKVPackedWithStash(
    poplar::Graph& graph, 
    const poplar::Tensor& qkv,  // Shape 3 x G x L x D
    uint32_t num_chunks_q, 
    uint32_t num_chunks_kv,
    poplar::program::Sequence& prog,
    const poplar::DebugContext& dc);

poplar::Tensor flashAttentionQKVPacked(
    poplar::Graph& graph, 
    const poplar::Tensor& qkv,  // Shape 3 x G x L x D
    uint32_t num_chunks_q, 
    uint32_t num_chunks_kv,
    poplar::program::Sequence& prog,
    const poplar::DebugContext& dc);

poplar::Tensor flashAttentionQKVPackedGrad(
    poplar::Graph& graph,
    const poplar::Tensor& grad, // Shape G x L x D
    const poplar::Tensor& qkv, // Shape 3 x G x L x D
    uint32_t num_chunks_q,
    uint32_t num_chunks_kv,
    poplar::program::Sequence& prog,
    const poplar::DebugContext& dc);

#endif
