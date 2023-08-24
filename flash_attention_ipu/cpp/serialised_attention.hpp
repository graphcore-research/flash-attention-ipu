#ifndef SERIALISED_ATTENTION_HPP
#define SERIALISED_ATTENTION_HPP

#include <poplar/DebugContext.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>

std::vector<poplar::Tensor> serialisedAttentionImpl(
    poplar::Graph& graph, 
    const poplar::Tensor& qkv,  // Shape 3 x G x L x D
    uint32_t num_chunks_q, 
    uint32_t num_chunks_kv,
    poplar::program::Sequence& prog,
    const poplar::DebugContext& dc);

poplar::Tensor serialisedAttention(
    poplar::Graph& graph, 
    const poplar::Tensor& qkv,  // Shape 3 x G x L x D
    uint32_t num_chunks_q, 
    uint32_t num_chunks_kv,
    poplar::program::Sequence& prog,
    const poplar::DebugContext& dc);

poplar::Tensor serialisedAttentionGrad(
    poplar::Graph& graph,
    const poplar::Tensor& grad, // Shape G x L x D
    const poplar::Tensor& qkv, // Shape 3 x G x L x D
    uint32_t num_chunks_q,
    uint32_t num_chunks_kv,
    poplar::program::Sequence& prog,
    const poplar::DebugContext& dc);

#endif
