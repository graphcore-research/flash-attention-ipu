// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#ifndef VANILLA_ATTENTION_HPP
#define VANILLA_ATTENTION_HPP

#include <poplar/DebugContext.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>

poplar::Tensor vanillaAttention(
    poplar::Graph& graph,
    const poplar::Tensor& qkv, // Shape 3 x G x L x D
    poplar::program::Sequence& prog,
    const poplar::DebugContext& dc);

poplar::Tensor vanillaAttentionGrad(
    poplar::Graph& graph,
    const poplar::Tensor& grad, // shape G x L x D
    const poplar::Tensor& qkv, // shape 3 x G x L x D
    poplar::program::Sequence& prog,
    const poplar::DebugContext& dc);

#endif
