// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>
#include <poplar/DebugContext.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Fill.hpp>
#include <popops/Operation.hpp>
#include <popops/Reduce.hpp>
#include <popops/Zero.hpp>
#include <poplin/MatMul.hpp>

#include "vanilla_attention.hpp"

void triu(
    poplar::Graph& graph,
    const poplar::Tensor& t,
    const int32_t k,
    poplar::program::Sequence& prog,
    const poplar::DebugContext& dc) {

    assert(t.rank() >= 2);
    int m = t.dim(t.rank() - 2);
    int n = t.dim(t.rank() - 1);

    size_t start = 0;
    for (int i = m; i > 0 && i-1+k > 0; --i){
        size_t end = size_t(std::min(i-1+k, n));
        popops::zero(graph, t.slice({size_t(i-1), start}, {size_t(i), end}), prog, {dc, "triu_zero"});
    }
}

poplar::Tensor vanillaAttention(
    poplar::Graph& graph,
    const poplar::Tensor& qkv, // Shape 3 x G x L x D
    poplar::program::Sequence& prog,
    const poplar::DebugContext& dc) {

    assert(qkv.dim(0) == 3);

    // q, k, v = qkv
    auto query = qkv[0];
    auto key = qkv[1];
    auto value = qkv[2];

    // q @ k.T + mask
    auto attn = poplin::matMulGrouped(graph, query, key.dimShuffle({0, 2, 1}), prog, query.elementType(), {dc, "attn = Q@K.T"});
    
    // generate causal masks
    // clone attention matrix to colocate mask elements with attn matrix elements
    auto mask = graph.clone(attn.elementType(), attn[0], {dc, "mask = array_like(attn[0])"});
    popops::fill(graph, mask, prog, -10000.0, {dc, "fill(mask, -10000)"});
    triu(graph, mask, 1, prog, {dc, "triu(mask)"});
    popops::addInPlace(graph, attn, mask.expand({0}), prog, {dc, "attn += mask"});

    // Stable softmax(x)
    auto m = popops::reduce(graph, attn, attn.elementType(), {2}, {popops::Operation::MAX}, prog, {dc, "m = max(attn, dim=2)"});
    popops::subInPlace(graph, attn, m.expand({2}), prog, {dc, "attn -= m"});
    popops::expInPlace(graph, attn, prog, {dc, "attn = exp(attn)"});
    auto s = popops::reduce(graph, attn, attn.elementType(), {2}, {popops::Operation::ADD}, prog, {dc, "s = sum(attn, dim=2)"});
    popops::divInPlace(graph, attn, s.expand({2}), prog, {dc, "attn /= s"});

    // attn @ v
    auto out = poplin::matMulGrouped(graph, attn, value, prog, value.elementType(), {dc, "out = attn@V"});
    return out;
}

poplar::Tensor vanillaAttentionGrad(
    poplar::Graph& graph,
    const poplar::Tensor& grad, // shape G x L x D
    const poplar::Tensor& qkv, // shape 3 x G x L x D
    poplar::program::Sequence& prog,
    const poplar::DebugContext& dc) 
    {
    assert(qkv.dim(0) == 3);
    auto query = qkv[0];
    auto key = qkv[1];
    auto value = qkv[2];

    /* 
    Recompute attention 
    */

    // q @ k.T + mask
    auto attn = poplin::matMulGrouped(graph, query, key.dimShuffle({0, 2, 1}), prog, query.elementType(), {dc, "attn = Q@K.T"});
    
    // generate causal masks
    // clone attention matrix to colocate mask elements with attn matrix elements
    auto mask = graph.clone(attn.elementType(), attn[0], {dc, "mask = array_like(attn[0])"});
    popops::fill(graph, mask, prog, -10000.0, {dc, "fill(mask, -10000)"});
    triu(graph, mask, 1, prog, {dc, "triu(mask)"});
    popops::addInPlace(graph, attn, mask.expand({0}), prog, {dc, "attn += mask"});

    // Stable softmax(x)
    auto m = popops::reduce(graph, attn, attn.elementType(), {2}, {popops::Operation::MAX}, prog, {dc, "m = max(attn, dim=2)"});
    popops::subInPlace(graph, attn, m.expand({2}), prog, {dc, "attn -= m"});
    popops::expInPlace(graph, attn, prog, {dc, "attn = exp(attn)"});
    auto s = popops::reduce(graph, attn, attn.elementType(), {2}, {popops::Operation::ADD}, prog, {dc, "s = sum(attn, dim=2)"});
    popops::divInPlace(graph, attn, s.expand({2}), prog, {dc, "attn /= s"});

    /*
    Compute gradients
    */

    // grad_v
    auto value_grad = poplin::matMulGrouped(graph, attn.dimShuffle({0, 2, 1}), grad, prog, attn.elementType(), {dc, "dV = attn.T@dO"});
    auto attn_grad = poplin::matMulGrouped(graph, grad, value.dimShuffle({0, 2, 1}), prog, grad.elementType(), {dc, "dattn = dO@V.T"});
    
    // softmax grad
    popops::mulInPlace(graph, attn_grad, attn, prog, {dc, "dattn *= attn"});
    s = popops::reduce(graph, attn_grad, attn_grad.elementType(), {2}, {popops::Operation::ADD}, prog, {dc, "s = sum(attn_grad, dim=2)"});
    popops::mulInPlace(graph, attn, s.expand({2}), prog, {dc, "attn *= s"});
    popops::subInPlace(graph, attn_grad, attn, prog, {dc, "dattn -= attn"});

    // grad_q
    auto query_grad = poplin::matMulGrouped(graph, attn_grad, key, prog, attn_grad.elementType(), {dc, "dQ = dattn @ K"});
    // grad_k
    auto key_grad = poplin::matMulGrouped(graph, attn_grad.dimShuffle({0, 2, 1}), query, prog, attn_grad.elementType(), {dc, "dK = dattn.T @ Q"});

    // concat
    auto qkv_grad = poplar::concat({query_grad.expand({0}), key_grad.expand({0}), value_grad.expand({0})}, 0);
    
    return qkv_grad;
}
