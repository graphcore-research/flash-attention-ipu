#include <poplar/Graph.hpp>
#include <poplar/DeviceManager.hpp>
#include <poplar/Tensor.hpp>
#include <poplar/DebugContext.hpp>
#include <poplar/OptionFlags.hpp>
#include <popops/Cast.hpp>
#include <popops/DynamicSlice.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Fill.hpp>
#include <popops/Operation.hpp>
#include <popops/Reduce.hpp>
#include <popops/Zero.hpp>
#include <poplin/Convolution.hpp>
#include <poplin/MatMul.hpp>
#include <popnn/NonLinearity.hpp>
#include <poputil/Broadcast.hpp>

#include <poputil/TileMapping.hpp>
#include <poplar/SyncType.hpp>

#include <poplin/codelets.hpp>
#include <popops/codelets.hpp>
#include <popnn/codelets.hpp>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <popart/op.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/popx/opx.hpp>
#include <popart/popx/opxmanager.hpp>
#pragma GCC diagnostic pop

#include <iostream>
#include <algorithm>
#include <vector>

#include "serialised_attention.hpp"

using namespace poplar;
using namespace poplar::program;
namespace pe = popops::expr;

std::vector<int32_t> getTriuOffsetSequence(
    uint32_t numRows,
    uint32_t numCols
) {

    /* 
    A utility function for generating triu offsets needed for causal masks

    Rather than generate full [seqLen x seqLen] causal mask, generate only the causal mask blocks that you need.
    When blocks are square, block intersect with diagonal with the same offset every time they intersect
    
    Example: a 4 x 4 upper triangular matrix split into 4 blocks of 2 x 2:

    1 1 | 1 1
    0 1 | 1 1
    ---------
    0 0 | 1 1
    0 0 | 0 1

    Notice that blocks along the diagonal have the same pattern. As a result, you only need to generate this upper triangular
    block and use it every time

    When blocks are non-square, blocks can intersect with the diagonal with a different offset.

    Example: a 6 x 6 upper triangular matrix split into 6 blocks of 2 x 3:

    1 1 1 | 1 1 1
    0 1 1 | 1 1 1
    -------------
    0 0 1 | 1 1 1
    0 0 0 | 1 1 1
    -------------
    0 0 0 | 0 1 1
    0 0 0 | 0 0 1

    Notice now that each block that intersect with the main diagonal has a different pattern. As a result, you need to generate
    each of these blocks and use them in the order they are encountered. 

    When blocks are non-square but the lowest common multiple of their dimensions are less than the dimension of the full square matrix,
    a repeating pattern occurs.

    Example: a 8 x 8 upper triangular matrix split into 8 blocks of 2 x 4:

    1 1 1 1 | 1 1 1 1
    0 1 1 1 | 1 1 1 1
    -------------
    0 0 1 1 | 1 1 1 1
    0 0 0 1 | 1 1 1 1
    -------------
    0 0 0 0 | 1 1 1 1
    0 0 0 0 | 0 1 1 1
    -------------
    0 0 0 0 | 0 0 1 1
    0 0 0 0 | 0 0 0 1

    Notice now that the two upper left blocks have the same pattern as the two lower right blocks. As a result, you only need
    to generate two blocks and reuse these on each cycle through the block diagonal.
    */

    std::vector<int32_t> offsets = {1};
    int tmp_offset = 1;
    int max_offset = numCols - 1;
    int min_offset = 2 - numRows;

    while(true) {
        tmp_offset += numRows;
        if (tmp_offset > max_offset){
            tmp_offset -= (numRows + numCols);
        }
        if (tmp_offset == 1){
            break;
        }
        else {
            if (tmp_offset >= min_offset){
                offsets.push_back(tmp_offset);
            }
        }
    }
    return offsets;
}

void dynamicAddMask(
    poplar::Graph& graph,
    const poplar::Tensor& t,
    const poplar::Tensor& masks,
    const poplar::Tensor& maskCounter,
    poplar::program::Sequence& prog,
    const poplar::DebugContext& dc) {
    
    auto blockMask = popops::dynamicSlice(graph, masks, maskCounter, {0}, {1}, prog, {dc, "get_mask"}).squeeze({0});
    popops::addInPlace(graph, t, blockMask.expand({0}), prog, {dc, "attn_ij += mask_ij"});
    // update mask counter
    popops::mapInPlace(graph, ((pe::_1 + 1)%uint(masks.dim(0))), {maskCounter}, prog, {dc, "k = (k+1)%masks.size()"});
    }

std::vector<poplar::Tensor> serialisedAttentionImpl(
    poplar::Graph& graph, 
    const poplar::Tensor& qkv,  // Shape 3 x G x L x D
    uint32_t num_chunks_q, 
    uint32_t num_chunks_kv,
    poplar::program::Sequence& prog,
    const poplar::DebugContext& dc) {

    assert(qkv.dim(0) == 3);

    // get shape data
    auto groups = qkv.dim(1); // groups = batch_size * num_heads
    auto seqLen = qkv.dim(2);
    auto headDim = qkv.dim(3);
    
    assert(seqLen % num_chunks_q == 0);
    assert(seqLen % num_chunks_kv == 0);

    // compute sizes of chunked sequence length
    auto chunkedQueryLen = seqLen / num_chunks_q; 
    auto chunkedKVLen = seqLen / num_chunks_kv; 

    // Unpack q,k,v and copy data to sliceable tensors with nice tile mappings
    auto query = popops::createSliceableTensor(graph, qkv.elementType(), {num_chunks_q, groups, chunkedQueryLen, headDim}, {0}, {1}, 4, {dc, "create_query"});
    auto key = popops::createSliceableTensor(graph, qkv.elementType(), {num_chunks_kv, groups, chunkedKVLen, headDim}, {0}, {1}, 4, {dc, "create_key"});
    auto value = popops::createSliceableTensor(graph, qkv.elementType(), {num_chunks_kv, groups, chunkedKVLen, headDim}, {0}, {1}, 4, {dc, "create_value"});

    prog.add(Copy(qkv[0].reshape({groups, num_chunks_q, chunkedQueryLen, headDim}).dimShuffle({1, 0, 2, 3}), query));
    prog.add(Copy(qkv[1].reshape({groups, num_chunks_kv, chunkedKVLen, headDim}).dimShuffle({1, 0, 2, 3}), key));
    prog.add(Copy(qkv[2].reshape({groups, num_chunks_kv, chunkedKVLen, headDim}).dimShuffle({1, 0, 2, 3}), value));

    // create output tensor computed iteratively
    auto out = graph.clone(query, {dc, "create_output"});
    popops::zero(graph, out, prog, {dc, "zero_output"});

    // create tensors to store running logsumexp
    auto logSumExp = popops::createSliceableTensor(graph, query.elementType(), {num_chunks_q, groups, chunkedQueryLen}, {0}, {1}, 4, "create_logsumexp_store");
    popops::zero(graph, logSumExp, prog, {dc, "zero_logsumexp"});

    // outer loop counter on q read
    auto qCounter = graph.addVariable(poplar::UNSIGNED_INT, {1}, {dc, "init_qCounter(i=0)"});
    // inner loop counter on kv read
    auto kvCounter = graph.addVariable(poplar::UNSIGNED_INT, {1}, {dc, "init_kvCounter(j=0)"});
    
    // gimme tiles
    poputil::mapTensorLinearly(graph, qCounter);
    poputil::mapTensorLinearly(graph, kvCounter);

    popops::zero(graph, qCounter, prog, {dc, "zero_qCounter"});
    popops::zero(graph, kvCounter, prog, {dc, "zero_kvCounter"});

    // Setup repeat loops. Use whitespace indentation for python-like readability

    // kv loop body program
    Sequence qLoopProg; {
        // slice k and v
        auto qi = popops::dynamicSlice(graph, query, qCounter, {0}, {1}, qLoopProg, {dc, "q_i = q.at[i].get()"}).squeeze({0}); 
        auto oi = popops::dynamicSlice(graph, out, qCounter, {0}, {1}, qLoopProg, {dc, "o_i = o.at[i].get()"}).squeeze({0});

        auto li = popops::dynamicSlice(graph, logSumExp, qCounter, {0}, {1}, qLoopProg, {dc, "l_i = l.at[i].get()"}).squeeze({0});
        auto mi = graph.clone(li, {dc, "m_i = array_like(l_i)"});
        popops::fill(graph, mi, qLoopProg, -10000, {dc, "fill(m_i, -10000)"});

        // q loop body program
        Sequence kvLoopProg; {
            
            // Condition for executing (true) or skipping (false) block
            auto doBlock = popops::map(graph, ((pe::_1 + 1) * uint(chunkedQueryLen)) > (pe::_2 * uint(chunkedKVLen)), {qCounter, kvCounter}, kvLoopProg, {dc, "(i+1) * q_chunk_size > j * kv_chunk_size"})[0];
            
            // Conditional execute block program body
            Sequence doBlockProg;

            // slice q, output, and softmax running stats
            auto kj = popops::dynamicSlice(graph, key, kvCounter, {0}, {1}, doBlockProg, {dc, "k_j = k.at[j].get()"}).squeeze({0});
            auto vj = popops::dynamicSlice(graph, value, kvCounter, {0}, {1}, doBlockProg, {dc, "v_j = v.at[j].get()"}).squeeze({0});

            // compute qk^t
            auto t = poplin::matMulGrouped(graph, qi, kj.dimShuffle({0, 2, 1}), doBlockProg, kj.elementType(), {dc, "attn_ij = q_i @ k_j.T"});
            
            // Condition for making mask
            auto doMakeMasks = popops::map(graph, (pe::_1 == 0) && (pe::_2 == 0), {qCounter, kvCounter}, doBlockProg, {dc, "i==0 and j==0"})[0];
            
            // generate causal masks
            // clone attention matrix block to colocate mask block elements with attn matrix block elements
            Sequence doMakeMasksProg;
            std::vector<int32_t> offsets = getTriuOffsetSequence(chunkedQueryLen, chunkedKVLen);
            // use cloneN to generate as many as needed by offsets.size()
            auto masks = graph.cloneN(t.elementType(), t[0], offsets.size(), {dc, "masks = repeat(array_like(t[0]), offsets.size())"});
            popops::fill(graph, masks, doMakeMasksProg, -10000.0, {dc, "fill(masks, -10000.0)"});
            // Triu function above didn't work when passing a slice of a tensor, so it is inlined here
            for (size_t i = 0; i < offsets.size(); ++i){
                int k = offsets[i];
                int m = masks.dim(masks.rank() - 2);
                int n = masks.dim(masks.rank() - 1);

                size_t start = 0;
                for (int j = m; j > 0 && j-1+k > 0; --j){
                    size_t end = size_t(std::min(j-1+k, n));
                    popops::zero(graph, masks.slice({i, size_t(j-1), start}, {i+1, size_t(j), end}), doMakeMasksProg, {dc, "zero_for_triu(masks)"});
                    }
            }
            // mask counter on masked block execution
            auto maskCounter = graph.addVariable(poplar::UNSIGNED_INT, {1}, {dc, "init_maskCounter(k=0)"});
            poputil::mapTensorLinearly(graph, maskCounter);
            popops::zero(graph, maskCounter, doMakeMasksProg, {dc, "zero_maskCounter"});

            Sequence skipMakeMasksProg;
            doBlockProg.add(If(doMakeMasks, doMakeMasksProg, skipMakeMasksProg, {dc, "initialise_masks"}));

            // Condition for adding mask to q@k.T
            auto doMask = popops::map(graph, (pe::_1 * uint(chunkedQueryLen) < ((pe::_2 + 1) * uint(chunkedKVLen) - 1)), {qCounter, kvCounter}, doBlockProg, {dc, "i * q_chunk_size < (j+1) * kv_chunk_size - 1"})[0];
            
            // Conditional add mask program body
            Sequence doMaskProg;
            dynamicAddMask(graph, t, masks, maskCounter, doMaskProg, {dc, "add_mask"}); 

            // Empty skip mask program
            Sequence skipMaskProg;

            // Add conditional mask program to execute block program 
            doBlockProg.add(If(doMask, doMaskProg, skipMaskProg, {dc, "q@k.T + mask if i==j else q@k.T"}));

            // compute qk^T max for stable softmax
            auto newMaxs = popops::reduce(graph, t, t.elementType(), {2}, {popops::Operation::MAX}, doBlockProg, {dc, "m_tmp = sum(attn_ij, dim=2)"});
            newMaxs = popops::max(graph, mi, newMaxs, doBlockProg, {dc, "m_new = max(m_i, m_tmp)"});
            auto c = popops::map(graph, pe::Exp(pe::_1 - pe::_2), {mi, newMaxs}, doBlockProg, {dc, "c = exp(m_i - m_new)"});
            doBlockProg.add(Copy(newMaxs, mi));
            
            // subtract max from qk^T
            popops::subInPlace(graph, t, newMaxs.expand({2}), doBlockProg, {dc, "attn_ij -= m_tmp"});
            // compute softmax numerator: exp(qk^T - max)
            popops::expInPlace(graph, t, doBlockProg, {dc, "attn_ij = exp(attn_ij)"});
            
            // compute sum exps
            auto s = popops::reduce(graph, t, t.elementType(), {2}, {popops::Operation::ADD}, doBlockProg, {dc, "s = sum(attn_ij, dim=2)"});

            // compute running max update
            auto newSums = popops::map(graph, pe::_1 * pe::_2 + pe::_3, {li, c, s}, doBlockProg, {dc, "l_new = l_i * c + s"});
            doBlockProg.add(Copy(newSums, li));
            
            // compute output(o_i = (c*o_i + attn_ij @ v_j))
            popops::mulInPlace(graph, oi, c.expand({2}), doBlockProg, {dc, "o_i *= c"});
            poplin::matMulGroupedAcc(graph, oi, 1.0, t, vj, doBlockProg, {dc, "o_i += attn_ij @ v_j"});
            
            Sequence skipBlockProg;
            kvLoopProg.add(If(doBlock, doBlockProg, skipBlockProg));

            // update kv loop counter
            popops::mapInPlace(graph, pe::_1 + 1, {kvCounter}, kvLoopProg, {dc, "j+=1"});
        }
        // repeat kv loop body in q loop body
        qLoopProg.add(Repeat(key.dim(0), kvLoopProg, {dc, "serialised_attention_inner_loop_repeat"}));
        
        popops::divInPlace(graph, oi, li.expand({2}), qLoopProg, {dc, "o_i /= l_i"});
        li = popops::map(graph, pe::_1 + pe::Log(pe::_2), {mi, li}, qLoopProg, {dc, "l_i = m_i + log(l_i)"});
        
        // update output and running softmax stats
        popops::dynamicUpdate(graph, out, oi.expand({0}), qCounter, {0}, {1}, qLoopProg, {dc, "o = o.at[i].set(o_i)"});
        popops::dynamicUpdate(graph, logSumExp, li.expand({0}), qCounter, {0}, {1}, qLoopProg, {dc, "update logSumExp"});

        // update q loop counter
        popops::mapInPlace(graph, pe::_1 + 1, {qCounter}, qLoopProg, {dc, "i+=1"});
        // reset kv loop counter
        popops::zero(graph, kvCounter, qLoopProg, {dc, "j=0"});
    }
    prog.add(Repeat(query.dim(0), qLoopProg, {dc, "serialised_attention_outer_loop_repeat"}));
    out = out.dimShuffle({1, 0, 2, 3}).reshape({groups, seqLen, headDim});

    std::vector<poplar::Tensor> tensors;
    tensors.push_back(out);
    tensors.push_back(logSumExp);
    return tensors;
}

poplar::Tensor serialisedAttention(
    poplar::Graph& graph, 
    const poplar::Tensor& qkv,  // Shape 3 x G x L x D
    uint32_t num_chunks_q, 
    uint32_t num_chunks_kv,
    poplar::program::Sequence& prog,
    const poplar::DebugContext& dc) {

    auto out_tensors = serialisedAttentionImpl(graph, qkv, num_chunks_q, num_chunks_kv, prog, {dc, "serialised_attention"});
    return out_tensors[0];
}

poplar::Tensor serialisedAttentionGrad(
    poplar::Graph& graph,
    const poplar::Tensor& grad, // Shape G x L x D
    const poplar::Tensor& qkv, // Shape 3 x G x L x D
    uint32_t num_chunks_q,
    uint32_t num_chunks_kv,
    poplar::program::Sequence& prog,
    const poplar::DebugContext& dc) {

    auto out_tensors = serialisedAttentionImpl(graph, qkv, num_chunks_q, num_chunks_kv, prog, {dc, "recompute_output"});
    auto out = out_tensors[0];
    auto logSumExp = out_tensors[1];
    assert(qkv.dim(0) == 3);

    // get shape data
    auto groups = qkv.dim(1); // groups = batch_size * num_heads
    auto seqLen = qkv.dim(2);
    auto headDim = qkv.dim(3);
    
    assert(seqLen % num_chunks_q == 0);
    assert(seqLen % num_chunks_kv == 0);

    // compute sizes of chunked sequence length
    auto chunkedQueryLen = seqLen / num_chunks_q; 
    auto chunkedKVLen = seqLen / num_chunks_kv; 
    
    popops::mulInPlace(graph, out, grad, prog, {dc, "D = out * grad"});
    auto sumOutXGradUnmapped = popops::reduce(graph, out, out.elementType(), {2}, {popops::Operation::ADD}, prog, {dc, "s=sum(D)"});
    auto sumOutXGrad = graph.clone(logSumExp);
    prog.add(Copy(sumOutXGradUnmapped.reshape({groups, num_chunks_q, chunkedQueryLen}).dimShuffle({1, 0, 2}), sumOutXGrad));

    // Unpack q,k,v and copy data to sliceable tensors with nice tile mappings
    auto query = popops::createSliceableTensor(graph, qkv.elementType(), {num_chunks_q, groups, chunkedQueryLen, headDim}, {0}, {1}, 4, {dc, "create_query"});
    auto key = popops::createSliceableTensor(graph, qkv.elementType(), {num_chunks_kv, groups, chunkedKVLen, headDim}, {0}, {1}, 4, {dc, "create_key"});
    auto value = popops::createSliceableTensor(graph, qkv.elementType(), {num_chunks_kv, groups, chunkedKVLen, headDim}, {0}, {1}, 4, {dc, "create_value"});
    auto out_grad = popops::createSliceableTensor(graph, grad.elementType(), {num_chunks_q, groups, chunkedQueryLen, headDim}, {0}, {1}, 4, {"create_grad"});

    prog.add(Copy(qkv[0].reshape({groups, num_chunks_q, chunkedQueryLen, headDim}).dimShuffle({1, 0, 2, 3}), query));
    prog.add(Copy(qkv[1].reshape({groups, num_chunks_kv, chunkedKVLen, headDim}).dimShuffle({1, 0, 2, 3}), key));
    prog.add(Copy(qkv[2].reshape({groups, num_chunks_kv, chunkedKVLen, headDim}).dimShuffle({1, 0, 2, 3}), value));
    prog.add(Copy(grad.reshape({groups, num_chunks_q, chunkedQueryLen, headDim}).dimShuffle({1, 0, 2, 3}), out_grad));

    auto query_grad = graph.clone(query, {dc, "dquery = array_like(query)"});
    auto key_grad = graph.clone(key, {dc, "dkey = array_like(key)"});
    auto value_grad = graph.clone(value, {dc, "dvalue = array_like(value)"});

    popops::zero(graph, query_grad, prog, {dc, "zero_dquery"});
    popops::zero(graph, key_grad, prog, {dc, "zero_dkey"});
    popops::zero(graph, value_grad, prog, {dc, "zero_dvalue"});

    // outer loop counter on kv read
    auto kvCounter = graph.addVariable(poplar::UNSIGNED_INT, {1}, {dc, "init_kvCounter(j=0)"});
    // inner loop counter on q read
    auto qCounter = graph.addVariable(poplar::UNSIGNED_INT, {1}, {dc, "init_qCounter(i=0)"});
    
    // gimme tiles
    poputil::mapTensorLinearly(graph, kvCounter);
    poputil::mapTensorLinearly(graph, qCounter);

    popops::zero(graph, kvCounter, prog, {dc, "zero_kvCounter"});
    popops::zero(graph, qCounter, prog, {dc, "zero_qCounter"});

    Sequence kvLoopProg; {
        auto kj = popops::dynamicSlice(graph, key, kvCounter, {0}, {1}, kvLoopProg, {dc, "k_j = k.at[j].get()"}).squeeze({0});
        auto dkj = popops::dynamicSlice(graph, key_grad, kvCounter, {0}, {1}, kvLoopProg, {dc, "dk_j = dk.at[j].get()"}).squeeze({0});
        auto vj = popops::dynamicSlice(graph, value, kvCounter, {0}, {1}, kvLoopProg, {dc, "v_j = v.at[j].get()"}).squeeze({0});
        auto dvj = popops::dynamicSlice(graph, value_grad, kvCounter, {0}, {1}, kvLoopProg, {dc, "dv_j = dv.at[j].get()"}).squeeze({0});

        Sequence qLoopProg; {

            // Condition for executing (true) or skipping (false) block
            auto doBlock = popops::map(graph, ((pe::_1 + 1) * uint(chunkedQueryLen)) > (pe::_2 * uint(chunkedKVLen)), {qCounter, kvCounter}, qLoopProg, {dc, "(i+1) * q_chunk_size > j * kv_chunk_size"})[0];

            // Conditional execute block program body
            Sequence doBlockProg; 

            auto qi = popops::dynamicSlice(graph, query, qCounter, {0}, {1}, doBlockProg, {dc, "q_i = q.at[i].get()"}).squeeze({0});
            auto dqi = popops::dynamicSlice(graph, query_grad, qCounter, {0}, {1}, doBlockProg, {dc, "dq_i = dq.at[i].get()"}).squeeze({0});
            auto doi = popops::dynamicSlice(graph, out_grad, qCounter, {0}, {1}, doBlockProg, {dc, "do_i = do.at[i].get()"}).squeeze({0});
            auto li = popops::dynamicSlice(graph, logSumExp, qCounter, {0}, {1}, doBlockProg, {dc, "lse_i = lse.at[i].get()"}).squeeze({0});
            auto si = popops::dynamicSlice(graph, sumOutXGrad, qCounter, {0}, {1}, doBlockProg, {dc, "s_i = s.at[i].get()"}).squeeze({0});

            auto t = poplin::matMulGrouped(graph, qi, kj.dimShuffle({0, 2, 1}), doBlockProg, kj.elementType(), {dc, "attn_ij = q_i @ k_j.T"});
            
            // Condition for making mask
            auto doMakeMasks = popops::map(graph, (pe::_1 == 0) && (pe::_2 == 0), {qCounter, kvCounter}, doBlockProg, {dc, "i==0 and j==0"})[0];
            
            // generate causal masks
            // clone attention matrix block to colocate mask block elements with attn matrix block elements
            Sequence doMakeMasksProg;
            std::vector<int32_t> offsets = getTriuOffsetSequence(chunkedQueryLen, chunkedKVLen);
            // use cloneN to generate as many as needed by offsets.size()
            auto masks = graph.cloneN(t.elementType(), t[0], offsets.size(), {dc, "masks = repeat(array_like(t[0]), offsets.size())"});
            popops::fill(graph, masks, doMakeMasksProg, -10000.0, {dc, "fill(masks, -10000.0)"});
            // Triu function above didn't work when passing a slice of a tensor, so it is inlined here
            for (size_t i = 0; i < offsets.size(); ++i){
                int k = offsets[i];
                int m = masks.dim(masks.rank() - 2);
                int n = masks.dim(masks.rank() - 1);

                size_t start = 0;
                for (int j = m; j > 0 && j-1+k > 0; --j){
                    size_t end = size_t(std::min(j-1+k, n));
                    popops::zero(graph, masks.slice({i, size_t(j-1), start}, {i+1, size_t(j), end}), doMakeMasksProg, {dc, "zero_for_triu(masks)"});
                    }
            }
            // mask counter on masked block execution
            auto maskCounter = graph.addVariable(poplar::UNSIGNED_INT, {1}, {dc, "init_maskCounter(k=0)"});
            poputil::mapTensorLinearly(graph, maskCounter);
            popops::zero(graph, maskCounter, doMakeMasksProg, {dc, "zero_maskCounter"});

            Sequence skipMakeMasksProg;
            doBlockProg.add(If(doMakeMasks, doMakeMasksProg, skipMakeMasksProg, {dc, "initialise_masks"}));

            // Condition for adding mask to q@k.T
            auto doMask = popops::map(graph, (pe::_1 * uint(chunkedQueryLen) < ((pe::_2 + 1) * uint(chunkedKVLen) - 1)), {qCounter, kvCounter}, doBlockProg, {dc, "i * q_chunk_size < (j+1) * kv_chunk_size - 1"})[0];
            
            // Conditional add mask program body
            Sequence doMaskProg; 
            auto blockMask = popops::dynamicSlice(graph, masks, maskCounter, {0}, {1}, doMaskProg, {dc, "get_mask"}).squeeze({0});
            popops::addInPlace(graph, t, blockMask.expand({0}), doMaskProg, {dc, "attn_ij += mask_ij"});
            // update mask counter
            popops::mapInPlace(graph, ((pe::_1 + 1)%uint(masks.dim(0))), {maskCounter}, doMaskProg, {dc, "k = (k+1)%masks.size()"});

            // Empty skip mask program
            Sequence skipMaskProg;

            // Add conditional mask program to execute block program 
            doBlockProg.add(If(doMask, doMaskProg, skipMaskProg, {dc, "q@k.T + mask if i==j else q@k.T"}));
            
            popops::subInPlace(graph, t, li.expand({2}), doBlockProg, {dc, "attn_ij -= li"});
            popops::expInPlace(graph, t, doBlockProg, "attn_ij = exp(att_ij)");

            poplin::matMulGroupedAcc(graph, dvj, 1.0, t.dimShuffle({0, 2, 1}), doi, doBlockProg, {dc, "dv_j += attn_ij.T @ do_i"});
            
            auto dt = poplin::matMulGrouped(graph, doi, vj.dimShuffle({0, 2, 1}), doBlockProg, doi.elementType(), {dc, "dattn_ij = do_i @ v_j.T"});
            popops::subInPlace(graph, dt, si.expand({2}), doBlockProg, {dc, "dattn_ij -= s_i"});
            popops::mulInPlace(graph, dt, t, doBlockProg, {dc, "dattn_ij *= attn_ij"});
            
            poplin::matMulGroupedAcc(graph, dqi, 1.0, dt, kj, doBlockProg, {dc, "dq_i += dattn_ij @ k_j"});
            poplin::matMulGroupedAcc(graph, dkj, 1.0, dt.dimShuffle({0, 2, 1}), qi, doBlockProg, {dc, "dk_j += dattn_ij.T @ q_i"});

            popops::dynamicUpdate(graph, query_grad, dqi.expand({0}), qCounter, {0}, {1}, doBlockProg, {dc, "dq = dq.at[i].set(dq_i"});

            Sequence skipBlockProg;
            qLoopProg.add(If(doBlock, doBlockProg, skipBlockProg));

            popops::mapInPlace(graph, pe::_1 + 1, {qCounter}, qLoopProg, {dc, "i+=1"});
        }
        kvLoopProg.add(Repeat(query.dim(0), qLoopProg, {dc, "backward_inner_repeat"}));

        popops::dynamicUpdate(graph, key_grad, dkj.expand({0}), kvCounter, {0}, {1}, kvLoopProg, {dc, "dk = dk.at[j].set(dk_j)"});
        popops::dynamicUpdate(graph, value_grad, dvj.expand({0}), kvCounter, {0}, {1}, kvLoopProg, {dc, "dv = dv.at[j].set(dv_j)"});

        // update kv loop counter
        popops::mapInPlace(graph, pe::_1 + 1, {kvCounter}, kvLoopProg, {dc, "j+=1"});
        // reset kv loop counter
        popops::zero(graph, qCounter, kvLoopProg, {dc, "i=0"});
    }
    prog.add(Repeat(key.dim(0), kvLoopProg, {dc, "backward_outer_repeat"}));

    query_grad = query_grad.dimShuffle({1, 0, 2, 3}).reshape({groups, seqLen, headDim});
    key_grad = key_grad.dimShuffle({1, 0, 2, 3}).reshape({groups, seqLen, headDim});
    value_grad = value_grad.dimShuffle({1, 0, 2, 3}).reshape({groups, seqLen, headDim});
    auto dqkv_cat = poplar::concat({query_grad.expand({0}), key_grad.expand({0}), value_grad.expand({0})}, 0);
    auto dqkv = graph.clone(qkv);
    prog.add(Copy(dqkv_cat, dqkv));

    return dqkv;
}

// popart

const popart::OperatorIdentifier SerialisedAttentionId = {"ai.graphcore", "SerialisedAttention", 1};
const popart::OperatorIdentifier SerialisedAttentionGradId = {"ai.graphcore", "SerialisedAttentionGrad", 1};

class SerialisedAttentionOp;
class SerialisedAttentionGradOp;

class SerialisedAttentionGradOp : public popart::Op {
    public:
    SerialisedAttentionGradOp(const SerialisedAttentionOp& fwdOp);

    std::unique_ptr<popart::Op> clone() const final { 
        return std::make_unique<SerialisedAttentionGradOp>(*this);
    }

    void setup() final {outInfo(0) = inInfo(1);};

    void appendAttributes(popart::OpSerialiserBase& os) const override;

    void appendOutlineAttributes(popart::OpSerialiserBase& os) const override;

    const std::vector<popart::GradInOutMapper> &gradInputInfo() const {
        static const std::vector<popart::GradInOutMapper> inInfo = {
            {0, 0, popart::GradOpInType::GradOut},
            {1, 0, popart::GradOpInType::In}};
        return inInfo;
    }

    const std::map<int, int> &gradOutToNonGradIn() const {
        static const std::map<int, int> outInfo = {{0, 0}};
        return outInfo;
    }

    float getSubgraphValue() const final {return getHighSubgraphValue();}

    bool requiresRandomSeed() const override { return false; }

    unsigned getNumChunksQ() const {return num_chunks_q; }
    unsigned getNumChunksKV() const {return num_chunks_kv; }

    private:
    unsigned num_chunks_q;
    unsigned num_chunks_kv;
};

class SerialisedAttentionOp : public popart::Op {
    public:
    SerialisedAttentionOp(
        const popart::OperatorIdentifier& _opid,
        unsigned _num_chunks_q,
        unsigned _num_chunks_kv,
        const popart::Op::Settings& settings_
    ) : popart::Op(_opid, settings_), num_chunks_q(_num_chunks_q), num_chunks_kv(_num_chunks_kv) {}

    std::unique_ptr<Op> clone() const final { return std::make_unique<SerialisedAttentionOp>(*this); }

    void setup() final {
        auto qkvInfo = inInfo(0);
        assert(qkvInfo.rank() == 4);
        assert(qkvInfo.dim(0) == 3);

        outInfo(0) = popart::TensorInfo(qkvInfo.dataType(), {qkvInfo.dim(1), qkvInfo.dim(2), qkvInfo.dim(3)});
    }

    std::vector<std::unique_ptr<popart::Op>> getGradOps() {
        std::vector<std::unique_ptr<Op>> upops;
        upops.emplace_back(new SerialisedAttentionGradOp(*this));
        return upops;
    }

    void appendAttributes(popart::OpSerialiserBase& os) const override {
        popart::Op::appendAttributes(os);
        os.appendAttribute("num_chunks_q", getNumChunksQ());
        os.appendAttribute("num_chunks_kv", getNumChunksKV());
    }

    void appendOutlineAttributes(popart::OpSerialiserBase& os) const override {
        Op::appendOutlineAttributes(os);
        os.appendAttribute("num_chunks_q", getNumChunksQ());
        os.appendAttribute("num_chunks_kv", getNumChunksKV());
    }

    float getSubgraphValue() const final { return getHighSubgraphValue(); }

    bool requiresRandomSeed() const override { return false; }

    unsigned getNumChunksQ() const {return num_chunks_q; }
    unsigned getNumChunksKV() const {return num_chunks_kv; }

    private:
    unsigned num_chunks_q;
    unsigned num_chunks_kv;
};

static popart::OpDefinition::DataTypes T = {popart::DataType::FLOAT16, popart::DataType::FLOAT};

static popart::OpDefinition 
    SerialisedAttentionOpDef({
        popart::OpDefinition::Inputs({{"qkv", T}}),
        popart::OpDefinition::Outputs({{"output", T}}),
        popart::OpDefinition::Attributes({{"num_chunks_q", {"int"}}, {"num_chunks_kv", {"int"}}})
    });

static popart::OpCreator<SerialisedAttentionOp> SerialisedAttentionOpCreator(
    popart::OpDefinitions({{SerialisedAttentionId, SerialisedAttentionOpDef}}),
    [](const popart::OpCreatorInfo& info) {
        auto num_chunks_q = unsigned(info.attributes.getAttribute<popart::Attributes::Int>("num_chunks_q", 1u));
        auto num_chunks_kv = unsigned(info.attributes.getAttribute<popart::Attributes::Int>("num_chunks_kv", 1u));
        return std::make_unique<SerialisedAttentionOp>(info.opid, num_chunks_q, num_chunks_kv, info.settings);
    },
    true);

class SerialisedAttentionOpx : public popart::popx::Opx {
    public:
    SerialisedAttentionOpx(popart::Op* op, popart::popx::Devicex* devicex) : popart::popx::Opx(op, devicex) {
        verifyOp<SerialisedAttentionOp>(op, {SerialisedAttentionId});
    }

    void grow(poplar::program::Sequence& prog) const final {
        auto op = getOp<SerialisedAttentionOp>();
        poplar::Tensor qkv = getInTensor(0);
        auto num_chunks_q = op.getNumChunksQ();
        auto num_chunks_kv = op.getNumChunksKV();
        poplar::Tensor out = serialisedAttention(graph(), qkv, num_chunks_q, num_chunks_kv, prog, "attention");
        setOutTensor(0, out);
    }
};

class SerialisedAttentionGradOpx : public popart::popx::Opx {
    public:
    SerialisedAttentionGradOpx(popart::Op* op, popart::popx::Devicex* devicex) : popart::popx::Opx(op, devicex) {
        verifyOp<SerialisedAttentionGradOp>(op, {SerialisedAttentionGradId});
    }

    void grow(poplar::program::Sequence& prog) const final {
        auto op = getOp<SerialisedAttentionGradOp>();
        poplar::Tensor grad = getInTensor(0);
        poplar::Tensor qkv = getInTensor(1);
        auto num_chunks_q = op.getNumChunksQ();
        auto num_chunks_kv = op.getNumChunksKV();
        poplar::Tensor out = serialisedAttentionGrad(graph(), grad, qkv, num_chunks_q, num_chunks_kv, prog, "attention_grad");
        setOutTensor(0, out);
    }
};

SerialisedAttentionGradOp::SerialisedAttentionGradOp(const SerialisedAttentionOp& fwdOp)
    : popart::Op(SerialisedAttentionGradId, fwdOp.settings), num_chunks_q(fwdOp.getNumChunksQ()), num_chunks_kv(fwdOp.getNumChunksKV()) {}

void SerialisedAttentionGradOp::appendAttributes(popart::OpSerialiserBase& os) const {
    popart::Op::appendAttributes(os);
    os.appendAttribute("num_chunks_q", getNumChunksQ());
    os.appendAttribute("num_chunks_kv", getNumChunksKV());
}

void SerialisedAttentionGradOp::appendOutlineAttributes(popart::OpSerialiserBase& os) const {
    Op::appendOutlineAttributes(os);
    os.appendAttribute("num_chunks_q", getNumChunksQ());
    os.appendAttribute("num_chunks_kv", getNumChunksKV());
}

static popart::popx::OpxCreator<SerialisedAttentionOpx> SerialisedAttentionOpxCreator({SerialisedAttentionId});
static popart::popx::OpxCreator<SerialisedAttentionGradOpx> SerialisedAttentionGradOpxCreator({SerialisedAttentionGradId});
