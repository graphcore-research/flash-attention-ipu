#include <poplar/Engine.hpp>
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
#include <poprand/RandomGen.hpp>
#include <poplar/CycleCount.hpp>
#include <poplar/SyncType.hpp>

#include <poplin/codelets.hpp>
#include <popops/codelets.hpp>
#include <popnn/codelets.hpp>
#include <poprand/codelets.hpp>

#include <iostream>
#include <algorithm>
#include <vector>

using namespace poplar;
using namespace poplar::program;
namespace pe = popops::expr;

template <typename S>

void printVectorElements(std::vector<S> vec){
    for (auto elm : vec){
        std::cout << elm << " ";
    }
    std::cout << std::endl;
}

std::vector<int32_t> getTriuOffsetSequence(
    uint32_t numRows,
    uint32_t numCols
) {
    std::vector<int32_t> offsets = {1};
    int tmp_offset = 1;
    int max_offset = numRows - 1;
    int min_offset = 2 - numCols;

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
        popops::zero(graph, t.slice({size_t(i-1), start}, {size_t(i), end}), prog);
        }
    }

poplar::Tensor vanillaAttention(
    poplar::Graph& graph,
    const poplar::Tensor& qkv, // Shape 3 x G x L x D
    poplar::program::Sequence& prog,
    const poplar::DebugContext& dc) {

    assert(qkv.dim(0) == 3);

    // get shape data
    auto groups = qkv.dim(1); // groups = batch_size * num_heads
    auto seqLen = qkv.dim(2);
    auto headDim = qkv.dim(3);

    // q, k, v = qkv
    auto query = qkv[0];
    auto key = qkv[1];
    auto value = qkv[2];

    auto mask = graph.addVariable(query.elementType(), {seqLen, seqLen}, {dc, "mask = array(L, L)"});
    poputil::mapTensorLinearly(graph, mask);
    popops::fill(graph, mask, prog, -10000.0, {dc, "fill(mask, -10000)"});
    triu(graph, mask, 1, prog, {dc, "triu(mask)"});

    // q @ k.T + mask
    auto attn = poplin::matMulGrouped(graph, query, key.dimShuffle({0, 2, 1}), prog, query.elementType(), {dc, "attn = Q@K.T"});
    popops::addInPlace(graph, attn, mask.expand({0}), prog, {dc, "attn += mask"});

    // Stable softmax(x)
    auto m = popops::reduce(graph, attn, attn.elementType(), {2}, {popops::Operation::MAX}, prog, {dc, "m = max(attn, dim=2)"});
    popops::subInPlace(graph, attn, m.expand({2}), prog, {dc, "attn -= m"});
    popops::expInPlace(graph, attn, prog, {dc, "attn = exp(attn)"});
    auto s = popops::reduce(graph, attn, attn.elementType(), {2}, {popops::Operation::ADD}, prog, {dc, "s = sum(attn, dim=2)"});
    popops::invInPlace(graph, s, prog, {dc, "s = 1/s"});
    popops::mulInPlace(graph, attn, s.expand({2}), prog, {dc, "attn *= s"});

    // attn @ v
    auto out = poplin::matMulGrouped(graph, attn, value, prog, value.elementType(), {dc, "out = attn@V"});
    return out;
}

poplar::Tensor serialisedAttention(
    poplar::Graph& graph, 
    const poplar::Tensor& qkv,  // Shape 3 x G x L x D
    const uint32_t& num_chunks_q, 
    const uint32_t& num_chunks_kv,
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

    // create tensors to store running softmax statistics
    auto runningStats = popops::createSliceableTensor(graph, query.elementType(), {num_chunks_q, 2, groups, chunkedQueryLen}, {0}, {1}, 4, {dc, "create running softmax stats"});

    popops::zero(graph, runningStats.slice({0, 1}, 1), prog, {dc, "zero_running_sum"});
    popops::fill(graph, runningStats.slice({1, 2}, 1), prog, -10000.0, "fill_running_max");
    
    // create the masks needed ahead of time (this could be horrible, but hopefully not for typical use cases)
    
    /* 
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

    The function `getTriuOffsetSequence` generates the correct sequence of offsets for use in a causal mask

    We then use this sequence to generate the mask blocks needed for use in the serialised attention loop 
    */
    std::vector<int32_t> offsets = getTriuOffsetSequence(chunkedQueryLen, chunkedKVLen);
    Tensor tmp_masks = graph.addConstant(qkv.elementType(), {offsets.size(), chunkedQueryLen, chunkedKVLen}, -10000.0, {"initialise_masks"});
    poputil::mapTensorLinearly(graph, tmp_masks);
    Tensor masks = popops::createSliceableTensor(graph, qkv.elementType(), {offsets.size(), chunkedQueryLen, chunkedKVLen}, {0}, {1}, 1, {"sliceable_masks"});
    prog.add(Copy(tmp_masks, masks));

    // Triu function above didn't work when passing a slice of a tensor, so it is inlined here
    for (size_t i = 0; i < offsets.size(); ++i){
        int k = offsets[i];
        int m = masks.dim(masks.rank() - 2);
        int n = masks.dim(masks.rank() - 1);

        size_t start = 0;
        for (int j = m; j > 0 && j-1+k > 0; --j){
            size_t end = size_t(std::min(j-1+k, n));
            popops::zero(graph, masks.slice({i, size_t(j-1), start}, {i+1, size_t(j), end}), prog);
            }
    }
    
    // outer loop counter on kv read
    auto kvCounter = graph.addVariable(poplar::UNSIGNED_INT, {1}, {dc, "init_kvCounter(j=0)"});
    // inner loop counter on q read
    auto qCounter = graph.addVariable(poplar::UNSIGNED_INT, {1}, {dc, "init_qCounter(i=0)"});
    
    // mask counter on masked block execution
    auto maskCounter = graph.addVariable(poplar::UNSIGNED_INT, {1}, {dc, "init_maskCounter(k=0)"});

    // identity constants for inplace updates
    Tensor oneu = graph.addConstant<uint32_t>(UNSIGNED_INT, {1}, {1}, {dc, "init_counterIncrement"}); // for counters
    Tensor onef = graph.addConstant<float>(qkv.elementType(), {1}, {1.0}, {dc, "init_mmAccScale"}); // for output
    
    // gimme tiles
    poputil::mapTensorLinearly(graph, masks);
    poputil::mapTensorLinearly(graph, maskCounter);
    poputil::mapTensorLinearly(graph, kvCounter);
    poputil::mapTensorLinearly(graph, qCounter);
    poputil::mapTensorLinearly(graph, oneu);
    poputil::mapTensorLinearly(graph, onef);

    // Setup repeat loops. Use whitespace indentation for python-like readability

    // kv loop body program
    Sequence kvLoopProg; {
        // slice k and v
        auto kj = popops::dynamicSlice(graph, key, kvCounter, {0}, {1}, kvLoopProg, {dc, "k_j = k.at[j].get()"}).squeeze({0});
        auto vj = popops::dynamicSlice(graph, value, kvCounter, {0}, {1}, kvLoopProg, {dc, "v_j = v.at[j].get()"}).squeeze({0});

        // q loop body program
        Sequence qLoopProg; {
            
            // Condition for executing (true) or skipping (false) block
            auto doBlock = popops::map(graph, ((pe::_1 + 1) * uint(chunkedQueryLen) - 1) > (pe::_2 * uint(chunkedKVLen)), {qCounter, kvCounter}, qLoopProg, {dc, "(i+1) * q_chunk_size > j * kv_chunk_size"})[0];            
            
            // Conditional execute block program body
            Sequence doBlockProg; 

            // slice q, output, and softmax running stats
            auto qi = popops::dynamicSlice(graph, query, qCounter, {0}, {1}, doBlockProg, {dc, "q_i = q.at[i].get()"}).squeeze({0}); 
            auto oi = popops::dynamicSlice(graph, out, qCounter, {0}, {1}, doBlockProg, {dc, "o_i = o.at[i].get()"}).squeeze({0});             
            
            auto runningStatsi = popops::dynamicSlice(graph, runningStats, qCounter, {0}, {1}, doBlockProg, {dc, "get chunk stats"}).squeeze({0});
            auto runningSumsi = runningStatsi.slice({0, 1}, 0).squeeze({0});
            auto runningMaxsi = runningStatsi.slice({1, 2}, 0).squeeze({0});

            // compute qk^t
            auto t = poplin::matMulGrouped(graph, qi, kj.dimShuffle({0, 2, 1}), doBlockProg, kj.elementType(), {dc, "attn_ij = q_i @ k_j.T"});
            
            // Condition for adding mask to q@k.T
            auto doMask = popops::map(graph, (pe::_1 * uint(chunkedQueryLen) <= ((pe::_2 + 1) * uint(chunkedKVLen) - 1)), {qCounter, kvCounter}, doBlockProg, {dc, "i * q_chunk_size <= (j+1) * kv_chunk_size"})[0];
            
            // Conditional add mask program body
            Sequence doMaskProg; 
            auto blockMask = popops::dynamicSlice(graph, masks, maskCounter, {0}, {1}, doMaskProg, {dc, "get_mask"}).squeeze({0});
            popops::addInPlace(graph, t, blockMask.expand({0}), doMaskProg, {dc, "attn_ij += mask_ij"});
            // update mask counter
            popops::mapInPlace(graph, ((pe::_1 + 1)%uint(masks.dim(0))), {maskCounter}, prog, {dc, "k = (k+1)%masks.size()"});

            // Empty skip mask program
            Sequence skipMaskProg;

            // Add conditional mask program to execute block program 
            doBlockProg.add(If(doMask, doMaskProg, skipMaskProg, {dc, "q@k.T + mask if i==j else q@k.T"}));

            // compute qk^T max for stable softmax
            auto tmpMaxs = popops::reduce(graph, t, t.elementType(), {2}, {popops::Operation::MAX}, doBlockProg, {dc, "m_tmp = sum(attn_ij, dim=2)"});
            // subtract max from qk^T
            popops::subInPlace(graph, t, tmpMaxs.expand({2}), doBlockProg, {dc, "attn_ij -= m_tmp"});
            // compute softmax numerator: exp(qk^T - max)
            popops::expInPlace(graph, t, doBlockProg, {dc, "attn_ij = exp(attn_ij)"});
            
            // compute running max update
            auto newMaxs = popops::max(graph, runningMaxsi, tmpMaxs, doBlockProg, {dc, "m_new = max(m_i, m_tmp)"});
            
            // compute softmax update scaling factors
            auto tmpSumScale = popops::sub(graph, tmpMaxs, newMaxs, doBlockProg, {dc, "c_tmp = m_tmp - m_new"});
            popops::expInPlace(graph, tmpSumScale, doBlockProg, {dc, "c_tmp = exp(c_tmp)"});
            auto runningSumScale = popops::sub(graph, runningMaxsi, newMaxs, doBlockProg, {dc, "c_i = m_i - m_new"});
            popops::expInPlace(graph, runningSumScale, doBlockProg, {dc, "c_i = exp(c_i)"});

            // compute running sum update
            auto newSums = popops::reduce(graph, t, t.elementType(), {2}, {popops::Operation::ADD}, doBlockProg, {dc, "s_new = sum(attn_ij, dim=2)"});
            
            // scale updates from past statistics
            popops::mulInPlace(graph, newSums, tmpSumScale, doBlockProg, {dc, "s_new *= c_tmp"});
            popops::mulInPlace(graph, runningSumsi, runningSumScale, doBlockProg, {dc, "s_i *= c_i"});
            popops::addInPlace(graph, newSums, runningSumsi, doBlockProg, {dc, "s_new += s_i"});

            // compute 
            popops::mulInPlace(graph, oi, runningSumsi.expand({2}), doBlockProg, {dc, "o_i *= s_i"});
            popops::mulInPlace(graph, t, tmpSumScale.expand({2}), doBlockProg, {dc, "attn_ij *= c_tmp"});
            poplin::matMulGroupedAcc(graph, oi, onef, t, vj, doBlockProg, {dc, "oi += attn_ij @ v_j"});
            auto invNewSums = popops::inv(graph, newSums.expand({2}), doBlockProg, {dc, "s_inv = 1 / s_new"});
            popops::mulInPlace(graph, oi, invNewSums, doBlockProg, {dc, "o_i *= s_inv"});

            // update output and running softmax stats
            popops::dynamicUpdate(graph, out, oi.expand({0}), qCounter, {0}, {1}, doBlockProg, {dc, "o = o.at[i].set(o_i)"});
            popops::dynamicUpdate(graph, runningStats, poplar::concat(newSums.expand({0}), newMaxs.expand({0})).expand({0}), qCounter, {0}, {1}, doBlockProg, {dc, "update chunk stats"});
            
            Sequence skipBlockProg;
            qLoopProg.add(If(doBlock, doBlockProg, skipBlockProg));

            popops::addInPlace(graph, qCounter, oneu, qLoopProg, {dc, "i+=1"});
            // update q loop counter
        }
        // repeat q loop body in kv loop body
        kvLoopProg.add(Repeat(query.dim(0), qLoopProg, {dc, "serialised_attention_inner_loop_repeat"}));
        // update kv loop counter
        popops::addInPlace(graph, kvCounter, oneu, kvLoopProg, {dc, "j+=1"});
        // reset q loop counter
        popops::zero(graph, qCounter, kvLoopProg, {dc, "i=0"});
    }
    prog.add(Repeat(key.dim(0), kvLoopProg, {dc, "serialised_attention_outer_loop_repeat"}));

    out = out.dimShuffle({1, 0, 2, 3}).reshape({groups, seqLen, headDim});
    return out;
}

int main(){

    Device device;
    auto manager = DeviceManager::createDeviceManager();
    auto devices = manager.getDevices(poplar::TargetType::IPU, 1);
    std::cout << "Trying to attach to IPU \n";
    
    auto it = std::find_if(devices.begin(), devices.end(), [](Device &device){return device.attach();});

    if (it == devices.end()) {
        std::cerr << "Error attaching to device \n";
        return 1;
    }
    device = std::move(*it);
    std::cout << "Attached to IPU " << device.getId() << std::endl;

    Graph graph(device.getTarget());

    popops::addCodelets(graph);
    poplin::addCodelets(graph);
    popnn::addCodelets(graph);
    poprand::addCodelets(graph);

    Tensor qkv = graph.addVariable(HALF, {3, 8, 2048, 128});
    poputil::mapTensorLinearly(graph, qkv);

    const Tensor seed = graph.addConstant<uint32_t>(UNSIGNED_INT, {2}, {40, 90});
    poputil::mapTensorLinearly(graph, seed);

    poplar::DebugContext dc;

    Sequence prog;
    
    qkv = poprand::normal(graph, &seed, 0, qkv, qkv.elementType(), 0.0, 1.0, prog);
    
    Sequence vanillaAttentionProg;
    Sequence serialisedAttentionProg;

    auto out_v = vanillaAttention(graph, qkv, vanillaAttentionProg, {dc, "vanilla_attention"});
    auto out_s = serialisedAttention(graph, qkv, 4, 4, serialisedAttentionProg, {dc, "serialised_attention"});

    auto vanillaAttentionCycles = poplar::cycleCount(graph, vanillaAttentionProg, 0, poplar::SyncType::EXTERNAL, {dc, "count cycles"});
    auto serialisedAttentionCycles = poplar::cycleCount(graph, serialisedAttentionProg, 0, poplar::SyncType::EXTERNAL,{dc, "count cycles"});

    prog.add(vanillaAttentionProg);
    prog.add(serialisedAttentionProg);

    auto err = popops::sub(graph, out_v, out_s, prog, "e = x - y");
    popops::absInPlace(graph, err, prog, "e = abs(e)");
    auto maxErr = popops::reduce(graph, err, err.elementType(), {0, 1, 2}, {popops::Operation::MAX}, prog, "m = max(e)");
    prog.add(program::PrintTensor("maxErr", maxErr));
    prog.add(program::PrintTensor("vanilla attention cycles", vanillaAttentionCycles[0])); // fine so long as it isn't > 2**31 cycles
    prog.add(program::PrintTensor("serialised attention cycles", serialisedAttentionCycles[0])); // fine so long as it isn't > 2**31 cycles

    Engine engine(graph, prog, {{"debug.instrument", "true"}});
    engine.load(device);
    engine.run(0);
    return 0;

}