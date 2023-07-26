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
#include <poplin/Norms.hpp>
#include <popnn/LayerNorm.hpp>
#include <popnn/NonLinearity.hpp>
#include <poprand/RandomGen.hpp>
#include <poputil/Broadcast.hpp>
#include <poputil/TileMapping.hpp>

#include <poplin/codelets.hpp>
#include <popops/codelets.hpp>
#include <popnn/codelets.hpp>
#include <poprand/codelets.hpp>

#include <iostream>
#include <algorithm>
#include <vector>

using namespace poplar;
using namespace poplar::program;

template <typename S>

void printVectorElements(std::vector<S> vec){
    for (auto elm : vec){
        std::cout << elm << " ";
    }
    std::cout << std::endl;
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
    
    // q @ k.T
    auto attn = poplin::matMulGrouped(graph, query, key.dimShuffle({0, 2, 1}), prog, query.elementType(), {dc, "attn = Q@K.T"});
    
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
    prog.add(Copy(qkv[1].reshape({groups, num_chunks_kv, chunkedQueryLen, headDim}).dimShuffle({1, 0, 2, 3}), key));
    prog.add(Copy(qkv[2].reshape({groups, num_chunks_kv, chunkedQueryLen, headDim}).dimShuffle({1, 0, 2, 3}), value));

    // create output tensor computed iteratively
    auto out = graph.clone(query, {dc, "create_output"});
    popops::zero(graph, out, prog, {dc, "zero_output"});

    // create tensors to store running softmax statistics
    auto runningSums = popops::createSliceableTensor(graph, query.elementType(), {num_chunks_q, groups, chunkedQueryLen}, {0}, {1}, 4, {dc, "create_running_sum"});
    auto runningMaxs = popops::createSliceableTensor(graph, query.elementType(), {num_chunks_q, groups, chunkedQueryLen}, {0}, {1}, 4, {dc, "create_running_max"});

    popops::zero(graph, runningSums, prog, {dc, "zero_running_sum"});
    popops::fill(graph, runningMaxs, prog, -10000.0, "fill_running_max");
    
    // outer loop counter on kv read
    auto kvCounter = graph.addVariable(poplar::UNSIGNED_INT, {1}, {dc, "init_kvCounter"});
    // inner loop counter on q read
    auto qCounter = graph.addVariable(poplar::UNSIGNED_INT, {1}, {dc, "init_qCounter"});

    // identity constants for inplace updates
    Tensor oneu = graph.addConstant<uint32_t>(UNSIGNED_INT, {1}, {1}, {dc, "init_counterIncrement"}); // for counters
    Tensor onef = graph.addConstant<float>(qkv.elementType(), {1}, {1.0}, {dc, "init_mmAccScale"}); // for output
    
    // gimme tiles
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
            // slice q, output, and softmax running stats
            auto qi = popops::dynamicSlice(graph, query, qCounter, {0}, {1}, qLoopProg, {dc, "q_i = q.at[i].get()"}).squeeze({0}); 
            auto oi = popops::dynamicSlice(graph, out, qCounter, {0}, {1}, qLoopProg, {dc, "o_i = o.at[i].get()"}).squeeze({0});             
            
            auto runningMaxsi = popops::dynamicSlice(graph, runningMaxs, qCounter, {0}, {1}, qLoopProg, {dc, "m_i = m.at[i].get()"}).squeeze({0}); 
            auto runningSumsi = popops::dynamicSlice(graph, runningSums, qCounter, {0}, {1}, qLoopProg, {dc, "s_i = s.at[i].get()"}).squeeze({0}); 

            // compute qk^T
            auto t = poplin::matMulGrouped(graph, qi, kj.dimShuffle({0, 2, 1}), qLoopProg, kj.elementType(), {dc, "attn_ij = q_i @ k_j.T"});
            
            // compute qk^T max for stable softmax
            auto tmpMaxs = popops::reduce(graph, t, t.elementType(), {2}, {popops::Operation::MAX}, qLoopProg, {dc, "m_tmp = sum(attn_ij, dim=2)"});
            // subtract max from qk^T
            popops::subInPlace(graph, t, tmpMaxs.expand({2}), qLoopProg, {dc, "attn_ij -= m_tmp"});
            // compute softmax numerator: exp(qk^T - max)
            popops::expInPlace(graph, t, qLoopProg, {dc, "attn_ij = exp(attn_ij)"});
            
            // compute running max update
            auto newMaxs = popops::max(graph, runningMaxsi, tmpMaxs, qLoopProg, {dc, "m_new = max(m_i, m_tmp)"});
            
            // compute softmax update scaling factors
            auto tmpSumScale = popops::sub(graph, tmpMaxs, newMaxs, qLoopProg, {dc, "c_tmp = m_tmp - m_new"});
            popops::expInPlace(graph, tmpSumScale, qLoopProg, {dc, "c_tmp = exp(c_tmp)"});
            auto runningSumScale = popops::sub(graph, runningMaxsi, newMaxs, qLoopProg, {dc, "c_i = m_i - m_new"});
            popops::expInPlace(graph, runningSumScale, qLoopProg, {dc, "c_i = exp(c_i)"});

            // compute running sum update
            auto newSums = popops::reduce(graph, t, t.elementType(), {2}, {popops::Operation::ADD}, qLoopProg, {dc, "s_new = sum(attn_ij, dim=2)"});
            
            // scale updates from past statistics
            popops::mulInPlace(graph, newSums, tmpSumScale, qLoopProg, {dc, "s_new *= c_tmp"});
            popops::mulInPlace(graph, runningSumsi, runningSumScale, qLoopProg, {dc, "s_i *= c_i"});
            popops::addInPlace(graph, newSums, runningSumsi, qLoopProg, {dc, "s_new += s_i"});

            // compute 
            popops::mulInPlace(graph, oi, runningSumsi.expand({2}), qLoopProg, {dc, "o_i *= s_i"});
            popops::mulInPlace(graph, t, tmpSumScale.expand({2}), qLoopProg, {dc, "attn_ij *= c_tmp"});
            poplin::matMulGroupedAcc(graph, oi, onef, t, vj, qLoopProg, {dc, "oi += attn_ij @ v_j"});
            auto invNewSums = popops::inv(graph, newSums.expand({2}), qLoopProg, {dc, "s_inv = 1 / s_new"});
            popops::mulInPlace(graph, oi, invNewSums, qLoopProg, {dc, "o_i *= s_inv"});

            // update output and running softmax stats
            popops::dynamicUpdate(graph, out, oi.expand({0}), qCounter, {0}, {1}, qLoopProg, {dc, "o = o.at[i].set(o_i)"});
            popops::dynamicUpdate(graph, runningMaxs, newMaxs.expand({0}), qCounter, {0}, {1}, qLoopProg, {dc, "m = m.at[i].set(m_new)"});
            popops::dynamicUpdate(graph, runningSums, newSums.expand({0}), qCounter, {0}, {1}, qLoopProg, {dc, "s = s.at[i].set(s_new)"});
            
            // update q loop counter
            popops::addInPlace(graph, qCounter, oneu, qLoopProg, {dc, "i+=1"});
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

    auto out_v = vanillaAttention(graph, qkv, prog, {dc, "vanilla_attention"});
    auto out_s = serialisedAttention(graph, qkv, 4, 4, prog, {dc, "serialised_attention"});

    auto err = popops::sub(graph, out_v, out_s, prog, "e = x - y");
    popops::absInPlace(graph, err, prog, "e = abs(e)");
    auto maxErr = popops::reduce(graph, err, err.elementType(), {0, 1, 2}, {popops::Operation::MAX}, prog, "m = max(e)");
    prog.add(program::PrintTensor("maxErr", maxErr));

    Engine engine(graph, prog, {{"debug.instrument", "true"}});
    engine.load(device);
    engine.run(0);
    return 0;

}