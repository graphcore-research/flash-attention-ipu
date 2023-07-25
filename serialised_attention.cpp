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
    const poplar::Tensor& qkv,
    poplar::program::Sequence& prog,
    const poplar::DebugContext& dc) {

    // get shape data
    auto groups = qkv.dim(1); // groups = batch_size * num_heads
    auto seqLen = qkv.dim(2);
    auto headDim = qkv.dim(3);

    // q, k, v = qkv
    auto query = qkv.slice({0, 1}, 0).reshape({groups, seqLen, headDim});
    auto key = qkv.slice({1, 2}, 0).reshape({groups, seqLen, headDim});
    auto value = qkv.slice({2, 3}, 0).reshape({groups, seqLen, headDim});
    
    // q @ k.T
    auto attn = poplin::matMulGrouped(graph, query, key.dimShuffle({0, 2, 1}), prog, query.elementType(), {dc, "qk_matmul"});
    
    // Stable softmax(x)
    auto m = popops::reduce(graph, attn, attn.elementType(), {2}, {popops::Operation::MAX}, prog, {dc, "softmax_max"});
    popops::subInPlace(graph, attn, m.expand({2}), prog, {dc, "softmax_sub"});
    popops::expInPlace(graph, attn, prog, {dc, "softmax_exp"});
    auto s = popops::reduce(graph, attn, attn.elementType(), {2}, {popops::Operation::ADD}, prog, {dc, "softmax_sum"});
    popops::invInPlace(graph, s, prog, {dc, "softmax_inv"});
    popops::mulInPlace(graph, attn, s.expand({2}), prog, {dc, "softmax_mul"});

    // attn @ v
    auto out = poplin::matMulGrouped(graph, attn, value, prog, value.elementType(), {dc, "attn_matmul"});
    
    return out;
}

poplar::Tensor serialisedAttention(
    poplar::Graph& graph, 
    const poplar::Tensor& qkv, 
    const uint32_t& num_chunks_q, 
    const uint32_t& num_chunks_kv,
    poplar::program::Sequence& prog,
    const poplar::DebugContext& dc) {

    // get shape data
    auto groups = qkv.dim(1); // groups = batch_size * num_heads
    auto seqLen = qkv.dim(2);
    auto headDim = qkv.dim(3);

    // compute sizes of chunked sequence length
    auto chunkedQueryLen = seqLen / num_chunks_q; 
    auto chunkedKVLen = seqLen / num_chunks_kv; 

    // Unpack q,k,v and copy data to sliceable tensors with nice tile mappings
    auto query = popops::createSliceableTensor(graph, qkv.elementType(), {num_chunks_q, groups, chunkedQueryLen, headDim}, {0}, {1}, 4, {dc, "create_query"});
    auto key = popops::createSliceableTensor(graph, qkv.elementType(), {num_chunks_kv, groups, chunkedKVLen, headDim}, {0}, {1}, 4, {dc, "create_key"});
    auto value = popops::createSliceableTensor(graph, qkv.elementType(), {num_chunks_kv, groups, chunkedKVLen, headDim}, {0}, {1}, 4, {dc, "create_value"});

    prog.add(Copy(qkv.slice({0, 1}, 0).reshape({groups, num_chunks_q, chunkedQueryLen, headDim}).dimShuffle({1, 0, 2, 3}), query));
    prog.add(Copy(qkv.slice({1, 2}, 0).reshape({groups, num_chunks_kv, chunkedQueryLen, headDim}).dimShuffle({1, 0, 2, 3}), key));
    prog.add(Copy(qkv.slice({2, 3}, 0).reshape({groups, num_chunks_kv, chunkedQueryLen, headDim}).dimShuffle({1, 0, 2, 3}), value));

    // create output tensor computed iteratively
    auto out = graph.clone(query, {dc, "create_output"});
    popops::zero(graph, out, prog, {dc, "zero_output"});

    // create tensors to store running softmax statistics
    auto runningSums = popops::createSliceableTensor(graph, query.elementType(), {num_chunks_q, groups, chunkedQueryLen}, {0}, {1}, 4, {dc, "create_running_sum"});
    auto runningMaxs = popops::createSliceableTensor(graph, query.elementType(), {num_chunks_q, groups, chunkedQueryLen}, {0}, {1}, 4, {dc, "create_running_maxs"});

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
        Sequence kvLoopProg;
        // slice k and v
        auto kj = popops::dynamicSlice(graph, key, kvCounter, {0}, {1}, kvLoopProg, {dc, "dynamic_slice_k"}).squeeze({0});
        auto vj = popops::dynamicSlice(graph, value, kvCounter, {0}, {1}, kvLoopProg, {dc, "dynamic_slice_v"}).squeeze({0});

            // q loop body program
            Sequence qLoopProg;
            // slice q, output, and softmax running stats
            auto qi = popops::dynamicSlice(graph, query, qCounter, {0}, {1}, qLoopProg, {dc, "dynamic_slice_q"}).squeeze({0}); 
            auto oi = popops::dynamicSlice(graph, out, qCounter, {0}, {1}, qLoopProg, {dc, "dynamic_slice_out"}).squeeze({0});             
            
            auto runningMaxsi = popops::dynamicSlice(graph, runningMaxs, qCounter, {0}, {1}, qLoopProg, {dc, "dynamic_slice_runningmax"}).squeeze({0}); 
            auto runningSumsi = popops::dynamicSlice(graph, runningSums, qCounter, {0}, {1}, qLoopProg, {dc, "dynamic_slice_runningmax"}).squeeze({0}); 

            // compute qk^T
            auto t = poplin::matMulGrouped(graph, qi, kj.dimShuffle({0, 2, 1}), qLoopProg, kj.elementType(), {dc, "qk_matmul"});
            
            // compute qk^T max for stable softmax
            auto tmpMaxs = popops::reduce(graph, t, t.elementType(), {2}, {popops::Operation::MAX}, qLoopProg, {dc, "softmax_tmp_max"});
            // subtract max from qk^T
            popops::subInPlace(graph, t, tmpMaxs.expand({2}), qLoopProg, {dc, "softmax_tmp_sub"});
            // compute softmax numerator: exp(qk^T - max)
            popops::expInPlace(graph, t, qLoopProg, {dc, "softmax_tmp_exp"});
            
            // compute running max update
            auto newMaxs = popops::max(graph, runningMaxsi, tmpMaxs, qLoopProg, {dc, "softmax_new_max"});
            
            // compute softmax update scaling factors
            auto tmpSumScale = popops::sub(graph, tmpMaxs, newMaxs, qLoopProg, {dc, "softmax_tmp_scale_sub"});
            popops::expInPlace(graph, tmpSumScale, qLoopProg, {dc, "softmax_tmp_scale_exp"});
            auto runningSumScale = popops::sub(graph, runningMaxsi, newMaxs, qLoopProg, {dc, "softmax_running_scale_sub"});
            popops::expInPlace(graph, runningSumScale, qLoopProg, {dc, "softmax_running_scale_exp"});

            // compute running sum update
            auto newSums = popops::reduce(graph, t, t.elementType(), {2}, {popops::Operation::ADD}, qLoopProg, {dc, "softmax_tmp_sum"});
            
            // scale updates from past statistics
            popops::mulInPlace(graph, newSums, tmpSumScale, qLoopProg, {dc, "softmax_tmp_scale_mul"});
            popops::mulInPlace(graph, runningSumsi, runningSumScale, qLoopProg, {dc, "softmax_running_scale_mul"});
            popops::addInPlace(graph, newSums, runningSumsi, qLoopProg, {dc, "softmax_new_sum"});

            // compute 
            popops::mulInPlace(graph, oi, runningSumsi.expand({2}), qLoopProg, {dc, "output_scale"});
            popops::mulInPlace(graph, t, tmpSumScale.expand({2}), qLoopProg, {dc, "tmp_scale"});
            poplin::matMulGroupedAcc(graph, oi, onef, t, vj, qLoopProg, {dc, "unscaled_attn_matmul"});
            auto invNewSums = popops::inv(graph, newSums.expand({2}), qLoopProg, {dc, "softmax_inv_sum"});
            popops::mulInPlace(graph, oi, invNewSums, qLoopProg, {dc, "attn_matmul_inv_sum_scale"});

            // update output and running softmax stats
            popops::dynamicUpdate(graph, out, oi.expand({0}), qCounter, {0}, {1}, qLoopProg, {dc, "dynamic_update_out"});
            popops::dynamicUpdate(graph, runningMaxs, newMaxs.expand({0}), qCounter, {0}, {1}, qLoopProg, {dc, "dynamic_update_running_max"});
            popops::dynamicUpdate(graph, runningSums, newSums.expand({0}), qCounter, {0}, {1}, qLoopProg, {dc, "dynamic_update_running_sum"});
            
            // update q loop counter
            popops::addInPlace(graph, qCounter, oneu, qLoopProg, {dc, "increment_qCounter"});

        // repeat q loop body in kv loop body
        kvLoopProg.add(Repeat(query.dim(0), qLoopProg, {dc, "serialised_attention_inner_loop_repeat"}));
        // update kv loop counter
        popops::addInPlace(graph, kvCounter, oneu, kvLoopProg, {dc, "increment_kvCounter"});
        // reset q loop counter
        popops::zero(graph, qCounter, kvLoopProg, {dc, "zero_qCounter"});

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

    auto err = popops::sub(graph, out_v, out_s, prog, "maxabserr_sub");
    popops::absInPlace(graph, err, prog, "maxabserr_abs");
    auto maxErr = popops::reduce(graph, err, err.elementType(), {0, 1, 2}, {popops::Operation::MAX}, prog, "maxabserr_max");
    prog.add(program::PrintTensor("maxErr", maxErr));

    Engine engine(graph, prog, {{"debug.instrument", "true"}});
    engine.load(device);
    engine.run(0);
    return 0;

}