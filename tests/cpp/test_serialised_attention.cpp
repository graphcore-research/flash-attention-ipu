#include <catch_amalgamated.hpp>
#include <iostream>

#include <poplar/Engine.hpp>
#include <poplar/Device.hpp>
#include <poplar/DeviceManager.hpp>
#include <poplar/CycleCount.hpp>
#include <poplar/DebugContext.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>

#include <poprand/RandomGen.hpp>
#include <popops/Reduce.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Reduce.hpp>
#include <popops/Cast.hpp>
#include <poputil/TileMapping.hpp>

#include <poplin/codelets.hpp>
#include <popops/codelets.hpp>
#include <popnn/codelets.hpp>
#include <poprand/codelets.hpp>

#include "cpp/serialised_attention.hpp"

using namespace poplar;
using namespace poplar::program;

namespace {

    float compare(
        uint32_t numGroups,
        uint32_t sequenceLength,
        uint32_t hiddenDim,
        uint32_t qChunks,
        uint32_t kvChunks,
        std::vector<uint32_t> seed,
        const poplar::Type& type 
        ) {
    
        poplar::Device device;

        auto manager = poplar::DeviceManager::createDeviceManager();
        auto devices = manager.getDevices(poplar::TargetType::IPU, 1);
        
        auto it = std::find_if(devices.begin(), devices.end(), [](Device &device){return device.attach();});

        if (it == devices.end()) {
            std::cerr << "Error attaching to device \n";
        }
        device = std::move(*it);
        
        poplar::Graph graph(device.getTarget());

        popops::addCodelets(graph);
        poplin::addCodelets(graph);
        popnn::addCodelets(graph);
        poprand::addCodelets(graph);

        poplar::Tensor qkv = graph.addVariable(type, {3, numGroups, sequenceLength, hiddenDim});

        poputil::mapTensorLinearly(graph, qkv);

        const poplar::Tensor seedTensor = graph.addConstant<uint32_t>(poplar::UNSIGNED_INT, {2}, seed);
        poputil::mapTensorLinearly(graph, seedTensor);

        poplar::DebugContext dc;

        poplar::program::Sequence prog;

        qkv = poprand::normal(graph, &seedTensor, 0, qkv, qkv.elementType(), 0.0, 1.0, prog);

        poplar::program::Sequence vanillaAttentionProg;
        poplar::program::Sequence serialisedAttentionProg;

        auto out_v = vanillaAttention(graph, qkv, vanillaAttentionProg, {dc, "vanilla_attention"});
        auto out_s = serialisedAttention(graph, qkv, qChunks, kvChunks, serialisedAttentionProg, {dc, "serialised_attention"});

        prog.add(vanillaAttentionProg);
        prog.add(serialisedAttentionProg);

        auto err = popops::sub(graph, out_v, out_s, prog, "e = x - y");
        popops::absInPlace(graph, err, prog, "e = abs(e)");
        auto maxErr = popops::reduce(graph, err, err.elementType(), {0, 1, 2}, {popops::Operation::MAX}, prog, "m = max(e)");
        maxErr = popops::cast(graph, maxErr, poplar::FLOAT, prog, "m.as(float32)");

        auto maxErrFifo = graph.addDeviceToHostFIFO("maxErr", maxErr.elementType(), maxErr.numElements());
        prog.add(Copy(maxErr, maxErrFifo, false, {"maxErr_d2h"}));
        
        poplar::Engine engine(graph, prog, {{"debug.instrument", "true"}});
        engine.load(device);

        auto maxErrHost = std::vector<float>(maxErr.numElements());
        engine.connectStream("maxErr", maxErrHost.data());

        engine.run(0);

        return maxErrHost[0];
    }

    void benchmark(
        uint32_t numGroups,
        uint32_t sequenceLength,
        uint32_t hiddenDim,
        uint32_t qChunks,
        uint32_t kvChunks,
        std::vector<uint32_t> seed,
        const poplar::Type& type
    ){

        poplar::Device device;
        auto manager = poplar::DeviceManager::createDeviceManager();
        auto devices = manager.getDevices(poplar::TargetType::IPU, 1);
        
        auto it = std::find_if(devices.begin(), devices.end(), [](Device &device){return device.attach();});

        if (it == devices.end()) {
            std::cerr << "Error attaching to device \n";
        }
        device = std::move(*it);
        
        poplar::Graph graph(device.getTarget());

        popops::addCodelets(graph);
        poplin::addCodelets(graph);
        popnn::addCodelets(graph);
        poprand::addCodelets(graph);


        poplar::Tensor qkv = graph.addVariable(type, {3, numGroups, sequenceLength, hiddenDim});

        poputil::mapTensorLinearly(graph, qkv);

        const poplar::Tensor seedTensor = graph.addConstant<uint32_t>(poplar::UNSIGNED_INT, {2}, seed);
        poputil::mapTensorLinearly(graph, seedTensor);

        poplar::DebugContext dc;

        poplar::program::Sequence prog;

        qkv = poprand::normal(graph, &seedTensor, 0, qkv, qkv.elementType(), 0.0, 1.0, prog);

        poplar::program::Sequence serialisedAttentionProg;
        auto out = serialisedAttention(graph, qkv, qChunks, kvChunks, serialisedAttentionProg, {dc, "serialised_attention"});
        auto cycles = poplar::cycleCount(graph, serialisedAttentionProg, 0, poplar::SyncType::EXTERNAL, {dc, "count cycles"});
        prog.add(serialisedAttentionProg);

        auto cyclesFifo = graph.addDeviceToHostFIFO("cycles", cycles.elementType(), cycles.numElements());
        prog.add(Copy(cycles, cyclesFifo, false, {"cycles_d2h"}));
        
        poplar::Engine engine(graph, prog, {{"debug.instrument", "true"}});
        engine.load(device);

        auto cyclesHost = std::vector<uint32_t>(cycles.numElements());
        engine.connectStream("cycles", cyclesHost.data());

        engine.run(0);

        std::ostringstream str;

        str << "Cycles= " << cyclesHost[0] << " (for: " << type.toString() << " QKV[" << numGroups << ", " << sequenceLength << ", " << hiddenDim << "], "
            << "Chunks[" << qChunks << ", " << kvChunks << "])\n";

        std::cerr << str.str();
    }


    TEST_CASE("compare vanillaAttention vs serialisedAttention output", "[attention]") {
        SECTION("float32 4x6x6 test cases"){
            REQUIRE(compare(4, 6, 2, 1, 1, {40, 90}, poplar::FLOAT) <= 1e-5);
            REQUIRE(compare(4, 6, 2, 1, 2, {40, 90}, poplar::FLOAT) <= 1e-5);
            REQUIRE(compare(4, 6, 2, 2, 1, {40, 90}, poplar::FLOAT) <= 1e-5);
            REQUIRE(compare(4, 6, 2, 2, 2, {40, 90}, poplar::FLOAT) <= 1e-5);
            REQUIRE(compare(4, 6, 2, 2, 3, {40, 90}, poplar::FLOAT) <= 1e-5);
            REQUIRE(compare(4, 6, 2, 3, 2, {40, 90}, poplar::FLOAT) <= 1e-5);
            REQUIRE(compare(4, 6, 2, 3, 3, {40, 90}, poplar::FLOAT) <= 1e-5);
        }

        // Medium
        SECTION("float32 8x256x256 test cases"){
            REQUIRE(compare(8, 256, 128, 2, 2, {40, 90}, poplar::FLOAT) <= 1e-5);
            REQUIRE(compare(8, 256, 128, 2, 4, {40, 90}, poplar::FLOAT) <= 1e-5);
            REQUIRE(compare(8, 256, 128, 4, 2, {40, 90}, poplar::FLOAT) <= 1e-5);
            REQUIRE(compare(8, 256, 128, 4, 4, {40, 90}, poplar::FLOAT) <= 1e-5);
        }

        // Large
        SECTION("float16 8x2048x2048 test cases"){
            REQUIRE(compare(8, 2048, 128, 2, 2, {40, 90}, poplar::HALF) <= 1e-2);
            REQUIRE(compare(8, 2048, 128, 2, 4, {40, 90}, poplar::HALF) <= 1e-2);
            REQUIRE(compare(8, 2048, 128, 4, 2, {40, 90}, poplar::HALF) <= 1e-2);
            REQUIRE(compare(8, 2048, 128, 4, 4, {40, 90}, poplar::HALF) <= 1e-2);
        }
    }

    TEST_CASE("benchmark serialisedAttention performance", "[attentionperf]") {
        SECTION("float32 8x256x256 cases"){
            benchmark(8, 256, 128, 2, 2, {40, 90}, poplar::FLOAT);
            benchmark(8, 256, 128, 2, 4, {40, 90}, poplar::FLOAT);
            benchmark(8, 256, 128, 4, 2, {40, 90}, poplar::FLOAT);
            benchmark(8, 256, 128, 4, 4, {40, 90}, poplar::FLOAT);
        }

        SECTION("float16 8x2048x256 cases"){
            benchmark(8, 2048, 128, 2, 2, {40, 90}, poplar::HALF);
            benchmark(8, 2048, 128, 2, 4, {40, 90}, poplar::HALF);
            benchmark(8, 2048, 128, 4, 2, {40, 90}, poplar::HALF);
            benchmark(8, 2048, 128, 4, 4, {40, 90}, poplar::HALF);
        }
    }
} // namespace