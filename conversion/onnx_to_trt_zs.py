import argparse
import tensorrt as trt
import os

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)


def build_engine(
    onnx_file_path,
    opt_size,
    min_size,
    max_size,
    use_fp16,
    use_sparse,
    disable_timing_cache,
    gpu_fallback,
    debug_mode,
    workspace_size
):
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(EXPLICIT_BATCH) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser, \
         builder.create_builder_config() as config:

        # Precision flag
        if use_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        else:
            config.clear_flag(trt.BuilderFlag.FP16)

        # Sparse weights flag
        if use_sparse:
            config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
        else:
            config.clear_flag(trt.BuilderFlag.SPARSE_WEIGHTS)

        # Disable timing cache
        if disable_timing_cache:
            config.set_flag(trt.BuilderFlag.DISABLE_TIMING_CACHE)
        else:
            config.clear_flag(trt.BuilderFlag.DISABLE_TIMING_CACHE)

        # GPU fallback for INT8
        if gpu_fallback:
            config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
        
        # Debug mode
        if debug_mode:
            config.set_flag(trt.BuilderFlag.DEBUG)

        # Workspace limit
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE,
            workspace_size
        )

        # Tactic sources
        config.set_tactic_sources(
            1 << int(trt.TacticSource.CUBLAS) |
            1 << int(trt.TacticSource.CUDNN)
        )

        # Parse ONNX
        print(f"🔍 Parsing ONNX model: {onnx_file_path}")
        if not parser.parse_from_file(onnx_file_path):
            print("❌ Failed to parse ONNX model")
            for i in range(parser.num_errors):
                print(f"   ▶ {parser.get_error(i)}")
            return None

        # Dynamic shape profile
        input_tensor = network.get_input(0)
        profile = builder.create_optimization_profile()
        profile.set_shape(
            input_tensor.name,
            (1, 3, min_size, min_size),    # MIN
            (1, 3, opt_size, opt_size),    # OPTIMAL
            (1, 3, max_size, max_size)     # MAX
        )
        config.add_optimization_profile(profile)

        # Build engine
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            print("❌ Failed to serialize TRT engine")
            return None
        runtime = trt.Runtime(TRT_LOGGER)
        return runtime.deserialize_cuda_engine(serialized)


def save_engine(engine, engine_file_path):
    with open(engine_file_path, "wb") as f:
        f.write(engine.serialize())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build dynamic TRT engine with optimization flags")
    parser.add_argument("--weights", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--fp16", dest="fp16", action="store_true", default=True, help="Enable FP16 precision")
    parser.add_argument("--no-fp16", dest="fp16", action="store_false", help="Disable FP16 precision")
    parser.add_argument("--sparse", dest="sparse", action="store_true", default=True, help="Enable sparse weights tactics")
    parser.add_argument("--no-sparse", dest="sparse", action="store_false", help="Disable sparse weights tactics")
    parser.add_argument("--disable-timing-cache", action="store_true", default=False, help="Disable timing cache")
    parser.add_argument("--gpu-fallback", action="store_true", default=False, help="Allow GPU fallback for INT8")
    parser.add_argument("--debug", action="store_true", default=False, help="Enable debug mode")
    parser.add_argument("--workspace", type=int, default=1 << 29, help="Workspace size in bytes")
    args = parser.parse_args()

    # Hardcoded sizes
    OPT_SIZE = 480
    MIN_SIZE = 256
    MAX_SIZE = 910

    # Paths
    tag = os.path.basename(args.weights).split('_')[0]
    onnx_path = f"models/onnx_engines/lseg_img_enc_rn101_{tag}.onnx"

    # Build flag suffix
    flags = [
        "fp16" if args.fp16 else "fp32",
        "sparse" if args.sparse else None,
        "noTC" if args.disable_timing_cache else None,
        "gpuFB" if args.gpu_fallback else None,
        "dbg" if args.debug else None,
        f"ws{args.workspace>>20}MiB"
    ]
    flags = [f for f in flags if f]
    suffix = "_".join(flags)

    # Engine output
    engine_basename = f"lseg_img_enc_rn101_{tag}_{OPT_SIZE}__{suffix}.trt"
    engine_dir = os.path.join("outputs", "models")
    os.makedirs(engine_dir, exist_ok=True)
    engine_path = os.path.join(engine_dir, engine_basename)

    # Build and save
    engine = build_engine(
        onnx_path,
        OPT_SIZE,
        MIN_SIZE,
        MAX_SIZE,
        args.fp16,
        args.sparse,
        args.disable_timing_cache,
        args.gpu_fallback,
        args.debug,
        args.workspace
    )
    if engine:
        save_engine(engine, engine_path)
        print(f"\n✅ Engine saved as: {engine_path}")
    else:
        print("❌ TensorRT engine build failed.")
