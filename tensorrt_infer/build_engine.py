"""Builds a TensorRT engine from an ONNX model.

Requires the `tensorrt` Python package and a CUDA-capable GPU with TensorRT
installed. Not available in a plain CPU/dev environment - run this on a
machine with TensorRT set up (e.g. an NVIDIA GPU box, a TensorRT-enabled
Kaggle/Colab image, or a Jetson).
"""
import argparse
from pathlib import Path

import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def build_engine(onnx_path, engine_path, fp16=True, max_batch_size=32, workspace_gb=4):
    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError(f"Failed to parse ONNX model: {onnx_path}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30))
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # The ONNX model was exported with a dynamic batch dimension - give the
    # builder a concrete range of shapes to optimize for.
    input_tensor = network.get_input(0)
    _, c, h, w = input_tensor.shape
    profile = builder.create_optimization_profile()
    profile.set_shape(input_tensor.name, (1, c, h, w), (max(1, max_batch_size // 2), c, h, w), (max_batch_size, c, h, w))
    config.add_optimization_profile(profile)

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("TensorRT engine build failed")

    engine_path = Path(engine_path)
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    print(f"Saved TensorRT engine to {engine_path}")
    return engine_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx-model", default="onnx_models/vit_lora_best.onnx")
    parser.add_argument("--engine", default="onnx_models/vit_lora_best.trt")
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--max-batch-size", type=int, default=32)
    args = parser.parse_args()
    build_engine(args.onnx_model, args.engine, fp16=not args.no_fp16, max_batch_size=args.max_batch_size)


if __name__ == "__main__":
    main()
