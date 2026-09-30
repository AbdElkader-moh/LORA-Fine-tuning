import argparse
from pathlib import Path

from tensorrt_infer.build_engine import build_engine
from tensorrt_infer.inference import TensorRTInference
from vit_lora.config import load_config
from vit_lora.data import EuroSATData
from vit_lora.processor import ImageProcessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--onnx-model", default="onnx_models/vit_lora_best.onnx")
    parser.add_argument("--engine", default="onnx_models/vit_lora_best.trt")
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the engine even if it already exists")
    args = parser.parse_args()

    config = load_config(args.config)

    engine_path = Path(args.engine)
    if args.rebuild or not engine_path.exists():
        build_engine(args.onnx_model, engine_path, fp16=not args.no_fp16,
                     max_batch_size=config.training.batch_size)

    processor = ImageProcessor(config.model.name)
    data = EuroSATData(config, processor).prepare()
    _, _, test_loader = data.get_loaders(config.training.batch_size, config.training.num_workers)

    runner = TensorRTInference(engine_path)
    loss, acc, f1 = runner.evaluate(test_loader)
    print(f"[TensorRT] test loss={loss:.4f}  accuracy={acc:.4f}  macro-F1={f1:.4f}")


if __name__ == "__main__":
    main()
