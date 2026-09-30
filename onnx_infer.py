import argparse

from onnx_runtime_infer.inference import ONNXRuntimeInference
from vit_lora.config import load_config
from vit_lora.data import EuroSATData
from vit_lora.processor import ImageProcessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--onnx-model", default="onnx_models/vit_lora_best.onnx")
    args = parser.parse_args()

    config = load_config(args.config)
    processor = ImageProcessor(config.model.name)
    data = EuroSATData(config, processor).prepare()
    _, _, test_loader = data.get_loaders(config.training.batch_size, config.training.num_workers)

    runner = ONNXRuntimeInference(args.onnx_model)
    loss, acc, f1 = runner.evaluate(test_loader)
    print(f"[ONNX Runtime] test loss={loss:.4f}  accuracy={acc:.4f}  macro-F1={f1:.4f}")


if __name__ == "__main__":
    main()
