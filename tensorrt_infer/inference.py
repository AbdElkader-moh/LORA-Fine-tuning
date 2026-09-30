"""Runs inference on the test set using a built TensorRT engine.

Requires `tensorrt` and `pycuda` plus a CUDA-capable GPU. Not available in a
plain CPU/dev environment - run this wherever TensorRT is installed.
"""
import numpy as np
import pycuda.autoinit  # noqa: F401  (creates/attaches the CUDA context)
import pycuda.driver as cuda
import tensorrt as trt
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class TensorRTInference:
    """Loads a serialized TensorRT engine and runs inference through it."""

    def __init__(self, engine_path):
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)
        self.stream = cuda.Stream()

    def predict_logits(self, pixel_values: np.ndarray) -> np.ndarray:
        pixel_values = np.ascontiguousarray(pixel_values.astype(np.float32))

        self.context.set_input_shape(self.input_name, pixel_values.shape)
        output_shape = tuple(self.context.get_tensor_shape(self.output_name))
        output = np.empty(output_shape, dtype=np.float32)

        d_input = cuda.mem_alloc(pixel_values.nbytes)
        d_output = cuda.mem_alloc(output.nbytes)
        try:
            self.context.set_tensor_address(self.input_name, int(d_input))
            self.context.set_tensor_address(self.output_name, int(d_output))

            cuda.memcpy_htod_async(d_input, pixel_values, self.stream)
            self.context.execute_async_v3(stream_handle=self.stream.handle)
            cuda.memcpy_dtoh_async(output, d_output, self.stream)
            self.stream.synchronize()
        finally:
            d_input.free()
            d_output.free()

        return output

    def evaluate(self, loader, criterion=None):
        criterion = criterion or nn.CrossEntropyLoss()
        running_loss = 0.0
        n_samples = 0
        all_preds, all_labels = [], []

        for images, labels in loader:
            logits = self.predict_logits(images.numpy())
            logits_t = torch.from_numpy(logits)
            loss = criterion(logits_t, labels)
            running_loss += loss.item() * images.size(0)
            n_samples += images.size(0)
            all_preds.extend(logits_t.argmax(dim=1).numpy())
            all_labels.extend(labels.numpy())

        avg_loss = running_loss / n_samples
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="macro")
        return avg_loss, acc, f1
