## Installation

For **inference**, install:

```shell
pip install fast-alpr[onnx-gpu]
```

???+ warning
    By default, **no ONNX runtime is installed**.

    To run inference, you **must install** one of the ONNX extras:

    - `onnx` - for CPU inference (cross-platform)
    - `onnx-gpu` - for NVIDIA GPUs (CUDA)
    - `onnx-openvino` - for Intel CPUs / VPUs
    - `onnx-directml` - for Windows devices via DirectML
    - `onnx-qnn` - for Qualcomm chips on mobile

Dependencies for inference are kept **minimal by default**. Inference-related packages like **ONNX runtimes** are
**optional** and not installed unless **explicitly requested via extras**.
