# KD-AFA-Net ONNX Runtime Custom Operator

This repository provides an ONNX Runtime custom operator implementation for KD-AFA-Net, especially for the FFT-based Adaptive Frequency Attention (AFA) module.

KD-AFA-Net uses an AFA module that performs frequency-domain processing with `torch.fft.fft2`, `fftshift`, magnitude and phase decomposition, low/high-frequency attention, complex reconstruction, and inverse FFT. However, the default PyTorch ONNX exporter does not directly support `torch.fft.fft2`. Because of this, exporting the original PyTorch model to ONNX fails with an unsupported operator error such as `aten::fft_fft2`.

To solve this issue, the AFA module is exported as a custom ONNX operator named `kdafa::AFA`. Then, the custom operator is implemented in two stages: CPU custom op and CUDA/cuFFT custom op. This allows the original AFA operation to be preserved without removing or approximating the FFT-based module.

---

## Motivation

The original AFA module contains FFT-based operations, including FFT2, frequency shifting, magnitude and phase decomposition, low/high-frequency split, attention-based magnitude refinement, complex reconstruction, inverse shift, and IFFT2.

The default PyTorch ONNX exporter cannot convert `torch.fft.fft2` into a standard ONNX graph. Therefore, the AFA module is wrapped with a custom `torch.autograd.Function`, and its ONNX symbolic function exports the entire module as a custom ONNX operator.

Overall concept:

```text
PyTorch AFA module
→ kdafa::AFA custom ONNX node
→ ONNX Runtime custom op implementation
```

---

## Overall Workflow
```Default ONNX export fails
→ Wrap AFA as kdafa::AFA custom ONNX operator
→ Export ONNX graph with custom AFA nodes
→ Implement CPU custom op
→ Verify PyTorch vs ONNX Runtime CPU output
→ Implement CUDA/cuFFT custom op
→ Verify PyTorch vs ONNX Runtime CUDA output
→ Remove unnecessary decoder Pad for fixed input size
→ Confirm all major operators run on CUDAExecutionProvider
```

Final acceleration result:
- CPU custom op / hybrid execution: about 42 ms
- CUDA custom op with AFA on GPU: about 16 ms
- CUDA custom op with Pad removed: about 5.97 ms

---

## Environment

The implementation was tested with the following environment.

```
OS: Windows
IDE: PyCharm
Build shell: x64 Native Tools Command Prompt for VS 2022
CUDA Toolkit: 12.4
ONNX Runtime Python package: 1.23.2
ONNX Runtime C++ package: 1.23.2
```

The ONNX Runtime C++ package version must match the Python ONNX Runtime version.

For example, if Python uses:

onnxruntime-gpu==1.23.2

then the C++ package should also be:

onnxruntime-win-x64-1.23.2

---

## ONNX Export with Custom AFA Operator

The AFA module is wrapped with a custom autograd function. During normal PyTorch inference, the original FFT-based AFA operation is executed. During ONNX export, the symbolic function exports the AFA module as kdafa::AFA.

The ONNX export should include the custom domain:

```
custom_opsets={"kdafa": 1}
```

After export, the ONNX graph should contain:

```
[ONNX opset imports]
domain='', version=19
domain='kdafa', version=1

[Custom nodes]
kdafa::AFA
kdafa::AFA
kdafa::AFA
```

The three kdafa::AFA nodes correspond to the three AFA modules used at skip connections.

---

## CPU Custom Op Build

The CPU custom op is implemented using ONNX Runtime C++ API and pocketfft.

Required files:

```
custom_ops/kdafa_ort
├─ CMakeLists.txt
├─ kdafa_afa_op.cpp
└─ pocketfft_hdronly.h
```

Use x64 Native Tools Command Prompt for VS 2022.

Move to the CPU custom op directory:

```
cd <PROJECT_ROOT>\custom_ops\kdafa_ort
```

Configure and build:

```
cmake -S . -B build -A x64 -DORT_ROOT=<ORT_ROOT>
cmake --build build --config Release
```

Example:

```
cmake -S . -B build -A x64 -DORT_ROOT=C:/onnxruntime-win-x64-1.23.2
cmake --build build --config Release
```

After a successful build, the DLL is generated at:

```
<PROJECT_ROOT>\custom_ops\kdafa_ort\build\Release\kdafa_custom_ops.dll
```

For clarity, the CPU custom op DLL can be copied or renamed as:
```
kdafa_custom_ops_cpu.dll
```

---

## CPU Custom Op Verification

The custom op DLL is registered in Python through SessionOptions.register_custom_ops_library().

The CPU custom op produced the following output comparison result:

```
[Compare Torch vs ONNXRuntime-Custom]
mean abs error: 8.947609e-09
max abs error : 2.9802322e-08
torch out min/max: -0.060895726 0.3894014
ort out min/max  : -0.060895734 0.38940138
```

This shows that the original PyTorch AFA operation and the ONNX Runtime CPU custom op produce nearly identical outputs.

Summary:

```
The default PyTorch ONNX exporter cannot directly export torch.fft.fft2.
To preserve the original model, the AFA module was exported as kdafa::AFA.
Then, a C++ ONNX Runtime custom op was implemented.
The output difference between PyTorch and ONNX Runtime was about 1e-8.
```
---

## CUDA Custom Op Build

The CUDA custom op is implemented using CUDA kernels and cuFFT.

Required files:

```
custom_ops/kdafa_ort_cuda
├─ CMakeLists.txt
└─ kdafa_afa_cuda_op.cu
```

The CUDA implementation uses cuFFT for FFT2/IFFT2 and CUDA kernels for fftshift, magnitude, phase, low/high-frequency split, 1x1 attention, softmax, and complex reconstruction.

Use x64 Native Tools Command Prompt for VS 2022.

Move to the CUDA custom op directory:

```
cd <PROJECT_ROOT>\custom_ops\kdafa_ort_cuda
```

Configure with Visual Studio 2022 and CUDA 12.4:

```
cmake -S . -B build ^
-G "Visual Studio 17 2022" ^
-A x64 ^
-T cuda="<CUDA_ROOT>" ^
-DORT_ROOT=<ORT_ROOT>
```

Example:

```
cmake -S . -B build ^
-G "Visual Studio 17 2022" ^
-A x64 ^
-T cuda="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4" ^
-DORT_ROOT=C:/onnxruntime-win-x64-1.23.2
```

Build:

```
cmake --build build --config Release
```

After a successful build, the CUDA custom op DLL is generated at:

```
<PROJECT_ROOT>\custom_ops\kdafa_ort_cuda\build\Release\kdafa_custom_ops_cuda.dll
```

If CMake reports No CUDA toolset found, explicitly specify the Visual Studio 2022 generator and CUDA Toolkit path as shown above.

---

## CUDA Custom Op Verification

The CUDA custom op DLL is registered in Python through SessionOptions.register_custom_ops_library() and executed with CUDAExecutionProvider.

The CUDA custom op produced the following output comparison result:

```
[Compare Torch vs ONNXRuntime-CUDA-Custom]
mean abs error: 1.0035971e-08
max abs error : 4.4703484e-08
torch out min/max: -0.060294665 0.11553874
ort out min/max  : -0.060294673 0.11553873
```

Another test showed:

```
[Compare Torch vs ONNXRuntime-CUDA-Custom]
mean abs error: 2.0224618e-08
max abs error : 1.4901161e-07
torch out min/max: -0.40827447 0.105299264
ort out min/max  : -0.40827453 0.105299324
```

The CUDA custom op preserves the original AFA operation with only small floating-point differences.

---

## Benchmark Results
# CPU Custom Op / Hybrid Execution

Before moving AFA to CUDA, AFA was executed by the CPU custom op. Other ONNX operators could run on CUDA or TensorRT, but the AFA node caused CPU fallback and GPU-CPU memory transfer.


Provider summary:

```
CPUExecutionProvider::AFA
CUDAExecutionProvider::Conv
CUDAExecutionProvider::ConvTranspose
CUDAExecutionProvider::MaxPool
CUDAExecutionProvider::Relu
CUDAExecutionProvider::Concat
CUDAExecutionProvider::MemcpyFromHost
CUDAExecutionProvider::MemcpyToHost
```

This means that AFA was still on CPU and caused device-host memory copy overhead.

---

## CUDA Custom Op Before Pad Removal

After implementing the CUDA custom op:

```
[ONNXRuntime-CUDA-Custom] average: 16.252 ms
```

Provider summary:

```
CPUExecutionProvider::Pad: 90
CUDAExecutionProvider::AFA: 90
CUDAExecutionProvider::Concat: 180
CUDAExecutionProvider::Conv: 900
CUDAExecutionProvider::ConvTranspose: 90
CUDAExecutionProvider::MaxPool: 90
CUDAExecutionProvider::MemcpyFromHost: 90
CUDAExecutionProvider::MemcpyToHost: 90
CUDAExecutionProvider::Relu: 510
```

AFA was successfully executed on CUDA, but decoder Pad still caused CPU fallback.

---

## CUDA Custom Op After Pad Removal

For fixed input size, unnecessary decoder Pad operations were removed. This removed CPU fallback.

Provider summary:

```
CUDAExecutionProvider::AFA: 90
CUDAExecutionProvider::Concat: 180
CUDAExecutionProvider::Conv: 900
CUDAExecutionProvider::ConvTranspose: 90
CUDAExecutionProvider::MaxPool: 90
CUDAExecutionProvider::Relu: 510
```

This confirms that all major operators are executed by CUDAExecutionProvider.

---

## Pad Removal for Fixed Input Size

The original U-Net upsampling block may contain a Pad operation to handle spatial-size mismatch between the upsampled feature and the skip feature.

For fixed input size, the upsampled feature and skip feature have the same spatial size. Therefore, the Pad operation is unnecessary.

A conditional version can be used so that Pad is only applied when the feature sizes do not match. For fixed input size, the condition is traced as false, and the Pad node is removed from the ONNX graph.

Expected warning:

```
TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect.
```

This warning is acceptable when the deployment input size is fixed.

---

## ONNX Runtime Profiling

ONNX Runtime profiling was used to check which execution provider handled each operator.

Before CUDA custom op:

```
CPUExecutionProvider::AFA
CPUExecutionProvider::Pad
CUDAExecutionProvider::Conv
CUDAExecutionProvider::Relu
CUDAExecutionProvider::Concat
```

After CUDA custom op and Pad removal:

```
CUDAExecutionProvider::AFA
CUDAExecutionProvider::Conv
CUDAExecutionProvider::ConvTranspose
CUDAExecutionProvider::MaxPool
CUDAExecutionProvider::Concat
CUDAExecutionProvider::Relu
```

This confirmed that AFA and other major operators are executed on GPU.

---

## TensorRT Extension

TensorRT can also be used, but full TensorRT integration requires a dedicated TensorRT plugin for kdafa::AFA.

When TensorRTExecutionProvider is enabled without an AFA plugin, TensorRT reports:

```
Plugin not found, are the plugin name, version, and namespace correct?
```

This is expected because kdafa::AFA is currently implemented as an ONNX Runtime custom op, not as a TensorRT plugin.

Current stage:
```
ONNX Runtime CUDA custom op
→ AFA runs on CUDAExecutionProvider
→ Original AFA operation is preserved
→ PyTorch output and ONNX Runtime output match
```

Future TensorRT stage:
```
kdafa::AFA
→ TensorRT plugin
→ cuFFT + CUDA kernels inside TensorRT engine
```

Therefore, TensorRT optimization is considered a follow-up stage.

---

## Summary

This implementation solves the ONNX export issue caused by FFT-based AFA.

```
Default ONNX export failed because torch.fft.fft2 was unsupported.
AFA was exported as kdafa::AFA custom ONNX operator.
CPU custom op was implemented using ONNX Runtime C++ API.
CUDA custom op was implemented using cuFFT and CUDA kernels.
PyTorch and ONNX Runtime outputs were verified to match.
Provider profiling confirmed that AFA and other major operators run on CUDAExecutionProvider.
Average inference time improved from about 42 ms to about 5.97 ms after CUDA custom op and Pad removal.
```

This demonstrates that KD-AFA-Net can preserve its original FFT-based AFA module while being executed in an ONNX Runtime GPU environment.

---

## Notes

The current implementation assumes fixed input resolution.

If dynamic input resolution is required, the following parts should be modified:

```
AFA mask size computation
decoder Pad logic
ONNX dynamic axes
custom op shape handling
cuFFT plan caching for multiple feature sizes
```

The current implementation is mainly intended for fixed-size inference and deployment-oriented acceleration experiments.
