from torch.utils.data import DataLoader
from dataset import dataset_inference_time
import time
import os
import numpy as np
import torch

from models.unet.sep_unet_model_custom import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.backends.cudnn.benchmark = True


class ExportWrapper(torch.nn.Module):
    """
    ONNX export용 wrapper.
    원래 모델은 (logits, feature_dict)를 반환하므로,
    ONNX export에서는 logits만 반환하도록 감싼다.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        logits, _ = self.model(x)
        return logits


def export_onnx(model, example_input, onnx_path="model.onnx"):
    model.eval()

    with torch.no_grad():
        torch.onnx.export(
            model,
            example_input,
            onnx_path,
            export_params=True,
            opset_version=19,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=None,

            # 핵심: kdafa custom domain 등록
            custom_opsets={"kdafa": 1},

            # 현재 방식은 torchscript 기반 symbolic custom op 방식이므로
            # dynamo=True는 일단 사용하지 않는 것이 안전함
            # dynamo=True,
        )

    print("ONNX saved:", onnx_path)


def benchmark_torch_gpu(model, loader, warm=5):
    model.eval()
    times = []
    batch_sizes = []

    with torch.no_grad():
        for batch_idx, (_, _, mask_img, name, fol) in enumerate(loader):
            mask_img = mask_img.float().cuda(non_blocking=True)
            bs = int(mask_img.shape[0])
            batch_sizes.append(bs)

            torch.cuda.synchronize()
            t0 = time.perf_counter()

            _ = model(mask_img)

            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            times.append(dt)

            fps = bs / dt if dt > 0 else float("inf")
            print(f"[Torch-GPU] {batch_idx}: {dt * 1000:.3f} ms | {fps:.2f} FPS")

    valid_times = times[warm:]
    valid_bs = batch_sizes[warm:]

    avg_ms = np.mean(valid_times) * 1000 if len(valid_times) else 0.0
    avg_fps = np.mean([b / t for b, t in zip(valid_bs, valid_times)]) if len(valid_times) else 0.0

    print(f"\n[Torch-GPU] average (after warm={warm}) : {avg_ms:.3f} ms | {avg_fps:.2f} FPS")
    return avg_ms, avg_fps


def benchmark_onnxruntime_cpu(onnx_path, loader, warm=5):
    """
    주의:
    현재 ONNX 안에는 kdafa::AFA custom op가 들어가므로,
    custom op library가 없으면 ONNX Runtime 실행은 실패한다.

    즉, export 성공 확인용으로는 사용 가능하지만,
    실제 benchmark는 kdafa custom op 구현 후 실행해야 한다.
    """
    import onnxruntime as ort

    sess = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"]
    )
    input_name = sess.get_inputs()[0].name

    times = []
    batch_sizes = []

    for batch_idx, (_, _, mask_img, name, fol) in enumerate(loader):
        x = mask_img.float().numpy()
        bs = int(x.shape[0])
        batch_sizes.append(bs)

        t0 = time.perf_counter()
        _ = sess.run(None, {input_name: x})
        dt = time.perf_counter() - t0
        times.append(dt)

        fps = bs / dt if dt > 0 else float("inf")
        print(f"[ONNXRuntime-CPU] {batch_idx}: {dt * 1000:.3f} ms | {fps:.2f} FPS")

    valid_times = times[warm:]
    valid_bs = batch_sizes[warm:]

    avg_ms = np.mean(valid_times) * 1000 if len(valid_times) else 0.0
    avg_fps = np.mean([b / t for b, t in zip(valid_bs, valid_times)]) if len(valid_times) else 0.0

    print(f"\n[ONNXRuntime-CPU] average (after warm={warm}) : {avg_ms:.3f} ms | {avg_fps:.2f} FPS")
    return avg_ms, avg_fps


def inspect_onnx_nodes(onnx_path):
    """
    ONNX 안에 kdafa::AFA custom node가 제대로 들어갔는지 확인용.
    """
    import onnx

    model = onnx.load(onnx_path)

    print("\n[ONNX opset imports]")
    for opset in model.opset_import:
        print(f"domain='{opset.domain}', version={opset.version}")

    print("\n[Custom nodes]")
    for node in model.graph.node:
        if node.domain == "kdafa":
            print(f"{node.domain}::{node.op_type}")


def benchmark_onnxruntime_cpu_custom(onnx_path, custom_op_dll, loader, warm=5):
    import onnxruntime as ort
    import time
    import numpy as np

    print("[ONNXRuntime version]", ort.__version__)

    so = ort.SessionOptions()

    # 핵심: kdafa::AFA custom op DLL 등록
    so.register_custom_ops_library(custom_op_dll)

    sess = ort.InferenceSession(
        onnx_path,
        sess_options=so,
        providers=["CPUExecutionProvider"]
    )

    input_name = sess.get_inputs()[0].name

    times = []
    batch_sizes = []

    for batch_idx, (_, _, mask_img, name, fol) in enumerate(loader):
        x = mask_img.float().numpy()
        bs = int(x.shape[0])
        batch_sizes.append(bs)

        t0 = time.perf_counter()
        _ = sess.run(None, {input_name: x})
        dt = time.perf_counter() - t0
        times.append(dt)

        fps = bs / dt if dt > 0 else float("inf")
        print(f"[ONNXRuntime-CPU-Custom] {batch_idx}: {dt * 1000:.3f} ms | {fps:.2f} FPS")

    valid_times = times[warm:]
    valid_bs = batch_sizes[warm:]

    avg_ms = np.mean(valid_times) * 1000 if len(valid_times) else 0.0
    avg_fps = np.mean([b / t for b, t in zip(valid_bs, valid_times)]) if len(valid_times) else 0.0

    print(f"\n[ONNXRuntime-CPU-Custom] average (after warm={warm}) : {avg_ms:.3f} ms | {avg_fps:.2f} FPS")
    return avg_ms, avg_fps


def compare_torch_onnx_custom(model, onnx_path, custom_op_dll, example):
    import onnxruntime as ort
    import numpy as np
    import torch

    model.eval()

    with torch.no_grad():
        torch_out = model(example)

        # 혹시 wrapper가 아니라 원래 모델을 넣었을 때 대비
        if isinstance(torch_out, tuple):
            torch_out = torch_out[0]

        torch_out = torch_out.detach().cpu().numpy()

    so = ort.SessionOptions()
    so.register_custom_ops_library(custom_op_dll)

    sess = ort.InferenceSession(
        onnx_path,
        sess_options=so,
        providers=["CPUExecutionProvider"]
    )

    input_name = sess.get_inputs()[0].name

    ort_out = sess.run(
        None,
        {input_name: example.detach().cpu().numpy()}
    )[0]

    abs_diff = np.abs(torch_out - ort_out)

    print("\n[Compare Torch vs ONNXRuntime-Custom]")
    print("mean abs error:", abs_diff.mean())
    print("max abs error :", abs_diff.max())
    print("torch out min/max:", torch_out.min(), torch_out.max())
    print("ort out min/max  :", ort_out.min(), ort_out.max())

    return abs_diff.mean(), abs_diff.max()


if __name__ == "__main__":
    eval_data = dataset_inference_time(
        root="",
        transforms=None,
        imgSize=192,
        inputsize=128,
        pred_step=1,
        imglist=[]
    )

    eval_loader = DataLoader(
        eval_data,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    gen = V_Thin_Sep_UNet_4_custom_op(
        n_channels=3,
        n_classes=3
    ).cuda()
    gen.eval()

    # 학습된 weight가 있다면 여기서 load
    # ckpt = torch.load("your_weight_path.pth", map_location="cuda")
    # gen.load_state_dict(ckpt)

    # 핵심: export에는 gen이 아니라 wrapper를 넣어야 함
    export_model = ExportWrapper(gen).cuda().eval()

    _, _, mask_img0, _, _ = next(iter(eval_loader))
    example = mask_img0.float().cuda()

    # Windows면 예: "./kd_afa_custom.onnx" 또는 r"C:\...\kd_afa_custom.onnx"
    onnx_path = "./kd_afa_custom.onnx"

    export_onnx(
        export_model,
        example,
        onnx_path=onnx_path
    )

    inspect_onnx_nodes(onnx_path)

    warm = 5

    # PyTorch 기준 전체 모델 시간 측정
    benchmark_torch_gpu(export_model, eval_loader, warm=warm)

    custom_op_dll = r"C:\Users\8138\PycharmProjects\DION4FR_student_test\custom_ops\kdafa_ort\build\Release\kdafa_custom_ops.dll"

    compare_torch_onnx_custom(
        export_model,
        onnx_path,
        custom_op_dll,
        example
    )

    benchmark_onnxruntime_cpu_custom(
        onnx_path,
        custom_op_dll,
        eval_loader,
        warm=warm
    )

    # 주의:
    # kdafa::AFA custom op 실행 라이브러리가 없으면 아래는 실패함
    # benchmark_onnxruntime_cpu(onnx_path, eval_loader, warm=warm)