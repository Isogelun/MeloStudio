from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import Tensor, nn

from ..training.checkpoint import load_checkpoint


class InferenceWrapper(nn.Module):
    """Export contract: float features [B,T,72] -> waveform [B,1,T*hop]."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, features: Tensor) -> Tensor:
        return self.model(features)


def export_inference_checkpoint(source: str | Path, output: str | Path) -> Path:
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError("source checkpoint does not contain model weights")
    pure = {
        "model": payload["model"],
        "config": payload.get("config", {}),
        "format_version": payload.get("format_version", 3),
        "checkpoint_type": "inference",
        "step": payload.get("step", 0),
        "metrics": payload.get("metrics", {}),
    }
    for key in ("dsp_predictor", "dsp_config"):
        if key in payload:
            pure[key] = payload[key]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pure, output)
    return output


def export_torchscript(source: str | Path, output: str | Path, frames: int = 32) -> Path:
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if "dsp_predictor" in payload:
        raise NotImplementedError(
            "DSP post-trained graph export is not available yet; export the pure "
            "checkpoint and run it through VocoderSession"
        )
    model = load_checkpoint(source, "cpu").set_export_mode(False)
    wrapper = InferenceWrapper(model).eval()
    example = torch.zeros(1, frames, 72)
    example[..., 0] = 1
    example[..., 1] = 220
    traced = torch.jit.trace(wrapper, example, check_trace=False, strict=False)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(output))
    return output


def export_onnx(source: str | Path, output: str | Path, frames: int = 32) -> Path:
    try:
        import onnx
    except ImportError as error:
        raise RuntimeError("ONNX export requires: uv sync --extra export") from error
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if "dsp_predictor" in payload:
        raise NotImplementedError(
            "DSP post-trained graph export is not available yet; export the pure "
            "checkpoint and run it through VocoderSession"
        )
    model = load_checkpoint(source, "cpu").set_export_mode(True, onnx=True)
    wrapper = InferenceWrapper(model).eval()
    example = torch.zeros(1, frames, 72)
    example[..., 0] = 1
    example[..., 1] = 220
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        example,
        output,
        input_names=["features"],
        output_names=["waveform"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(output))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Export NHN inference-only artifacts")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--format", choices=("checkpoint", "torchscript", "onnx"), default="checkpoint")
    parser.add_argument("--frames", type=int, default=32, help="example frame count for graph export")
    args = parser.parse_args()
    exporters = {
        "checkpoint": export_inference_checkpoint,
        "torchscript": lambda source, output: export_torchscript(source, output, args.frames),
        "onnx": lambda source, output: export_onnx(source, output, args.frames),
    }
    result = exporters[args.format](args.checkpoint, args.output)
    print(f"exported {args.format}: {result} ({result.stat().st_size / 1024**2:.2f} MiB)")


if __name__ == "__main__":
    main()
