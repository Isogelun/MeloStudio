from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .checkpoint import load_checkpoint
from .data import LLSMWavDataset
from .dsp import DSPConfig, DSPControlPredictor, DifferentiableDSP
from .losses import NHNVocoderLoss
from .training_config import (
    parse_training_args,
    require_configured_paths,
    resolve_training_device,
)


def save_post_training_checkpoint(
    path: str | Path,
    base: torch.nn.Module,
    predictor: DSPControlPredictor,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    step: int = 0,
    epoch: int = 0,
    metrics: dict | None = None,
) -> Path:
    payload = {
        "model": base.state_dict(),
        "config": base.config.to_dict(),
        "format_version": 3,
        "checkpoint_type": "dsp_posttrained",
        "dsp_predictor": predictor.state_dict(),
        "dsp_config": predictor.config.to_dict(),
        "step": step,
        "epoch": epoch,
        "metrics": metrics or {},
    }
    if optimizer is not None:
        payload["dsp_optimizer"] = optimizer.state_dict()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Post-train a small DSP control head with the NHN base frozen"
    )
    parser.add_argument("data", type=Path, nargs="?", help="existing paired LLSM72/WAV dataset")
    parser.add_argument("base_checkpoint", type=Path, nargs="?")
    parser.add_argument("output", type=Path, nargs="?", help="output directory or .pt path")
    parser.add_argument("--config", type=Path, help="nhn_dsp_post_train YAML configuration")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--segment-seconds", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--control-regularization", type=float, default=1e-3)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parse_training_args(parser, "post_train", argv)
    require_configured_paths(parser, args, ("data", "base_checkpoint", "output"))
    args.device = resolve_training_device(args.device)
    if args.epochs < 1 or args.batch_size < 1 or args.segment_seconds <= 0:
        raise ValueError("epochs, batch-size and segment-seconds must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    base = load_checkpoint(args.base_checkpoint, device)
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    base.eval()

    source_payload = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    if "dsp_config" in source_payload:
        dsp_config = DSPConfig.from_dict(source_payload["dsp_config"])
    else:
        dsp_config = DSPConfig(hidden_channels=args.hidden_channels)
    predictor = DSPControlPredictor(base.config.feature_dim, dsp_config).to(device)
    if "dsp_predictor" in source_payload:
        predictor.load_state_dict(source_payload["dsp_predictor"])
    dsp = DifferentiableDSP(dsp_config).to(device)
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=args.learning_rate)
    if "dsp_optimizer" in source_payload:
        optimizer.load_state_dict(source_payload["dsp_optimizer"])

    segment_frames = max(
        1,
        round(args.segment_seconds * base.config.sample_rate / base.config.hop_length),
    )
    manifest = args.data / "manifests/train.jsonl"
    dataset = LLSMWavDataset(
        args.data,
        base.config.sample_rate,
        base.config.hop_length,
        segment_frames,
        random_crop=True,
        manifest=manifest if manifest.is_file() else None,
        group_variants=manifest.is_file(),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    criterion = NHNVocoderLoss(base.config.sample_rate).to(device)
    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    step = int(source_payload.get("step", 0)) if "dsp_predictor" in source_payload else 0

    output = args.output
    latest = output if output.suffix else output / "latest.pt"
    log_path = latest.with_name(f"{latest.stem}.jsonl")
    latest.parent.mkdir(parents=True, exist_ok=True)

    predictor.train()
    for epoch in range(args.epochs):
        epoch_losses = []
        for features, target in loader:
            features = features.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                dry = base(features, generator=generator)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=amp_enabled
            ):
                controls = predictor(features)
                estimate = dsp(dry, controls, generator=generator)
                reconstruction = criterion(estimate, target, features)
                normalized = controls.clone()
                normalized[:, 0] = normalized[:, 0] / 12.0
                regularization = normalized.square().mean()
                loss = reconstruction + args.control_regularization * regularization
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            step += 1
            loss_value = float(loss.item())
            epoch_losses.append(loss_value)
            if step % args.log_every == 0:
                print(
                    f"epoch={epoch + 1}/{args.epochs} step={step} "
                    f"loss={loss_value:.5f} reconstruction={float(reconstruction.item()):.5f}"
                )
            if step % args.save_every == 0:
                save_post_training_checkpoint(
                    latest, base, predictor, optimizer, step=step, epoch=epoch
                )
        record = {
            "epoch": epoch + 1,
            "step": step,
            "loss": float(np.mean(epoch_losses)),
            "base_frozen": True,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False))
        save_post_training_checkpoint(
            latest,
            base,
            predictor,
            optimizer,
            step=step,
            epoch=epoch + 1,
            metrics={"train_loss": record["loss"]},
        )


if __name__ == "__main__":
    main()
