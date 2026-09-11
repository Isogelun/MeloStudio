from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .checkpoint import load_training_checkpoint, save_checkpoint
from ..core.config import NHNVocoderConfig
from .data import LLSMWavDataset
from .discriminators import (
    MultiPeriodScaleDiscriminator,
    discriminator_loss,
    feature_matching_loss,
    generator_adversarial_loss,
)
from .losses import NHNVocoderLoss
from ..core.model import NHNVocoder
from ..preprocessing.stats import load_feature_range
from .config import (
    detect_training_section,
    parse_training_args,
    require_configured_paths,
    resolve_training_device,
)


def _checkpoint_paths(output: Path) -> Tuple[Path, Path, Path]:
    if output.suffix:
        latest = output
        best = output.with_name(f"{output.stem}.best{output.suffix}")
        log = output.with_name(f"{output.stem}.jsonl")
    else:
        latest = output / "latest.pt"
        best = output / "best.pt"
        log = output / "training.jsonl"
    return latest, best, log


def _split_datasets(
    root: Path,
    config: NHNVocoderConfig,
    segment_frames: int,
    validation_root: Optional[Path],
    validation_split: float,
    seed: int,
):
    train_manifest = root / "manifests/train.jsonl"
    valid_manifest = root / "manifests/valid.jsonl"
    if validation_root is None and train_manifest.is_file():
        train_data = LLSMWavDataset(
            root, config.sample_rate, config.hop_length, segment_frames,
            random_crop=True, manifest=train_manifest, group_variants=True,
        )
        validation_data = None
        if valid_manifest.is_file() and valid_manifest.stat().st_size:
            validation_data = LLSMWavDataset(
                root, config.sample_rate, config.hop_length, segment_frames,
                random_crop=False, manifest=valid_manifest, group_variants=False,
            )
        return train_data, validation_data
    train_data = LLSMWavDataset(
        root, config.sample_rate, config.hop_length, segment_frames, random_crop=True
    )
    if validation_root is not None:
        validation_data = LLSMWavDataset(
            validation_root,
            config.sample_rate,
            config.hop_length,
            segment_frames,
            random_crop=False,
        )
        return train_data, validation_data
    if len(train_data) < 2 or validation_split <= 0:
        return train_data, None
    validation_count = min(
        len(train_data) - 1, max(1, round(len(train_data) * validation_split))
    )
    order = torch.randperm(
        len(train_data), generator=torch.Generator().manual_seed(seed)
    ).tolist()
    validation_indices = order[:validation_count]
    train_indices = order[validation_count:]
    validation_view = LLSMWavDataset(
        root, config.sample_rate, config.hop_length, segment_frames, random_crop=False
    )
    return Subset(train_data, train_indices), Subset(validation_view, validation_indices)


def _evaluate(
    model: NHNVocoder,
    loader: DataLoader,
    criterion: NHNVocoderLoss,
    device: torch.device,
    amp_enabled: bool,
) -> float:
    model.eval()
    losses = []
    generator = torch.Generator(device=device).manual_seed(1234)
    with torch.inference_mode():
        for features, target in loader:
            features = features.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=amp_enabled
            ):
                estimate = model(features, generator=generator)
                losses.append(float(criterion(estimate, target, features).item()))
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the compact NHN LLSM vocoder")
    parser.add_argument("data", type=Path, nargs="?", help="directory of matching name.npy/name.wav pairs")
    parser.add_argument("output", type=Path, nargs="?", help="checkpoint file or output directory")
    parser.add_argument("--config", type=Path, help="nhn_base_train YAML configuration")
    parser.add_argument("--validation-data", type=Path, help="separate validation directory")
    parser.add_argument("--validation-split", type=float, default=0.05)
    parser.add_argument("--feature-stats", type=Path, help="feature_stats.npz made by nhn-stats")
    parser.add_argument("--resume", type=Path, help="restore all training state")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--segment-seconds", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gan-start-step", type=int, default=10_000)
    parser.add_argument("--adversarial-weight", type=float, default=1.0)
    parser.add_argument("--feature-matching-weight", type=float, default=2.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    return parser


def main(argv: list[str] | None = None) -> None:
    if detect_training_section(argv) == "post_train":
        from .post_train import main as post_train_main

        post_train_main(argv)
        return
    parser = build_parser()
    args = parse_training_args(parser, "base_train", argv)
    require_configured_paths(parser, args, ("data", "output"))
    args.device = resolve_training_device(args.device)
    if not 0 <= args.validation_split < 1:
        raise ValueError("validation_split must be in [0, 1)")
    if args.segment_seconds <= 0 or args.batch_size <= 0:
        raise ValueError("segment_seconds and batch_size must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.resume is not None:
        header = torch.load(args.resume, map_location="cpu", weights_only=False)
        config = NHNVocoderConfig.from_dict(header.get("config", {}))
    else:
        config_values = {}
        report_path = args.data / "preprocess_report.json"
        if report_path.is_file():
            report_config = json.loads(report_path.read_text(encoding="utf-8")).get("config", {})
            if report_config.get("task_mode") == "vocoder_bwe":
                config_values = {
                    "task_mode": "vocoder_bwe",
                    "trained_input_sample_rates": tuple(report_config["source_sample_rates"]),
                }
        config = NHNVocoderConfig(**config_values)
    segment_frames = max(
        1, round(args.segment_seconds * config.sample_rate / config.hop_length)
    )
    train_data, validation_data = _split_datasets(
        args.data,
        config,
        segment_frames,
        args.validation_data,
        args.validation_split,
        args.seed,
    )
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": args.device.startswith("cuda"),
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(train_data, shuffle=True, drop_last=False, **loader_options)
    validation_loader = (
        DataLoader(validation_data, shuffle=False, drop_last=False, **loader_options)
        if validation_data is not None
        else None
    )

    device = torch.device(args.device)
    model = NHNVocoder(config)
    stats_path = args.feature_stats
    if stats_path is None:
        automatic_stats = args.data / "feature_stats.npz"
        stats_path = automatic_stats if automatic_stats.is_file() else None
    if stats_path is not None and args.resume is None:
        minimum, maximum = load_feature_range(stats_path)
        model.input_normalizer.set_range(
            torch.from_numpy(minimum), torch.from_numpy(maximum)
        )
        print(f"using feature statistics: {stats_path}")
    model.to(device)
    discriminator = MultiPeriodScaleDiscriminator().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.8, 0.99),
        weight_decay=args.weight_decay,
    )
    discriminator_optimizer = torch.optim.AdamW(
        discriminator.parameters(), lr=args.learning_rate, betas=(0.8, 0.99)
    )
    total_steps = max(1, args.epochs * len(train_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    discriminator_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        discriminator_optimizer, T_max=total_steps
    )
    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    criterion = NHNVocoderLoss(config.sample_rate).to(device)

    step = 0
    start_epoch = 0
    best_validation = float("inf")
    if args.resume is not None:
        restored = load_training_checkpoint(
            args.resume, model, optimizer, scheduler, scaler, device=device
        )
        if "discriminator" in restored:
            discriminator.load_state_dict(restored["discriminator"])
        if "discriminator_optimizer" in restored:
            discriminator_optimizer.load_state_dict(restored["discriminator_optimizer"])
        if "discriminator_scheduler" in restored:
            discriminator_scheduler.load_state_dict(restored["discriminator_scheduler"])
        step = int(restored.get("step", 0))
        start_epoch = int(restored.get("epoch", 0))
        best_validation = float(
            restored.get("metrics", {}).get("best_validation", float("inf"))
        )
        # When --epochs extends a run, stretch the cosine schedule over the new
        # total rather than cycling after the old T_max.
        scheduler.T_max = total_steps
        discriminator_scheduler.T_max = total_steps
        for group, learning_rate in zip(
            optimizer.param_groups, scheduler._get_closed_form_lr()
        ):
            group["lr"] = learning_rate
        print(f"resumed {args.resume}: epoch={start_epoch} step={step}")

    latest_path, best_path, log_path = _checkpoint_paths(args.output)
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(latest_path.parent / "tensorboard"))
    except ImportError:
        writer = None
        print("tensorboard unavailable; continuing with console and JSONL logs")

    model.train()
    discriminator.train()

    def adversarial_state() -> dict:
        return {
            "discriminator": discriminator.state_dict(),
            "discriminator_optimizer": discriminator_optimizer.state_dict(),
            "discriminator_scheduler": discriminator_scheduler.state_dict(),
        }

    for epoch in range(start_epoch, args.epochs):
        epoch_losses = []
        for features, target in train_loader:
            features = features.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=amp_enabled
            ):
                estimate = model(features)
                auxiliary_loss, components = criterion(
                    estimate, target, features, return_components=True
                )
            gan_active = step >= args.gan_start_step
            discriminator_value = 0.0
            adversarial_value = 0.0
            feature_value = 0.0
            if gan_active:
                discriminator_optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=amp_enabled
                ):
                    real_outputs = discriminator(target)
                    fake_outputs = discriminator(estimate.detach())
                    discriminator_objective = discriminator_loss(real_outputs, fake_outputs)
                scaler.scale(discriminator_objective).backward()
                scaler.unscale_(discriminator_optimizer)
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 10.0)
                scaler.step(discriminator_optimizer)
                discriminator_scheduler.step()
                discriminator_value = float(discriminator_objective.item())
                for parameter in discriminator.parameters():
                    parameter.requires_grad_(False)
                with torch.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=amp_enabled
                ):
                    with torch.no_grad():
                        real_outputs = discriminator(target)
                    fake_outputs = discriminator(estimate)
                    adversarial = generator_adversarial_loss(fake_outputs)
                    feature_match = feature_matching_loss(real_outputs, fake_outputs)
                    loss = auxiliary_loss + args.adversarial_weight * adversarial + args.feature_matching_weight * feature_match
                adversarial_value = float(adversarial.item())
                feature_value = float(feature_match.item())
                for parameter in discriminator.parameters():
                    parameter.requires_grad_(True)
            else:
                loss = auxiliary_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            loss_value = float(loss.item())
            epoch_losses.append(loss_value)
            if writer is not None:
                writer.add_scalar("train/loss", loss_value, step)
                writer.add_scalar("train/learning_rate", scheduler.get_last_lr()[0], step)
                writer.add_scalar("train/auxiliary", float(auxiliary_loss.item()), step)
                if gan_active:
                    writer.add_scalar("train/discriminator", discriminator_value, step)
                    writer.add_scalar("train/adversarial", adversarial_value, step)
                    writer.add_scalar("train/feature_matching", feature_value, step)
                for name, value in components.items():
                    writer.add_scalar(f"loss/{name}", float(value.item()), step)
            if step % args.log_every == 0:
                print(
                    f"epoch={epoch + 1}/{args.epochs} step={step} "
                    f"loss={loss_value:.5f} lr={scheduler.get_last_lr()[0]:.3e}"
                    f" gan={'on' if gan_active else 'warmup'}"
                )
            if step % args.save_every == 0:
                save_checkpoint(
                    latest_path,
                    model,
                    optimizer,
                    step,
                    epoch,
                    scheduler,
                    scaler,
                    {"best_validation": best_validation},
                    adversarial_state(),
                )

        train_loss = float(np.mean(epoch_losses))
        validation_loss = (
            _evaluate(model, validation_loader, criterion, device, amp_enabled)
            if validation_loader is not None
            else train_loss
        )
        record = {
            "epoch": epoch + 1,
            "step": step,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "learning_rate": scheduler.get_last_lr()[0],
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False))
        if writer is not None:
            writer.add_scalar("validation/loss", validation_loss, step)
        metrics = {"best_validation": min(best_validation, validation_loss)}
        if validation_loss < best_validation:
            best_validation = validation_loss
            save_checkpoint(
                best_path, model, optimizer, step, epoch + 1, scheduler, scaler,
                metrics, adversarial_state(),
            )
        save_checkpoint(
            latest_path, model, optimizer, step, epoch + 1, scheduler, scaler, metrics,
            adversarial_state(),
        )
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
