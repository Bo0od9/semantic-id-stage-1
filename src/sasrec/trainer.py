"""Training loop for SASRec with val early stopping."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..utils.seed import free_device_memory

if TYPE_CHECKING:
    from ..logger import WandBWriter

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0
    eval_every: int = 500
    early_stopping_patience: int = 3
    tuning_metric: str = "recall@10"
    log_every: int = 50


class SASRecTrainer:
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        eval_fn: Callable[[nn.Module], dict[str, float]],
        cfg: TrainerConfig,
        save_dir: Path,
        device: torch.device,
        writer: "WandBWriter | None" = None,
        eval_callback: Callable[[int, dict[str, float]], None] | None = None,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.eval_fn = eval_fn
        self.cfg = cfg
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.writer = writer
        # Опциональный хук для Optuna-pruning. Вызывается после каждого eval'а.
        # Если выбросит исключение (optuna.TrialPruned), оно проберётся наверх через fit().
        self.eval_callback = eval_callback
        # non_blocking host→device copy имеет смысл только на CUDA с pin_memory;
        # на MPS это no-op, на CPU — лишний флаг.
        self._non_blocking = device.type == "cuda"

        # loss_fn stores item_encoder via non-registered wrapper, so loss_fn.parameters()
        # yields only its own learnables (log_tau). No duplication with model.parameters().
        self.optimizer = torch.optim.Adam(
            list(model.parameters()) + list(loss_fn.parameters()),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

        self.best_metric: float = -math.inf
        self.best_step: int = 0
        self.no_improve_count: int = 0
        self.step: int = 0
        self.epoch: int = 0

    def _train_step(self, batch: dict[str, torch.Tensor]) -> tuple[float, float]:
        self.model.train()
        nb = self._non_blocking
        items = batch["items"].to(self.device, non_blocking=nb)
        positives = batch["positives"].to(self.device, non_blocking=nb)
        lengths = batch["lengths"].to(self.device, non_blocking=nb)

        hidden, mask = self.model(items, lengths)
        loss = self.loss_fn(hidden, mask, positives)

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {self.step}: {loss.item()}")

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for group in self.optimizer.param_groups for p in group["params"]],
            max_norm=self.cfg.grad_clip_norm,
        )
        self.optimizer.step()
        return float(loss.item()), float(grad_norm.item())

    def _checkpoint(self, name: str) -> Path:
        path = self.save_dir / name
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "loss_state_dict": self.loss_fn.state_dict(),
                "best_metric": self.best_metric,
                "best_step": self.best_step,
                "step": self.step,
                "epoch": self.epoch,
            },
            path,
        )
        return path

    def _load_checkpoint(self, name: str) -> None:
        path = self.save_dir / name
        if not path.exists():
            logger.warning(f"checkpoint {path} does not exist")
            return
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state["model_state_dict"])
        self.loss_fn.load_state_dict(state["loss_state_dict"])
        self.best_metric = float(state["best_metric"])
        self.best_step = int(state["best_step"])

    def _run_eval(self) -> bool:
        free_device_memory(self.device)
        metrics = self.eval_fn(self.model)
        free_device_memory(self.device)
        metric_val = float(metrics.get(self.cfg.tuning_metric, -math.inf))
        improved = metric_val > self.best_metric

        if improved:
            self.best_metric = metric_val
            self.best_step = self.step
            self.no_improve_count = 0
            ckpt_path = self._checkpoint("model_best.pth")
            logger.info(
                f"[step {self.step}] new best {self.cfg.tuning_metric}={metric_val:.5f}"
            )
            if self.writer is not None:
                self.writer.add_checkpoint(str(ckpt_path), str(self.save_dir))
        else:
            self.no_improve_count += 1
            logger.info(
                f"[step {self.step}] {self.cfg.tuning_metric}={metric_val:.5f} "
                f"(best={self.best_metric:.5f} @ step {self.best_step}, "
                f"no_improve={self.no_improve_count}/{self.cfg.early_stopping_patience})"
            )

        if self.writer is not None:
            self.writer.set_step(self.step, mode="val")
            self.writer.add_scalars({**metrics, "best_metric": self.best_metric})

        if self.eval_callback is not None:
            self.eval_callback(self.step, metrics)

        return self.no_improve_count >= self.cfg.early_stopping_patience

    def fit(self, train_loader: DataLoader) -> None:
        for epoch in range(self.cfg.epochs):
            self.epoch = epoch
            for batch in train_loader:
                loss_val, grad_norm = self._train_step(batch)
                self.step += 1

                if self.step % self.cfg.log_every == 0:
                    stats = self.loss_fn.stats()
                    logger.info(
                        f"[epoch {epoch + 1}/{self.cfg.epochs} step {self.step}] "
                        f"loss={loss_val:.4f} tau={stats['tau']:.4f} grad_norm={grad_norm:.3f}"
                    )
                    if self.writer is not None:
                        self.writer.set_step(self.step, mode="train")
                        self.writer.add_scalars(
                            {
                                "loss": loss_val,
                                "grad_norm": grad_norm,
                                "tau": stats["tau"],
                                "log_tau": stats["log_tau"],
                            }
                        )

                if self.cfg.eval_every > 0 and self.step % self.cfg.eval_every == 0:
                    if self._run_eval():
                        logger.info(f"[step {self.step}] early stopping")
                        self._load_checkpoint("model_best.pth")
                        return

            if self.cfg.eval_every <= 0:
                if self._run_eval():
                    logger.info(f"[epoch {epoch + 1}] early stopping")
                    self._load_checkpoint("model_best.pth")
                    return

        self._run_eval()
        best_path = self.save_dir / "model_best.pth"
        assert best_path.exists(), (
            f"{best_path} was not created during training — no eval run yielded "
            f"a finite '{self.cfg.tuning_metric}' better than -inf. "
            f"Check that eval_fn returns a finite value and that the loader is non-empty."
        )
        self._load_checkpoint("model_best.pth")
