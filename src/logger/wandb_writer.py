"""Thin Weights & Biases writer adapted from pytorch_project_template.

Differences from the template:
- ``_object_name`` uses a ``{mode}/{name}`` prefix so the W&B UI groups
  panels by ``/`` (``train/``, ``val/``, ``final/``) instead of the
  suffix convention (``loss_train``).
- ``wandb.login()`` is called only when ``mode == "online"``: offline
  and disabled runs do not require an API key.
- Only scalar/checkpoint/summary methods are kept — image/audio/table
  logging from the template is unused in this project.
- ``import wandb`` happens inside ``__init__`` so projects with W&B
  disabled do not pay the import cost.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any


class WandBWriter:
    def __init__(
        self,
        *,
        project_config: dict[str, Any],
        project_name: str,
        entity: str | None = None,
        run_name: str | None = None,
        mode: str = "online",
        tags: list[str] | None = None,
        save_code: bool = False,
    ) -> None:
        import wandb

        if mode == "online":
            wandb.login()

        wandb.init(
            project=project_name,
            entity=entity,
            config=project_config,
            name=run_name,
            mode=mode,
            tags=list(tags) if tags else [],
            save_code=save_code,
        )
        self.wandb = wandb

        self.step = 0
        self.mode = ""
        self.timer = datetime.now()

    def set_step(self, step: int, mode: str = "train") -> None:
        self.mode = mode
        previous_step = self.step
        self.step = step
        if step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            seconds = duration.total_seconds()
            if seconds > 0:
                self.add_scalar(
                    "steps_per_sec", (self.step - previous_step) / seconds
                )
            self.timer = datetime.now()

    def _object_name(self, object_name: str) -> str:
        if not self.mode:
            return object_name
        return f"{self.mode}/{object_name}"

    def add_scalar(self, scalar_name: str, scalar: float) -> None:
        self.wandb.log(
            {self._object_name(scalar_name): scalar},
            step=self.step,
        )

    def add_scalars(self, scalars: dict[str, float]) -> None:
        self.wandb.log(
            {self._object_name(name): value for name, value in scalars.items()},
            step=self.step,
        )

    def add_checkpoint(self, checkpoint_path: str, save_dir: str) -> None:
        self.wandb.save(checkpoint_path, base_path=save_dir)

    def set_summary(self, summary: dict[str, Any]) -> None:
        for key, value in summary.items():
            self.wandb.run.summary[key] = value

    def finish(self) -> None:
        self.wandb.finish()
