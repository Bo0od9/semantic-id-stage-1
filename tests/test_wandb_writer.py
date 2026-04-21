"""Unit tests for src.logger.WandBWriter.

All tests stub the ``wandb`` module via ``sys.modules`` before constructing
the writer so no network calls, login prompts or on-disk state are made.
"""

from __future__ import annotations

import sys
import time
from types import SimpleNamespace
from unittest.mock import MagicMock


def _install_fake_wandb(monkeypatch) -> MagicMock:
    fake = MagicMock(name="wandb")
    fake.run = SimpleNamespace(summary={})
    monkeypatch.setitem(sys.modules, "wandb", fake)
    return fake


def _make_writer(monkeypatch, mode: str = "online"):
    fake = _install_fake_wandb(monkeypatch)
    from src.logger import WandBWriter

    writer = WandBWriter(
        project_config={"foo": 1},
        project_name="proj",
        entity=None,
        run_name="run",
        mode=mode,
        tags=["t1"],
    )
    return writer, fake


def test_init_offline_skips_login(monkeypatch):
    writer, fake = _make_writer(monkeypatch, mode="offline")
    assert fake.login.call_count == 0
    fake.init.assert_called_once()
    assert fake.init.call_args.kwargs["mode"] == "offline"
    assert fake.init.call_args.kwargs["tags"] == ["t1"]
    assert writer.wandb is fake


def test_init_online_calls_login(monkeypatch):
    writer, fake = _make_writer(monkeypatch, mode="online")
    assert fake.login.call_count == 1
    assert fake.init.call_args.kwargs["mode"] == "online"
    assert writer.mode == ""


def test_init_disabled_skips_login(monkeypatch):
    _writer, fake = _make_writer(monkeypatch, mode="disabled")
    assert fake.login.call_count == 0
    assert fake.init.call_args.kwargs["mode"] == "disabled"


def test_object_name_prefix(monkeypatch):
    writer, _fake = _make_writer(monkeypatch, mode="disabled")
    writer.mode = "train"
    assert writer._object_name("loss") == "train/loss"
    writer.mode = ""
    assert writer._object_name("loss") == "loss"


def test_add_scalars_uses_prefix(monkeypatch):
    writer, fake = _make_writer(monkeypatch, mode="disabled")
    writer.mode = "train"
    writer.step = 42
    writer.add_scalars({"loss": 0.5, "tau": 0.1})
    fake.log.assert_called_with(
        {"train/loss": 0.5, "train/tau": 0.1}, step=42
    )


def test_add_scalar_uses_prefix(monkeypatch):
    writer, fake = _make_writer(monkeypatch, mode="disabled")
    writer.mode = "val"
    writer.step = 7
    writer.add_scalar("recall@10", 0.123)
    fake.log.assert_called_with({"val/recall@10": 0.123}, step=7)


def test_set_step_logs_steps_per_sec_after_first_call(monkeypatch):
    writer, fake = _make_writer(monkeypatch, mode="disabled")
    writer.set_step(0, mode="train")
    assert fake.log.call_count == 0  # step==0 branch resets timer only
    time.sleep(0.01)
    writer.set_step(10, mode="train")
    assert fake.log.call_count == 1
    logged = fake.log.call_args.args[0]
    assert "train/steps_per_sec" in logged
    assert logged["train/steps_per_sec"] > 0


def test_add_checkpoint_calls_wandb_save(monkeypatch):
    writer, fake = _make_writer(monkeypatch, mode="disabled")
    writer.add_checkpoint("/tmp/run/model_best.pth", "/tmp/run")
    fake.save.assert_called_once_with(
        "/tmp/run/model_best.pth", base_path="/tmp/run"
    )


def test_set_summary_writes_to_run_summary(monkeypatch):
    writer, fake = _make_writer(monkeypatch, mode="disabled")
    writer.set_summary({"best/val_metric": 0.14, "best/step": 1234})
    assert fake.run.summary["best/val_metric"] == 0.14
    assert fake.run.summary["best/step"] == 1234


def test_finish_calls_wandb_finish(monkeypatch):
    writer, fake = _make_writer(monkeypatch, mode="disabled")
    writer.finish()
    fake.finish.assert_called_once()
