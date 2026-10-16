"""Stability's CUDA trainer, run inside the engine's environment with two things Khaos needs that its script does
not give: every step's loss written to khaos_losses.csv beside the checkpoints (the script logs no loss), and the
demo callback kept quiet (it renders unmarked audio at step 1 whatever --demo_every says). Stability's code is
not changed; its train_lora.py is imported and its main() run with the same arguments."""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

LOSSES = "khaos_losses.csv"


def main() -> int:
    import pytorch_lightning as pl  # noqa: PLC0415

    root = Path(os.environ.get("SA3_ROOT") or Path.cwd())
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "scripts"))
    import train_lora  # noqa: PLC0415  — Stability's scripts/train_lora.py, found through the checkout on the path

    args = sys.argv[1:]
    save_dir = Path(args[args.index("--save_dir") + 1]) if "--save_dir" in args else Path("./lora_checkpoints")
    save_dir.mkdir(parents=True, exist_ok=True)
    losses_path = save_dir / LOSSES

    class KhaosLosses(pl.Callback):
        """One row per optimiser step: the step (1-based) and the loss the wrapper returned."""

        def __init__(self):
            self._fh = losses_path.open("a", newline="", encoding="utf-8")
            self._w = csv.writer(self._fh)
            if losses_path.stat().st_size == 0:
                self._w.writerow(["step", "loss"])

        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs
            try:
                value = float(loss.detach().float().cpu()) if hasattr(loss, "detach") else float(loss)
            except (TypeError, ValueError):
                return
            self._w.writerow([int(trainer.global_step), f"{value:.6f}"])
            self._fh.flush()

        def on_train_end(self, trainer, module):
            self._fh.close()

    class QuietDemo(pl.Callback):
        """Stability's demo callback replaced: a Khaos run renders nothing unmarked."""

        def __init__(self, *a, **k):
            super().__init__()

    real_trainer = pl.Trainer

    # Stability's checkpoint callback saves on its cadence alone: when the run's length is not a multiple of it,
    # a second one saves the final step, so the last epoch's weights always exist to export.
    total = int(args[args.index("--steps") + 1]) if "--steps" in args else 0
    every = int(args[args.index("--checkpoint_every") + 1]) if "--checkpoint_every" in args else 0
    final = [pl.callbacks.ModelCheckpoint(every_n_train_steps=total, save_top_k=-1)] if total and every and total % every else []

    class TrainerWithLosses(real_trainer):
        def __init__(self, *a, **k):
            k["callbacks"] = [*(k.get("callbacks") or []), *final, KhaosLosses()]
            super().__init__(*a, **k)

    train_lora.pl.Trainer = TrainerWithLosses
    # The text encoder is read from Stability's open base repository: the same files, no account.
    from khaos_attribution.engines.sa3.loading import load_for_training  # noqa: PLC0415
    train_lora.load_model = load_for_training
    train_lora.DiffusionCondInpaintDemoCallback = QuietDemo
    train_lora.main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
