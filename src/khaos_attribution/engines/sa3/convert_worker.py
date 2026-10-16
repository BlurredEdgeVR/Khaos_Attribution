"""One Lightning checkpoint of Stability's trainer into the adapter's safetensors file, inside the engine's
environment: python convert_worker.py <in.ckpt> <out.safetensors>."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    from stable_audio_3.models.lora.utils import convert_lora_ckpt_to_safetensors  # noqa: PLC0415
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    convert_lora_ckpt_to_safetensors(str(src), str(tmp))
    tmp.replace(dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
