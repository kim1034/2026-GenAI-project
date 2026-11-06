"""Report software and CUDA readiness without downloading a model or dataset."""

import importlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path


PACKAGES = ("numpy", "scipy", "sklearn", "yaml", "pytest", "matplotlib", "statsmodels", "torch", "transformers")


def collect() -> dict:
    result = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(), "platform": platform.platform(), "packages": {}}
    for name in PACKAGES:
        try:
            module = importlib.import_module(name)
            result["packages"][name] = getattr(module, "__version__", "installed")
        except ImportError:
            result["packages"][name] = None
    if result["packages"]["torch"] is not None:
        import torch

        result["cuda_available"] = torch.cuda.is_available()
        result["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    else:
        result["cuda_available"] = False
        result["gpu_name"] = None
    return result


def main() -> None:
    report = collect()
    output = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    path = Path("reports/env.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(output, encoding="utf-8")
    sys.stdout.write(output)


if __name__ == "__main__":
    main()
