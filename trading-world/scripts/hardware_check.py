"""Print the machine facts needed for the training resource plan (architecture §16)."""
import os
import platform
import shutil
import subprocess

print("platform :", platform.platform())
print("python   :", platform.python_version())
print("cpu      :", platform.processor() or platform.machine(), f"{os.cpu_count()} threads")
try:
    import psutil  # optional
    print("ram GB   :", round(psutil.virtual_memory().total / 1e9, 1))
except ImportError:
    if hasattr(os, "sysconf") and "SC_PHYS_PAGES" in os.sysconf_names:
        print("ram GB   :", round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1))
    else:
        print("ram GB   : unknown (pip install psutil, or check Task Manager / About This Mac)")
print("disk free:", round(shutil.disk_usage(".").free / 1e9), "GB")
try:
    import torch
    print("torch    :", torch.__version__, "cuda:", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
except ImportError:
    print("torch    : not installed")
try:
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True, timeout=10).stdout.strip() or "nvidia-smi: no GPU")
except (FileNotFoundError, subprocess.TimeoutExpired):
    print("nvidia-smi: not available")
