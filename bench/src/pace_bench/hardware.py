import os
import re
import shutil
import subprocess
from pathlib import Path

from pace_bench.models import Cpu, Gpu, Hardware, Memory


def cpu() -> Cpu:
    cpuinfo = Path("/proc/cpuinfo").read_text()
    model = re.search(r"^model name\s*: (.+)$", cpuinfo, re.MULTILINE)
    if model is None:
        raise RuntimeError("No model name in /proc/cpuinfo")
    # Counts only the threads this process may run on, as allocated by Slurm
    threads = os.process_cpu_count()
    if threads is None:
        raise RuntimeError("Unable to count usable CPU threads")
    return Cpu(model=model[1].strip(), threads=threads)


def gpu() -> Gpu | None:
    if shutil.which("nvidia-smi"):
        query = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
        # Return code 6 means no GPU was found, as without a Slurm GPU allocation
        if query.returncode == 6:
            return None
        query.check_returncode()
        names = query.stdout.splitlines()
    elif shutil.which("amd-smi"):
        output = subprocess.run(
            ["amd-smi", "static", "--asic"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        names = re.findall(r"^\s*MARKET_NAME: (.+)$", output, re.MULTILINE)
    else:
        return None
    return Gpu(model=names[0].strip()) if names else None


def memory() -> Memory:
    # Installed memory per memory controller, as reported by the EDAC driver
    sizes = Path("/sys/devices/system/edac/mc").glob("mc*/size_mb")
    gib = sum(int(size.read_text()) for size in sizes) >> 10
    dmi = Path("/run/udev/data/+dmi:id").read_text()
    speed = re.search(r"_CONFIGURED_SPEED_MTS=(\d+)", dmi)
    if speed is None:
        raise RuntimeError("No configured memory speed in udev DMI data")
    return Memory(gib=gib, mts=int(speed[1]))


def detect() -> Hardware:
    """Describe the machine this process runs on."""
    return Hardware(cpu=cpu(), gpu=gpu(), memory=memory())
