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


def run(*args: str) -> subprocess.CompletedProcess[str] | None:
    if shutil.which(args[0]) is None:
        return None
    return subprocess.run(args, capture_output=True, text=True, check=False)


def nvidia() -> str | None:
    query = run("nvidia-smi", "--query-gpu=name", "--format=csv,noheader")
    # nvidia-smi returns 6 when no GPU is found
    if query is None or query.returncode == 6:
        return None
    query.check_returncode()
    return query.stdout.splitlines()[0]


def amd() -> str | None:
    query = run("amd-smi", "static", "--asic")
    # amd-smi returns 255 when no GPU is found
    if query is None or query.returncode == 255:
        return None
    query.check_returncode()
    name = re.search(r"^\s*MARKET_NAME: (.+)$", query.stdout, re.MULTILINE)
    return name[1].strip() if name else None


def gpu() -> Gpu | None:
    model = nvidia() or amd()
    return Gpu(model=model) if model else None


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
