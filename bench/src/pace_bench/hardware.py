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


def nvidia() -> str | None:
    if not shutil.which("nvidia-smi"):
        return None
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
    return names[0] if names else None


def amd() -> str | None:
    if not shutil.which("amd-smi"):
        return None
    query = subprocess.run(
        ["amd-smi", "static", "--asic"], capture_output=True, text=True, check=False
    )
    # Fails without a GPU, as the device files are denied without a Slurm GPU allocation
    if query.returncode:
        return None
    name = re.search(r"^\s*MARKET_NAME: (.+)$", query.stdout, re.MULTILINE)
    return name[1].strip() if name else None


def gpu() -> Gpu | None:
    model = nvidia() or amd()
    return Gpu(model=model) if model else None


def memory() -> Memory:
    # Installed memory per memory controller, as reported by the EDAC driver
    sizes = list(Path("/sys/devices/system/edac/mc").glob("mc*/size_mb"))
    if not sizes:
        raise RuntimeError("No EDAC memory controllers")
    gib = sum(int(size.read_text()) for size in sizes) >> 10
    dmi = Path("/run/udev/data/+dmi:id").read_text()
    speeds = re.findall(
        r"^E:MEMORY_DEVICE_\d+_CONFIGURED_SPEED_MTS=(\d+)$", dmi, re.MULTILINE
    )
    if not speeds:
        raise RuntimeError("No configured memory speed in udev DMI data")
    return Memory(gib=gib, mts=min(map(int, speeds)))


def detect() -> Hardware:
    """Describe the machine this process runs on."""
    return Hardware(cpu=cpu(), gpu=gpu(), memory=memory())
