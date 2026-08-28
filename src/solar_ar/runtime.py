"""CPU/RAM budgeting so training actually saturates the machine it is given.

The target box for this project is 15 GB RAM and 4 CPU cores, and the goal is to
keep all of it busy. Two things normally stop that from happening:

* **Idle cores.** PyTorch picks a thread count before the process knows its
  budget, and the BLAS libraries read ``OMP_NUM_THREADS`` at *import* time. If
  the count is wrong the math kernels quietly run single-threaded.
* **Idle cores waiting on I/O.** Decoding HDF5/FITS tiles is slow enough that a
  single-process data loader starves the model. Workers and RAM caching fix it.

This module detects the real limits (including cgroup limits, which is what a
container actually enforces), clamps them to the configured budget, and hands
back a plan the trainer can apply.
"""

from __future__ import annotations

import functools
import os
from dataclasses import asdict, dataclass
from pathlib import Path

#: Project defaults: if <= 0, we use all detected resources.
DEFAULT_MEMORY_BUDGET_GB = 0.0
DEFAULT_CPU_BUDGET = 0

#: Automatic headroom reserved so the OS and other apps stay responsive while
#: training saturates the machine. Applied only when the corresponding budget
#: is left at its "auto" default (<= 0); an explicit --cpu-budget or
#: --memory-budget-gb always wins. The default is 2 cores: the run grabs
#: maximum power (every core minus two, all RAM) while a Mac/box you keep
#: working on stays usable. Pass 0 to grab literally every core on a
#: dedicated training box.
DEFAULT_CPU_HEADROOM = 2
DEFAULT_MEMORY_HEADROOM_GB = 0.0

#: Environment variables read by the numeric libraries at import time.
_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)

_BYTES_PER_GB = 1024 ** 3


@dataclass
class ResourcePlan:
    """Concrete resource decisions, logged to the run directory for debugging."""

    cpu_count_detected: int
    cpu_budget: int
    torch_threads: int
    interop_threads: int
    memory_detected_gb: float
    memory_budget_gb: float
    disk_free_gb: float
    dataloader_workers: int
    prefetch_factor: int | None
    persistent_workers: bool
    pin_memory: bool
    cache_budget_bytes: int

    def describe(self) -> str:
        return (
            f"CPU: using {self.cpu_budget} of {self.cpu_count_detected} cores "
            f"(torch threads={self.torch_threads}, interop={self.interop_threads}, "
            f"loader workers={self.dataloader_workers})\n"
            f"RAM: budget {self.memory_budget_gb:.1f} GB of {self.memory_detected_gb:.1f} GB detected "
            f"(sample cache {self.cache_budget_bytes / _BYTES_PER_GB:.2f} GB)\n"
            f"DISK: {self.disk_free_gb:.1f} GB available"
        )


def _read_int(path: str) -> int | None:
    try:
        text = Path(path).read_text().strip()
    except (OSError, ValueError):
        return None
    if text in {"max", "-1", ""}:
        return None
    try:
        value = int(text)
    except ValueError:
        return None
    # cgroup v1 uses a huge sentinel for "unlimited".
    return value if 0 < value < 2 ** 62 else None


def detect_cpu_count() -> int:
    """Return usable cores, honouring cgroup quota and CPU affinity.

    ``os.cpu_count()`` reports the host's cores, not the container's share, so a
    4-core container on a 64-core host would otherwise spawn 64 threads and lose
    time to context switching.
    """
    candidates: list[int] = []

    # cgroup v2: "<quota> <period>", quota may be "max".
    try:
        quota_text = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if len(quota_text) == 2 and quota_text[0] != "max":
            quota, period = int(quota_text[0]), int(quota_text[1])
            if quota > 0 and period > 0:
                candidates.append(max(1, quota // period))
    except (OSError, ValueError):
        pass

    # cgroup v1
    quota = _read_int("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    period = _read_int("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    if quota and period:
        candidates.append(max(1, quota // period))

    # Scheduler affinity mask (taskset / cpuset).
    try:
        candidates.append(len(os.sched_getaffinity(0)))
    except AttributeError:  # pragma: no cover - non-Linux
        pass

    candidates.append(os.cpu_count() or 1)
    return max(1, min(candidates))


def detect_disk_gb() -> float:
    """Return free disk space in GB for the current directory."""
    import shutil
    try:
        total, used, free = shutil.disk_usage(".")
        return free / _BYTES_PER_GB
    except (OSError, ValueError):
        return 0.0


def detect_memory_gb() -> float:
    """Return usable RAM in GB, honouring cgroup limits."""
    candidates: list[float] = []

    for cgroup_path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        limit = _read_int(cgroup_path)
        if limit:
            candidates.append(limit / _BYTES_PER_GB)

    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
        candidates.append(page_size * page_count / _BYTES_PER_GB)
    except (AttributeError, ValueError, OSError):  # pragma: no cover - non-Linux
        pass

    return min(candidates) if candidates else 4.0


def available_memory_gb() -> float:
    """Best-effort free memory, used to keep the cache from pushing into swap."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / (1024 ** 2)
    except (OSError, ValueError, IndexError):  # pragma: no cover
        pass
    return detect_memory_gb()


def plan_resources(
    cpu_budget: int | None = None,
    memory_budget_gb: float | None = None,
    dataloader_workers: int | None = None,
    cache_fraction: float = 0.45,
    use_cuda: bool = False,
    cpu_headroom: int = DEFAULT_CPU_HEADROOM,
    memory_headroom_gb: float = DEFAULT_MEMORY_HEADROOM_GB,
) -> ResourcePlan:
    """Decide thread counts, worker counts and cache size for this run.

    ``cpu_budget``/``memory_budget_gb`` are *ceilings*: the plan never exceeds
    them, and never exceeds what the machine actually has. A non-positive
    budget means "auto" and selects ``detected - headroom``, so the machine
    stays usable (e.g. 8-core laptop -> 6 training cores) while still being
    saturated. Pass an explicit positive budget to override.
    """
    detected_cpus = detect_cpu_count()
    requested_cpus = cpu_budget if cpu_budget is not None and cpu_budget > 0 else 0
    if requested_cpus <= 0:
        # Auto: keep the requested headroom free for the OS / other apps.
        effective_cpus = max(1, detected_cpus - max(0, cpu_headroom))
    else:
        effective_cpus = max(1, min(requested_cpus, detected_cpus))

    detected_memory = detect_memory_gb()
    requested_memory = memory_budget_gb if memory_budget_gb is not None and memory_budget_gb > 0 else 0.0
    if requested_memory <= 0:
        effective_memory = max(0.5, detected_memory - max(0.0, memory_headroom_gb))
    else:
        effective_memory = max(0.5, min(requested_memory, detected_memory))

    if dataloader_workers is None or dataloader_workers < 0:
        # Keep one core for the main process (which runs the model's math) and
        # spend the rest on decoding. With <=2 cores, in-process loading wins
        # because worker startup and IPC cost more than they save.
        workers = 0 if effective_cpus <= 2 else effective_cpus - 1
    else:
        # Clamp to the core budget: more workers than cores oversubscribes the
        # CPU and makes each worker slower than it would be alone.
        workers = min(dataloader_workers, effective_cpus)

    # Reserve headroom for the model, optimizer state, autograd graph and the
    # copy each worker holds; the cache only gets a fraction of the budget.
    cache_budget = int(effective_memory * cache_fraction * _BYTES_PER_GB)

    return ResourcePlan(
        cpu_count_detected=detected_cpus,
        cpu_budget=effective_cpus,
        torch_threads=effective_cpus,
        # Inter-op parallelism competes with intra-op for the same cores; a
        # small fixed value avoids oversubscription.
        interop_threads=1 if effective_cpus <= 2 else 2,
        memory_detected_gb=detected_memory,
        memory_budget_gb=effective_memory,
        disk_free_gb=detect_disk_gb(),
        dataloader_workers=workers,
        # 8 prefetched batches per worker keeps every core fed: decoding
        # FITS/HDF5 tiles is the slow part, so workers stay ahead of the model
        # instead of stalling it between batches.
        prefetch_factor=8 if workers > 0 else None,
        persistent_workers=workers > 0,
        pin_memory=use_cuda,
        cache_budget_bytes=cache_budget,
    )


def apply_thread_environment(plan: ResourcePlan) -> None:
    """Export thread-count env vars.

    Call this **before** importing numpy/torch when possible: OpenMP and MKL
    read these at import time, so setting them afterwards has no effect on the
    already-initialised thread pools.
    """
    for name in _THREAD_ENV_VARS:
        os.environ[name] = str(plan.torch_threads)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    # Keep worker threads spinning briefly instead of sleeping between batches;
    # this measurably raises CPU utilisation on small tiles.
    os.environ.setdefault("OMP_WAIT_POLICY", "ACTIVE")
    os.environ.setdefault("KMP_BLOCKTIME", "1")


def preferred_device() -> str:
    """Return the best available accelerator: ``cuda``, ``mps``, or ``cpu``.

    Apple Silicon exposes its GPU through the Metal ``mps`` backend rather than
    CUDA; this lets the trainer and evaluator share one device decision so the
    same code runs on a Colab T4 (cuda) and a local Mac (mps) without edits.
    """
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def describe_accelerator() -> str:
    """One-line human summary of the accelerator the run will use."""
    import torch

    device = preferred_device()
    if device == "cuda":
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / _BYTES_PER_GB
        return f"GPU: CUDA {name} ({vram:.1f} GB VRAM)"
    if device == "mps":
        return "GPU: Apple Metal (MPS) using unified memory"
    return "GPU: none detected (CPU only)"


@functools.lru_cache(maxsize=1)
def _mps_autocast_supported() -> bool:
    """Probe whether this torch build accepts ``autocast(device_type="mps")``.

    MPS autocast (bfloat16) shipped in recent PyTorch releases; on older or
    CPU-only builds the constructor rejects the device type, so we fall back
    to plain fp32 instead of crashing the run.
    """
    import torch

    try:
        with torch.amp.autocast(device_type="mps", enabled=False):
            pass
        return True
    except (ValueError, RuntimeError, TypeError):
        return False


def amp_settings(device_type: str, enabled: bool = True) -> tuple[str, bool, bool]:
    """Decide the mixed-precision setup for a device: maximum speed, no NaNs.

    Returns ``(autocast_device_type, autocast_enabled, grad_scaler_enabled)``:

    * **CUDA** — classic fp16 autocast + GradScaler.
    * **MPS (Apple Silicon)** — bfloat16 autocast, which is the big free
      speedup on Mac GPUs. bfloat16 has fp32's exponent range so no GradScaler
      is needed (and none is provided for MPS).
    * **CPU / anything else** — plain fp32; everything disabled.

    Callers should always use all three values together so an unsupported
    build degrades to fp32 instead of mixing a scaler with a disabled autocast.
    """
    if not enabled:
        return "cpu", False, False
    if device_type == "cuda":
        return "cuda", True, True
    if device_type == "mps" and _mps_autocast_supported():
        return "mps", True, False
    return "cpu", False, False


def apply_torch_runtime(plan: ResourcePlan) -> None:
    """Apply the plan to an already-imported torch and enable CPU fast paths."""
    import torch

    torch.set_num_threads(plan.torch_threads)
    try:
        torch.set_num_interop_threads(plan.interop_threads)
    except RuntimeError:
        # Only settable before the first parallel region runs; harmless if late.
        pass

    # Denormal floats are catastrophically slow on x86; flushing them to zero is
    # safe for normalized image data.
    torch.set_flush_denormal(True)

    if torch.backends.mkldnn.is_available():
        torch.backends.mkldnn.enabled = True

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def configure_runtime(
    cpu_budget: int | None = None,
    memory_budget_gb: float | None = None,
    dataloader_workers: int | None = None,
    cache_fraction: float = 0.45,
    use_cuda: bool = False,
    cpu_headroom: int = DEFAULT_CPU_HEADROOM,
    memory_headroom_gb: float = DEFAULT_MEMORY_HEADROOM_GB,
) -> ResourcePlan:
    """Plan and apply the resource configuration in one call."""
    plan = plan_resources(
        cpu_budget=cpu_budget,
        memory_budget_gb=memory_budget_gb,
        dataloader_workers=dataloader_workers,
        cache_fraction=cache_fraction,
        use_cuda=use_cuda,
        cpu_headroom=cpu_headroom,
        memory_headroom_gb=memory_headroom_gb,
    )
    apply_thread_environment(plan)
    apply_torch_runtime(plan)
    return plan


def plan_to_dict(plan: ResourcePlan) -> dict[str, object]:
    return asdict(plan)


def process_memory_gb() -> float:
    """Resident set size of this process, for logging real usage per epoch."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / (1024 ** 2)
    except (OSError, ValueError, IndexError):  # pragma: no cover
        pass
    return 0.0


def suggest_batch_size(
    image_size: int,
    channels: int,
    base_channels: int,
    memory_budget_gb: float,
    minimum: int = 1,
    maximum: int = 64,
    data_parallel: bool = True,
) -> int:
    """Estimate the largest batch that fits the RAM budget.

    Activation memory for a U-Net scales with ``base_channels * H * W`` per
    sample; the constant below is an empirical fit that includes the autograd
    graph and the decoder's skip connections. Deliberately conservative -- it is
    a starting point for --auto-batch-size, not a hard guarantee.
    """
    import torch
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / _BYTES_PER_GB
        # Keep significant VRAM headroom for model weights, gradients, cuDNN
        # workspaces, and fragmentation. The estimate is per GPU.
        memory_budget_gb = min(memory_budget_gb, vram_gb * 0.55)

    bytes_per_sample = image_size * image_size * base_channels * 4 * 26
    bytes_per_sample += image_size * image_size * channels * 4 * 4
    # 0.70 of the (cache-free) budget: the per-sample constant above already
    # folds in the autograd graph, so the extra headroom buys a larger batch —
    # more throughput per epoch — while the 26x fit factor still covers the
    # decoder's skip-connection activations in practice.
    usable = memory_budget_gb * 0.70 * _BYTES_PER_GB
    estimate = int(usable // max(bytes_per_sample, 1))

    # DataParallel takes a global batch, distributed across the visible GPUs.
    # A single-GPU trainer (data_parallel=False, e.g. the streaming trainer)
    # runs the WHOLE batch on one GPU, so inflating it by the visible-GPU
    # count would OOM one of them (e.g. Kaggle's T4x2 runtime).
    num_gpus = (
        torch.cuda.device_count()
        if torch.cuda.is_available() and data_parallel
        else 1
    )
    if num_gpus > 1:
        estimate *= num_gpus

    return max(minimum, min(maximum, estimate))
