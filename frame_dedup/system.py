"""Resource controls: process priority, RAM, worker limits, logging.

The tool runs many CPU-heavy processes (Python CABAC workers and ffmpeg
decoders). By default it uses half the logical CPUs, keeps at most that many
full-resolution ffmpeg decoders alive at once (each with few threads), sizes
memory-heavy pools so that workers x peak stays under ~40% of physical RAM,
and runs everything at below-normal priority so the machine stays usable.
"""
import ctypes
import os
import subprocess
import sys
import time

IS_WIN = sys.platform == 'win32'
BELOW_NORMAL = 0x00004000  # BELOW_NORMAL_PRIORITY_CLASS

FFMPEG_THREADS = 2          # threads per full-res decoder when many run at once
RAM_FRACTION = 0.40         # budget for a pool: workers x peak <= this x physical RAM
REPAIR_PEAK = 1.6e9         # measured peak working set of one repair worker (bytes), with margin
VERIFY_PEAK = 0.4e9         # same for a verify worker


def default_workers():
    return max(1, (os.cpu_count() or 2) // 2)


def lower_priority():
    """Run the calling process at below-normal priority."""
    try:
        if IS_WIN:
            k = ctypes.windll.kernel32
            k.SetPriorityClass(k.GetCurrentProcess(), BELOW_NORMAL)
        else:
            os.nice(5)
    except Exception:
        pass


def popen_kwargs():
    """Keyword arguments that start a child process at below-normal priority."""
    return {'creationflags': subprocess.BELOW_NORMAL_PRIORITY_CLASS} if IS_WIN else {}


class _MemStatus(ctypes.Structure):
    _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong),
                ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]


def physical_ram():
    """(total, available) physical memory in bytes."""
    try:
        if IS_WIN:
            m = _MemStatus()
            m.dwLength = ctypes.sizeof(m)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return int(m.ullTotalPhys), int(m.ullAvailPhys)
        pg = os.sysconf('SC_PAGE_SIZE')
        return pg * os.sysconf('SC_PHYS_PAGES'), pg * os.sysconf('SC_AVPHYS_PAGES')
    except Exception:
        return 16 << 30, 8 << 30


class _PMC(ctypes.Structure):
    _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong),
                ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]


def peak_rss():
    """Peak working set of the calling process in bytes (0 if unknown)."""
    try:
        if IS_WIN:
            c = _PMC()
            c.cb = ctypes.sizeof(c)
            k = ctypes.windll.kernel32
            psapi = ctypes.windll.psapi
            psapi.GetProcessMemoryInfo(ctypes.c_void_p(k.GetCurrentProcess()), ctypes.byref(c), c.cb)
            return int(c.PeakWorkingSetSize)
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    except Exception:
        return 0


def pool_size(requested, per_worker_peak, n_tasks=None):
    total, _ = physical_ram()
    cap = max(1, int(total * RAM_FRACTION // per_worker_peak))
    n = max(1, min(requested, cap))
    return min(n, n_tasks) if n_tasks else n


def worker_init():
    lower_priority()


class Logger:
    """Timestamped, flushed log lines to stdout and to a log file."""

    def __init__(self, path=None):
        self.t0 = time.time()
        self.f = None
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            self.f = open(path, 'a', encoding='utf-8', buffering=1)

    def __call__(self, msg):
        line = f'[{time.time() - self.t0:7.1f}s] {msg}'
        print(line, flush=True)
        if self.f:
            self.f.write(line + '\n')
            self.f.flush()

    def close(self):
        if self.f:
            self.f.close()
            self.f = None
