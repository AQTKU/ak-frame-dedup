//! Resource controls: process priority, RAM, worker limits, logging.
//! Port of `frame_dedup/system.py`.
//!
//! The tool runs CPU-heavy workers (CABAC parsing, repair, verification) and
//! ffmpeg decoders. By default it uses half the logical CPUs, keeps at most
//! that many full-resolution ffmpeg decoders alive at once (each with few
//! threads), sizes memory-heavy pools so that workers x peak stays under ~40%
//! of physical RAM, and runs everything at below-normal priority.
//!
//! Unlike the Python, workers are threads of one process, so there is no
//! `worker_init` (the process priority is lowered once) and `peak_rss` is a
//! process-wide figure.

use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::process::Command;
use std::sync::Mutex;
use std::time::Instant;

/// Threads per full-res decoder when many run at once.
pub const FFMPEG_THREADS: usize = 2;
/// Budget for a pool: workers x peak <= this x physical RAM.
pub const RAM_FRACTION: f64 = 0.40;
/// Budget per repair worker (measured peak ~240 MB on sample.mkv).
pub const REPAIR_PEAK: f64 = 0.8e9;
/// Explicit-mode repair worker (holds a whole IDR period's frames).
pub const REPAIR_PEAK_EXPLICIT: f64 = 1.5e9;
/// Per-worker peak of the measured repair with residual corrections: it also
/// keeps reference frames, corrected records and their target samples.
pub const REPAIR_PEAK_CORR: f64 = 2.2e9;
/// Budget per verify worker (measured peak ~235 MB).
pub const VERIFY_PEAK: f64 = 0.5e9;

/// A child process that starts at below-normal priority (Python: `popen_kwargs()`).
/// Every ffmpeg / ffprobe / mkvmerge invocation goes through this.
pub fn command(program: &str) -> Command {
    #[allow(unused_mut)]
    let mut c = Command::new(program);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const BELOW_NORMAL_PRIORITY_CLASS: u32 = 0x0000_4000;
        c.creation_flags(BELOW_NORMAL_PRIORITY_CLASS);
    }
    c
}

/// Half the logical CPUs, at least 1.
pub fn default_workers() -> usize {
    let n = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(2);
    (n / 2).max(1)
}

/// Run the calling process at below-normal priority (best effort; a no-op off Windows).
pub fn lower_priority() {
    #[cfg(windows)]
    // SAFETY: plain Win32 calls on the current-process pseudo handle.
    unsafe {
        use windows_sys::Win32::System::Threading::{
            BELOW_NORMAL_PRIORITY_CLASS, GetCurrentProcess, SetPriorityClass,
        };
        SetPriorityClass(GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS);
    }
}

/// (total, available) physical memory in bytes. Falls back to (16 GiB, 8 GiB)
/// when unknown, like the Python.
pub fn physical_ram() -> (u64, u64) {
    #[cfg(windows)]
    {
        use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
        // SAFETY: MEMORYSTATUSEX is plain data; dwLength is set as the API requires.
        unsafe {
            let mut m: MEMORYSTATUSEX = std::mem::zeroed();
            m.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
            if GlobalMemoryStatusEx(&mut m) != 0 {
                return (m.ullTotalPhys, m.ullAvailPhys);
            }
        }
    }
    #[cfg(not(windows))]
    if let Ok(s) = std::fs::read_to_string("/proc/meminfo") {
        let field = |name: &str| -> Option<u64> {
            let l = s.lines().find(|l| l.starts_with(name))?;
            Some(l.split_whitespace().nth(1)?.parse::<u64>().ok()? * 1024)
        };
        if let (Some(t), Some(a)) = (field("MemTotal:"), field("MemAvailable:")) {
            return (t, a);
        }
    }
    (16 << 30, 8 << 30)
}

/// Peak working set of the calling *process* in bytes (0 if unknown).
///
/// With workers as threads this is the figure for the whole run; there is no
/// per-worker peak. Callers that used to report a worker's peak should report
/// this process peak (or drop the per-worker figure).
pub fn peak_rss() -> u64 {
    #[cfg(windows)]
    {
        use windows_sys::Win32::System::ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        // SAFETY: PROCESS_MEMORY_COUNTERS is plain data; cb is set as the API requires.
        unsafe {
            let mut c: PROCESS_MEMORY_COUNTERS = std::mem::zeroed();
            c.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
            if GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) != 0 {
                return c.PeakWorkingSetSize as u64;
            }
        }
        0
    }
    #[cfg(not(windows))]
    {
        std::fs::read_to_string("/proc/self/status")
            .ok()
            .and_then(|s| {
                let l = s.lines().find(|l| l.starts_with("VmHWM:"))?;
                Some(l.split_whitespace().nth(1)?.parse::<u64>().ok()? * 1024)
            })
            .unwrap_or(0)
    }
}

/// Python's float floor division `a // b` (positive operands).
fn py_floordiv(a: f64, b: f64) -> f64 {
    let m = a % b;
    let div = (a - m) / b;
    if div == 0.0 {
        return div;
    }
    let f = div.floor();
    if div - f > 0.5 { f + 1.0 } else { f }
}

/// Workers for a memory-heavy pool: `requested`, capped so that workers x
/// `per_worker_peak` stays within `RAM_FRACTION` of physical RAM and by the
/// number of tasks. Like the Python, `n_tasks` of `Some(0)` means "unlimited".
pub fn pool_size(requested: usize, per_worker_peak: f64, n_tasks: Option<usize>) -> usize {
    pool_size_for(physical_ram().0, requested, per_worker_peak, n_tasks)
}

fn pool_size_for(total_ram: u64, requested: usize, per_worker_peak: f64, n_tasks: Option<usize>) -> usize {
    let cap = (py_floordiv(total_ram as f64 * RAM_FRACTION, per_worker_peak) as usize).max(1);
    let n = requested.min(cap).max(1);
    match n_tasks {
        Some(t) if t > 0 => n.min(t),
        _ => n,
    }
}

/// Timestamped, flushed log lines to stdout and to a log file. Usable from
/// several threads (`&self`); each line is written atomically.
pub struct Logger {
    t0: Instant,
    file: Mutex<Option<File>>,
}

impl Logger {
    /// Starts the clock; appends to `path` (creating its directory) if given.
    pub fn new(path: Option<&Path>) -> std::io::Result<Logger> {
        let file = match path {
            Some(p) => {
                if let Some(d) = p.parent() {
                    if !d.as_os_str().is_empty() {
                        std::fs::create_dir_all(d)?;
                    }
                }
                Some(std::fs::OpenOptions::new().create(true).append(true).open(p)?)
            }
            None => None,
        };
        Ok(Logger { t0: Instant::now(), file: Mutex::new(file) })
    }

    /// The text of a log line for a given elapsed time: `[   12.3s] msg`.
    pub fn format_line(elapsed_s: f64, msg: &str) -> String {
        format!("[{elapsed_s:7.1}s] {msg}")
    }

    pub fn elapsed(&self) -> f64 {
        self.t0.elapsed().as_secs_f64()
    }

    pub fn log(&self, msg: &str) {
        let line = Self::format_line(self.elapsed(), msg);
        // Hold the lock across the stdout write too, so lines never interleave.
        let mut f = self.file.lock().unwrap_or_else(|e| e.into_inner());
        let mut out = std::io::stdout().lock();
        let _ = writeln!(out, "{line}");
        let _ = out.flush();
        if let Some(f) = f.as_mut() {
            let _ = writeln!(f, "{line}");
            let _ = f.flush();
        }
    }

    pub fn close(&self) {
        *self.file.lock().unwrap_or_else(|e| e.into_inner()) = None;
    }
}

/// `frame-dedup dev sysinfo`: physical RAM, peak RSS before/after touching 256 MiB, pool sizes.
pub fn dev_sysinfo() -> anyhow::Result<()> {
    let (t, a) = physical_ram();
    println!("physical_ram total={t} available={a} ({:.1} GiB / {:.1} GiB)", t as f64 / 2f64.powi(30), a as f64 / 2f64.powi(30));
    println!("default_workers={}", default_workers());
    println!("peak_rss before={}", peak_rss());
    let mut v = vec![0u8; 256 << 20];
    for i in (0..v.len()).step_by(4096) {
        v[i] = 1;
    }
    println!("peak_rss after touching 256 MiB={} (buffer sum {})", peak_rss(), v.iter().step_by(4096).map(|&b| b as u64).sum::<u64>());
    println!("pool_size(64, REPAIR_PEAK)={}", pool_size(64, REPAIR_PEAK, None));
    println!("pool_size(64, REPAIR_PEAK_EXPLICIT)={}", pool_size(64, REPAIR_PEAK_EXPLICIT, None));
    println!("pool_size(64, VERIFY_PEAK, 3 tasks)={}", pool_size(64, VERIFY_PEAK, Some(3)));
    lower_priority();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_size_caps() {
        let gb = 1u64 << 30;
        // 32 GiB x 0.4 = 13.7e9 budget; 0.8e9 per worker -> 17 workers
        assert_eq!(pool_size_for(32 * gb, 64, REPAIR_PEAK, None), 17);
        assert_eq!(pool_size_for(32 * gb, 8, REPAIR_PEAK, None), 8);
        assert_eq!(pool_size_for(32 * gb, 64, REPAIR_PEAK, Some(3)), 3);
        // zero tasks means unlimited, as `if n_tasks:` in the Python
        assert_eq!(pool_size_for(32 * gb, 5, REPAIR_PEAK, Some(0)), 5);
        // tiny RAM still gives one worker
        assert_eq!(pool_size_for(gb, 4, REPAIR_PEAK_EXPLICIT, None), 1);
        assert_eq!(pool_size_for(gb, 0, REPAIR_PEAK, None), 1);
        // 16 GiB x 0.4 / 1.5e9 = 4.58 -> 4
        assert_eq!(pool_size_for(16 * gb, 32, REPAIR_PEAK_EXPLICIT, None), 4);
    }

    #[test]
    fn log_line_format() {
        assert_eq!(Logger::format_line(12.34, "hi"), "[   12.3s] hi");
        assert_eq!(Logger::format_line(0.0, "x"), "[    0.0s] x");
    }

    #[test]
    fn ram_and_workers_sane() {
        let (t, a) = physical_ram();
        assert!(t > 0 && a <= t);
        assert!(default_workers() >= 1);
    }
}
