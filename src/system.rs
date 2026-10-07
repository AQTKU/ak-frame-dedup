//! Resource controls: process priority, RAM, worker limits, logging.
//! Port of `frame_dedup/system.py`.

use std::process::Command;

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
