//! Lossless removal of inserted duplicate frames from H.264 streams.
//!
//! A port of the Python `frame_dedup` package (v1.2.0). Module names follow
//! the Python modules they replace; see `port/PORTING.md`.

pub mod bits;
pub mod cabac;
pub mod cadence;
pub mod decode;
pub mod dev;
pub mod evidence;
pub mod explicit;
pub mod h264hdr;
pub mod mux;
pub mod mv;
pub mod patch;
pub mod picture;
pub mod pipeline;
pub mod plan;
pub mod rebuild;
pub mod reexpress;
pub mod records;
pub mod repair;
pub mod stream;
pub mod system;
pub mod verify;

pub const VERSION: &str = "2.0.0-dev";
