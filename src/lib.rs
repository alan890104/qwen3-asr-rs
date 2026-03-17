mod config;
mod decoder;
mod encoder;
mod error;
#[cfg(feature = "hub")]
pub(crate) mod hub;
mod inference;
mod linear;
mod mel;
mod streaming;

pub use encoder::EncoderCache;
pub use error::{AsrError, Result};
pub use inference::{AsrInference, TranscribeOptions, TranscribeResult};
pub use mel::load_audio_wav;
pub use streaming::{StreamingOptions, StreamingState};

/// Select the best available device: CUDA (if `cuda` feature) → Metal (if `metal` feature) → CPU.
///
/// Logs the selected device at `info` level and any fallback at `warn` level.
pub fn best_device() -> candle_core::Device {
    #[cfg(feature = "cuda")]
    {
        match candle_core::Device::new_cuda(0) {
            Ok(device) => {
                log::info!("Using CUDA device 0");
                return device;
            }
            Err(e) => {
                log::warn!("CUDA feature enabled but device creation failed: {e}, falling back");
            }
        }
    }
    #[cfg(feature = "metal")]
    {
        match candle_core::Device::new_metal(0) {
            Ok(device) => {
                log::info!("Using Metal device");
                return device;
            }
            Err(e) => {
                log::warn!("Metal feature enabled but device creation failed: {e}, falling back");
            }
        }
    }
    log::info!("Using CPU device");
    candle_core::Device::Cpu
}
