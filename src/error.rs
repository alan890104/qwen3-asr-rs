use thiserror::Error;

#[non_exhaustive]
#[derive(Error, Debug)]
pub enum AsrError {
    #[error("Model load failed: {0}")]
    ModelLoad(#[source] anyhow::Error),

    #[error("Audio decode failed: {0}")]
    AudioDecode(#[source] anyhow::Error),

    #[error("Inference failed: {0}")]
    Inference(#[source] anyhow::Error),
}

pub type Result<T> = std::result::Result<T, AsrError>;
