pub mod base;
pub mod builder;
pub mod config;
pub mod evaluation;
pub mod plot;
pub mod traits;

pub use base::FuzzyBaseModel;
pub use config::ModelConfig;
pub use traits::{FuzzyModel, TrainableModel};
