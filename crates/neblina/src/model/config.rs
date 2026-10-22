use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use crate::error::ModelError;
use crate::fuzzy::granularity::Granularity;
use crate::membership::mb_function::{FuzzyStrategy, MFKind};

/// Configuration parameters for an individual feature/variable.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeatureConfig {
    /// Optional partition granularity (can be inferred from custom knots if omitted).
    pub granularity: Option<Granularity>,
    /// Partition strategy (linear domain bounds or explicit custom grid knots).
    pub strategy: FuzzyStrategy,
    /// Optional custom linguistic labels for the membership functions.
    #[serde(default)]
    pub labels: Vec<String>,
}

/// Consolidated model configuration mapped by feature name to prevent JSON key redundancy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub kind: MFKind,
    pub features: HashMap<String, FeatureConfig>,
    pub target: Option<FeatureConfig>,
}

impl ModelConfig {
    /// Deserializes a `ModelConfig` directly from a JSON string.
    pub fn from_json(json_str: &str) -> Result<Self, ModelError> {
        serde_json::from_str(json_str).map_err(|e| {
            ModelError::InvalidStrategy(format!("Failed to parse configuration JSON: {}", e))
        })
    }

    /// Reads and deserializes a `ModelConfig` from a JSON file path.
    pub fn from_json_file<P: AsRef<Path>>(path: P) -> Result<Self, ModelError> {
        let file = File::open(path).map_err(|e| {
            ModelError::InvalidStrategy(format!("Failed to open configuration file: {}", e))
        })?;
        let reader = BufReader::new(file);

        serde_json::from_reader(reader).map_err(|e| {
            ModelError::InvalidStrategy(format!("Failed to parse configuration file: {}", e))
        })
    }
}
