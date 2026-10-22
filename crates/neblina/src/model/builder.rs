use crate::error::ModelError;
use crate::fuzzy::granularity::Granularity;
use crate::membership::mb_function::{FuzzyStrategy, MFKind};
use crate::model::base::FuzzyBaseModel;
use std::collections::HashMap;

/// Trait providing mutable access to the underlying fuzzy base model.
pub trait HasFuzzyBase {
    fn base_mut(&mut self) -> &mut FuzzyBaseModel;
}

/// Extended trait providing all input configuration builder methods
/// automatically for any type that implements `HasFuzzyBase`.
pub trait FuzzyBuilder: HasFuzzyBase {
    fn set_kind(
        &mut self,
        kind: impl TryInto<MFKind, Error = ModelError>,
    ) -> Result<&mut Self, ModelError> {
        self.base_mut().set_kind(kind)?;
        Ok(self)
    }

    fn add_labels<S: Into<String>>(
        &mut self,
        feature: impl Into<String>,
        labels: Vec<S>,
    ) -> &mut Self {
        self.base_mut().add_labels(feature, labels);
        self
    }

    fn set_labels(&mut self, labels: HashMap<String, Vec<String>>) -> &mut Self {
        self.base_mut().set_labels(labels);
        self
    }

    fn set_strategies(&mut self, strategies: HashMap<String, FuzzyStrategy>) -> &mut Self {
        self.base_mut().set_strategies(strategies);
        self
    }

    fn add_linear_strategy(
        &mut self,
        feature: impl Into<String>,
        min: f64,
        max: f64,
    ) -> Result<&mut Self, ModelError> {
        self.base_mut().add_linear_strategy(feature, min, max)?;
        Ok(self)
    }

    fn add_custom_strategy<T: Into<Vec<f64>>>(
        &mut self,
        feature: impl Into<String>,
        points: T,
    ) -> Result<&mut Self, ModelError> {
        self.base_mut().add_custom_strategy(feature, points)?;
        Ok(self)
    }

    fn set_granularities(&mut self, granularities: HashMap<String, Granularity>) -> &mut Self {
        self.base_mut().set_granularities(granularities);
        self
    }

    fn add_granularity(
        &mut self,
        feature: impl Into<String>,
        granularity: impl TryInto<Granularity, Error = ModelError>,
    ) -> Result<&mut Self, ModelError> {
        self.base_mut().add_granularity(feature, granularity)?;
        Ok(self)
    }

    fn with_limits(&mut self, limits: HashMap<String, Vec<f64>>) -> &mut Self {
        self.base_mut().with_limits(limits);
        self
    }

    fn add_limit(
        &mut self,
        feature: impl Into<String>,
        limits: impl Into<Vec<f64>>,
    ) -> Result<&mut Self, ModelError> {
        self.base_mut().add_limit(feature, limits)?;
        Ok(self)
    }
}

// Blanket implementation: any struct implementing HasFuzzyBase gets FuzzyBuilder for free!
impl<T: HasFuzzyBase> FuzzyBuilder for T {}
