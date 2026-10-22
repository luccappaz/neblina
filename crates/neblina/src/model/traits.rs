use crate::dataset::{TabularDataset, TargetVector};
use crate::error::ModelError;
use crate::model::base::FuzzyBaseModel;

/// Universal contract for any fuzzy inference model.
pub trait FuzzyModel {
    type Output;

    /// Access the base configuration of input spaces, MFs, and partitions.
    fn base_model(&self) -> &FuzzyBaseModel;

    /// Mutable access if configuration modification is needed.
    fn base_model_mut(&mut self) -> &mut FuzzyBaseModel;

    /// Generic batch prediction method accepting any TabularDataset implementation.
    fn predict<D: TabularDataset>(
        &mut self,
        input: &D,
        threshold: Option<f64>,
        fallback: Option<f64>,
    ) -> Result<Self::Output, ModelError>;
}

/// Contract for models that learn or generate rules from a dataset.
pub trait TrainableModel: FuzzyModel {
    type Config;

    /// Executes the training or rule-generation algorithm.
    fn train(
        &mut self,
        x_train: &dyn TabularDataset,
        y_train: &dyn TargetVector,
        config: &Self::Config,
    ) -> Result<(), ModelError>;
}
