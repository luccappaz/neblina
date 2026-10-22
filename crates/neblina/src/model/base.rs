use std::collections::HashMap;

use crate::{
    error::ModelError,
    fuzzy::granularity::Granularity,
    membership::{
        FuzzyStrategy, MFKind, MembershipFunction,
        mb_function::{MembershipOp, Partitionable},
    },
};

#[derive(Debug, Default, Clone)]
pub struct FuzzyBaseModel {
    pub kind: Option<MFKind>,
    pub strategies: HashMap<String, FuzzyStrategy>,
    pub granularities: HashMap<String, Granularity>,
    pub labels: HashMap<String, Vec<String>>,
    pub limits: HashMap<String, Vec<f64>>,
    pub mf_configs: HashMap<String, HashMap<String, MembershipFunction>>,
    pub mf_centers: HashMap<String, HashMap<String, f64>>,
    pub mf_configs_built: bool,
}

impl FuzzyBaseModel {
    /// Creates a new `FuzzyBaseModel` instance.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns whether internal membership function structures have been built.
    pub fn is_configured(&self) -> bool {
        self.mf_configs_built
    }

    /// Flags the model as unbuilt (necessary when altering parameters).
    pub fn mark_unbuilt(&mut self) {
        self.mf_configs_built = false;
    }

    // =========================================================================
    // Input Configuration Methods (Builders)
    // =========================================================================

    /// Sets the membership function kind in place.
    pub fn set_kind(
        &mut self,
        kind: impl TryInto<MFKind, Error = ModelError>,
    ) -> Result<&mut Self, ModelError> {
        self.kind = Some(kind.try_into()?);
        self.mark_unbuilt();
        Ok(self)
    }

    /// Registers custom linguistic labels for a specific feature.
    pub fn add_labels<S: Into<String>>(
        &mut self,
        feature: impl Into<String>,
        labels: Vec<S>,
    ) -> &mut Self {
        let converted_labels: Vec<String> = labels.into_iter().map(Into::into).collect();
        self.labels.insert(feature.into(), converted_labels);
        self.mark_unbuilt();
        self
    }

    /// Registers a map of custom labels for all features at once.
    pub fn set_labels(&mut self, labels: HashMap<String, Vec<String>>) -> &mut Self {
        self.labels = labels;
        self.mark_unbuilt();
        self
    }

    /// Sets a pre-constructed `FuzzyStrategy` map globally.
    pub fn set_strategies(&mut self, strategies: HashMap<String, FuzzyStrategy>) -> &mut Self {
        self.strategies = strategies;
        self.mark_unbuilt();
        self
    }

    /// Configures a linear partition strategy for a specific feature.
    pub fn add_linear_strategy(
        &mut self,
        feature: impl Into<String>,
        min: f64,
        max: f64,
    ) -> Result<&mut Self, ModelError> {
        let feature_name = feature.into();
        self.strategies
            .insert(feature_name, FuzzyStrategy::linear(min, max)?);
        self.mark_unbuilt();
        Ok(self)
    }

    /// Configures a custom partition strategy for a specific feature.
    pub fn add_custom_strategy<T: Into<Vec<f64>>>(
        &mut self,
        feature: impl Into<String>,
        points: T,
    ) -> Result<&mut Self, ModelError> {
        let feature_name = feature.into();
        self.strategies
            .insert(feature_name, FuzzyStrategy::custom(points)?);
        self.mark_unbuilt();
        Ok(self)
    }

    /// Sets the granularity map for all features at once.
    pub fn set_granularities(&mut self, granularities: HashMap<String, Granularity>) -> &mut Self {
        self.granularities = granularities;
        self.mark_unbuilt();
        self
    }

    /// Adds a single feature's granularity.
    pub fn add_granularity(
        &mut self,
        feature: impl Into<String>,
        granularity: impl TryInto<Granularity, Error = ModelError>,
    ) -> Result<&mut Self, ModelError> {
        self.granularities
            .insert(feature.into(), granularity.try_into()?);
        self.mark_unbuilt();
        Ok(self)
    }

    /// Sets feature limits map in place.
    pub fn with_limits(&mut self, limits: HashMap<String, Vec<f64>>) -> &mut Self {
        self.limits = limits;
        self.mark_unbuilt();
        self
    }

    /// Adds limits (grid knots) for a feature after validating invariants against the configured MFKind.
    pub fn add_limit(
        &mut self,
        feature: impl Into<String>,
        limits: impl Into<Vec<f64>>,
    ) -> Result<&mut Self, ModelError> {
        let feature_name = feature.into();
        let knots: Vec<f64> = limits.into();

        if knots.len() < 2 {
            return Err(ModelError::InvalidStrategy(format!(
                "Limits array for feature '{}' must contain at least 2 points, got {}.",
                feature_name,
                knots.len()
            )));
        }

        if !knots.windows(2).all(|w| w[0] < w[1]) {
            return Err(ModelError::InvalidStrategy(format!(
                "Limits array for feature '{}' must be strictly sorted in ascending order.",
                feature_name
            )));
        }

        if let Some(&configured) = self.granularities.get(&feature_name) {
            if knots.len() != configured.count() {
                return Err(ModelError::LimitsMismatch(
                    knots.len(),
                    feature_name,
                    configured.count(),
                ));
            }
        } else {
            let kind = self.kind.ok_or_else(|| {
                ModelError::InvalidStrategy(
                    "Membership function kind (MFKind) must be set via `set_kind` before adding limits.".into(),
                )
            })?;
            let inferred = kind.infer_granularity(knots.len())?;
            self.granularities.insert(feature_name.clone(), inferred);
        }

        self.limits.insert(feature_name, knots);
        self.mark_unbuilt();
        Ok(self)
    }

    // =========================================================================
    // Construction and Input Space Computation
    // =========================================================================

    /// Validates and constructs input partitions and membership functions.
    pub fn build_inputs(&mut self) -> Result<(), ModelError> {
        if self.mf_configs_built {
            return Ok(());
        }

        let kind = self.kind.ok_or_else(|| {
            ModelError::InvalidStrategy(
                "Membership function kind (MFKind) was not specified.".into(),
            )
        })?;

        for (feature, strategy) in &self.strategies {
            if !self.granularities.contains_key(feature)
                && let Some(inferred) = strategy.try_infer_granularity(kind)?
            {
                self.granularities.insert(feature.clone(), inferred);
            }
        }

        self.compute_labels()?;

        for (feature, strategy) in &self.strategies {
            let granularity = self.granularities.get(feature).copied().ok_or_else(|| {
                ModelError::InvalidStrategy(format!(
                    "Feature '{}' has strategy configured but lacks granularity.",
                    feature
                ))
            })?;

            let partition =
                MembershipFunction::create_partition(kind, strategy.clone(), granularity);
            let labels = &self.labels[feature];

            let mut feature_mfs = HashMap::with_capacity(partition.len());
            let mut feature_centers = HashMap::with_capacity(partition.len());
            let mut feature_limits = Vec::with_capacity(partition.len());

            for (label, mf) in labels.iter().zip(partition) {
                let center = mf.center();
                feature_limits.push(center);
                feature_centers.insert(label.clone(), center);
                feature_mfs.insert(label.clone(), mf);
            }

            self.limits.insert(feature.clone(), feature_limits);
            self.mf_configs.insert(feature.clone(), feature_mfs);
            self.mf_centers.insert(feature.clone(), feature_centers);
        }

        self.mf_configs_built = true;
        Ok(())
    }

    fn compute_labels(&mut self) -> Result<(), ModelError> {
        for (feature, &granularity) in &self.granularities {
            let expected_count = granularity.count();

            if let Some(custom_labels) = self.labels.get(feature) {
                if custom_labels.len() != expected_count {
                    return Err(ModelError::InvalidStrategy(format!(
                        "Feature '{}' expects {} labels for granularity {:?}, but {} custom labels were provided: {:?}",
                        feature,
                        expected_count,
                        granularity,
                        custom_labels.len(),
                        custom_labels
                    )));
                }
            } else {
                self.labels
                    .insert(feature.clone(), granularity.default_labels()?);
            }
        }
        Ok(())
    }

    /// Computes membership degrees ($\mu$) for a crisp sample input row.
    pub fn get_row_memberships(
        &self,
        row: &HashMap<String, f64>,
    ) -> HashMap<String, HashMap<String, f64>> {
        let mut memberships = HashMap::with_capacity(self.mf_configs.len());

        for (feature, fuzzy_sets) in &self.mf_configs {
            if let Some(&value) = row.get(feature) {
                let mut group_mus = HashMap::with_capacity(fuzzy_sets.len());
                for (group_name, mf) in fuzzy_sets {
                    let mu = mf.eval(value);
                    group_mus.insert(group_name.clone(), mu);
                }
                memberships.insert(feature.clone(), group_mus);
            }
        }
        memberships
    }
}
