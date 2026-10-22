use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fs::File,
    io::{BufReader, BufWriter},
    path::Path,
};

use neblina::{
    dataset::{TabularDataset, TargetVector},
    error::ModelError,
    fuzzy::{
        ActivatedRule, Antecedents, FuzzyRule, TNorm, antecedent::calculate_firing_strength,
        granularity::Granularity,
    },
    membership::mb_function::{FuzzyStrategy, MembershipFunction, MembershipOp, Partitionable},
    metrics::EvaluationMetrics,
    model::{
        FuzzyBaseModel, FuzzyModel, builder::HasFuzzyBase, evaluation::FuzzyEvaluationExt as _,
    },
    types::FuzzyValues,
};

use neblina::model::config::ModelConfig;

/// Configures and executes fuzzy inference using the Wang-Mendel method.
///
/// # Example
///
/// ```rust
/// use neblina::prelude::{MFKind, TNorm};
/// use std::collections::HashMap;
/// use wm_fuzzy::wm::WMModel;
///
/// let mut model = WMModel::new();
/// model.set_kind("triangular").unwrap();
/// model.add_custom_strategy("temp", vec![0.0, 50.0, 100.0]).unwrap();
/// model.build().unwrap();
///
/// let mut input = HashMap::new();
/// input.insert("temp".to_string(), 25.0);
///
/// let (y_pred, active_rules) = model.infer(&input, TNorm::Minimum);
/// assert!(y_pred.is_none()); // No rules generated yet
/// ```
#[derive(Debug, Default)]
pub struct WMModel {
    pub base: FuzzyBaseModel,

    // Configuration and mappings for the target attribute (Wang-Mendel Specific)
    target_strategy: Option<FuzzyStrategy>,
    target_granularity: Option<Granularity>,
    target_labels: Vec<String>,
    pub(super) target_mfs: HashMap<String, MembershipFunction>,
    target_centers: HashMap<String, f64>,

    // Extracted Rules and State
    rules: Vec<FuzzyRule>,
    mf_values: Vec<HashMap<String, FuzzyValues>>,
    norm: String,
}

impl FuzzyModel for WMModel {
    type Output = (Vec<f64>, Vec<u32>);

    fn base_model(&self) -> &FuzzyBaseModel {
        &self.base
    }

    fn base_model_mut(&mut self) -> &mut FuzzyBaseModel {
        &mut self.base
    }

    fn predict<D: TabularDataset>(
        &mut self,
        x_test: &D,
        threshold: Option<f64>,
        fallback: Option<f64>,
    ) -> Result<Self::Output, ModelError> {
        let fallback_val = fallback.unwrap_or(0.5);

        let t_norm = match self.norm.as_str() {
            "minimum" => TNorm::Minimum,
            _ => TNorm::Product,
        };

        self.predict_batch(x_test, threshold, fallback_val, t_norm)
    }
}

/// Build methods to setup the target
impl WMModel {
    /// Set strategy for the target value directly
    pub fn set_target_strategy(&mut self, strategy: FuzzyStrategy) -> &mut Self {
        self.target_strategy = Some(strategy);
        self.base.mark_unbuilt();
        self
    }
    /// Add custom strategy for the target value
    pub fn add_target_custom_strategy<T: Into<Vec<f64>>>(
        &mut self,
        points: T,
    ) -> Result<&mut Self, ModelError> {
        self.target_strategy = Some(FuzzyStrategy::custom(points)?);
        self.base.mark_unbuilt();
        Ok(self)
    }

    /// Add linear strategy for the target value
    pub fn add_target_linear_strategy(
        &mut self,
        min: f64,
        max: f64,
    ) -> Result<&mut Self, ModelError> {
        self.target_strategy = Some(FuzzyStrategy::linear(min, max)?);
        self.base.mark_unbuilt();
        Ok(self)
    }

    /// Add the granularity for the target value
    pub fn add_target_granularity(
        &mut self,
        granularity: impl TryInto<Granularity, Error = ModelError>,
    ) -> Result<&mut Self, ModelError> {
        self.target_granularity = Some(granularity.try_into()?);
        self.base.mark_unbuilt();
        Ok(self)
    }

    /// Set target granularity directly
    pub fn set_target_granularity(&mut self, granularity: Granularity) -> &mut Self {
        self.target_granularity = Some(granularity);
        self.base.mark_unbuilt();
        self
    }

    /// Set custom labels for the fuzzy groups in the target value
    pub fn set_target_labels<S: Into<String>>(&mut self, labels: Vec<S>) -> &mut Self {
        self.target_labels = labels.into_iter().map(Into::into).collect();
        self.base.mark_unbuilt();
        self
    }
}

// Implements the set's methods
impl HasFuzzyBase for WMModel {
    fn base_mut(&mut self) -> &mut FuzzyBaseModel {
        &mut self.base
    }
}

/// Builder global methods
impl WMModel {
    /// Creates a new `WMModel` instance with default parameters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builds and validates the model in-place, expanding partition strategies into grid knots and generating MFs.
    pub fn build(&mut self) -> Result<&mut Self, ModelError> {
        self.base.build_inputs()?;

        let kind = self.base.kind.ok_or_else(|| {
            ModelError::InvalidStrategy(
                "Membership function kind (MFKind) was not specified.".into(),
            )
        })?;

        // Setup the MF values for the target
        if let Some(target_strat) = &self.target_strategy {
            let target_gran = self.target_granularity.ok_or_else(|| {
                ModelError::InvalidStrategy(
                    "Target Y strategy configured but lacks granularity.".into(),
                )
            })?;

            let target_partition =
                MembershipFunction::create_partition(kind, target_strat.clone(), target_gran);

            if self.target_labels.is_empty() {
                self.target_labels = target_gran.default_labels()?;
            }

            if self.target_labels.len() != target_gran.count() {
                return Err(ModelError::InvalidStrategy(format!(
                    "Target variable 'y' expects {} labels for granularity {:?}, but {} custom labels were provided: {:?}",
                    target_gran.count(),
                    target_gran,
                    self.target_labels.len(),
                    self.target_labels
                )));
            }

            self.target_mfs.clear();
            self.target_centers.clear();

            for (label, mf) in self.target_labels.iter().zip(target_partition) {
                let center = mf.center();
                self.target_centers.insert(label.clone(), center);
                self.target_mfs.insert(label.clone(), mf);
            }
        }

        Ok(self)
    }
}

// Prediction and inference
impl WMModel {
    /// Performs batch prediction over a tabular dataset.
    pub fn predict_batch<D: TabularDataset>(
        &mut self,
        x_test: &D,
        threshold: Option<f64>,
        fallback: f64,
        t_norm: TNorm,
    ) -> Result<(Vec<f64>, Vec<u32>), ModelError> {
        self.build()?;

        let feature_names: Vec<String> = self.base.mf_configs.keys().cloned().collect();

        let feature_refs: Vec<&str> = feature_names.iter().map(|s| s.as_str()).collect();
        let height = x_test.num_rows();
        let mut continuous_preds = Vec::with_capacity(height);
        let mut binary_preds = Vec::with_capacity(height);
        let threshold = threshold.unwrap_or(0.5);

        for row_idx in 0..height {
            let row_buffer = x_test.get_row_as_map(row_idx, &feature_refs);
            let memberships = self.base.get_row_memberships(&row_buffer);

            let mut total_firing_strength = 0.0;
            let mut weighted_output_sum = 0.0;

            for rule in &self.rules {
                let firing_strength =
                    calculate_firing_strength(&rule.antecedents, &memberships, t_norm.clone());

                if firing_strength > 0.0 {
                    weighted_output_sum += firing_strength * rule.weighted_average;
                    total_firing_strength += firing_strength;
                }
            }

            let y_pred = if total_firing_strength > 0.0 {
                weighted_output_sum / total_firing_strength
            } else if let (Some(nearest), _) = self.min_distance_rule(&row_buffer) {
                nearest.weighted_average
            } else {
                fallback
            };

            continuous_preds.push(y_pred);
            binary_preds.push(if y_pred >= threshold { 1 } else { 0 });
        }

        Ok((continuous_preds, binary_preds))
    }

    /// Finds the nearest rule in Euclidean input space when no rules fire for an input sample.
    pub fn min_distance_rule(
        &self,
        crisp_row: &HashMap<String, f64>,
    ) -> (Option<&FuzzyRule>, Option<f64>) {
        if self.rules.is_empty() {
            return (None, None);
        }

        let mut nearest_rule: Option<&FuzzyRule> = None;
        let mut min_dist = f64::INFINITY;

        for rule in &self.rules {
            if let Some(input_centers) = rule.antecedents.get_input_center(&self.base.mf_centers) {
                let mut sum_sq = 0.0;
                let mut matched_all = true;

                for (feature, value) in crisp_row {
                    if let Some(&c1) = input_centers.get(feature) {
                        sum_sq += (c1 - value).powi(2);
                    } else {
                        matched_all = false;
                        break;
                    }
                }

                if matched_all {
                    let dist = sum_sq.sqrt();
                    if dist < min_dist {
                        min_dist = dist;
                        nearest_rule = Some(rule);
                    }
                }
            }
        }

        (
            nearest_rule,
            if min_dist.is_infinite() {
                None
            } else {
                Some(min_dist)
            },
        )
    }

    /// Performs single-sample inference and returns activated rules sorted by firing strength.
    pub fn infer(
        &self,
        crisp_row: &HashMap<String, f64>,
        t_norm: TNorm,
    ) -> (Option<f64>, Vec<ActivatedRule<'_>>) {
        let memberships = self.base.get_row_memberships(crisp_row);
        let mut activated = Vec::new();
        let mut total_firing_strength = 0.0;
        let mut weighted_output_sum = 0.0;

        for rule in &self.rules {
            let firing_strength =
                calculate_firing_strength(&rule.antecedents, &memberships, t_norm.clone());

            if firing_strength > 0.0 {
                weighted_output_sum += firing_strength * rule.weighted_average;
                total_firing_strength += firing_strength;

                activated.push(ActivatedRule {
                    rule,
                    firing_strength,
                });
            }
        }

        activated.sort_by(|a, b| {
            b.firing_strength
                .partial_cmp(&a.firing_strength)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut y_pred = None;

        if activated.is_empty() {
            println!("\n⚠️ No rules were activated for the given row:");
            println!("  {:?}", crisp_row);
        } else {
            println!("\n🔥 Activated Rules (Total: {}):", activated.len());
            for rule_act in &activated {
                println!("{}", rule_act.rule);
            }

            if total_firing_strength > 0.0 {
                let val = weighted_output_sum / total_firing_strength;
                y_pred = Some(val);
                println!("\n📊 [Defuzzification / Weighted Mean]:");
                println!("  ↳ y inferred: {:.4}", val);
                println!("  ↳ Combined firing strength: {:.4}", total_firing_strength);
            }
        }

        (y_pred, activated)
    }

    /// Resets the extracted rules and flags the model for rebuilding.
    pub fn clear_rules(&mut self) {
        self.rules.clear();
        self.base.mark_unbuilt();
    }

    /// Returns the total number of extracted fuzzy rules.
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    /// Returns whether internal membership function structures have been built.
    pub fn is_configured(&self) -> bool {
        self.base.is_configured()
    }
}

// Rules generation
impl WMModel {
    fn extract_antecedents(
        &self,
        row_memberships: &[HashMap<String, HashMap<String, f64>>],
        feature_names: &[&str],
    ) -> Vec<Antecedents> {
        let mut seen = HashSet::new();
        let mut candidate_antecedents = Vec::new();

        for row_mems in row_memberships {
            let mut antecedent_map = BTreeMap::new();

            for &feature in feature_names {
                if let Some(fuzzy_sets) = row_mems.get(feature)
                    && let Some((best_label, _)) = fuzzy_sets
                        .iter()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                {
                    antecedent_map.insert(feature.to_string(), best_label.clone());
                }
            }

            if antecedent_map.len() == feature_names.len() {
                let antecedents = Antecedents {
                    data: antecedent_map,
                };
                if seen.insert(antecedents.clone()) {
                    candidate_antecedents.push(antecedents);
                }
            }
        }

        candidate_antecedents
    }

    pub fn generate_rules<D: TabularDataset, Y: TargetVector>(
        &mut self,
        x_train: &D,
        y_train: &Y,
        feature_names: Option<&[&str]>,
        t_norm: TNorm,
        min_support: f64,
        min_confidence: f64,
    ) -> Result<Vec<FuzzyRule>, ModelError> {
        self.build()?;

        match t_norm {
            TNorm::Product => self.norm = String::from("product"),
            _ => self.norm = String::from("minimum"),
        }

        let n_samples = x_train.num_rows();
        if n_samples == 0 || y_train.num_samples() == 0 {
            return Ok(Vec::new());
        }

        if n_samples != y_train.num_samples() {
            return Err(ModelError::DatasetError(format!(
                "X and y length mismatch: {} vs {}",
                n_samples,
                y_train.num_samples()
            )));
        }

        let default_names: Vec<String>;
        let feature_names: Vec<&str> = match feature_names {
            Some(names) => names.to_vec(),
            None => {
                default_names = self.base.mf_configs.keys().cloned().collect();
                default_names.iter().map(|s| s.as_str()).collect()
            }
        };

        let y_data = y_train.to_vec_f64()?;
        let (y_min, y_max) = y_data.iter().fold(
            (f64::INFINITY, f64::NEG_INFINITY),
            |(min_v, max_v), &val| (min_v.min(val), max_v.max(val)),
        );
        let variance_range = (y_max - y_min).max(f64::EPSILON);

        let mut row_memberships = Vec::with_capacity(n_samples);
        for row_idx in 0..n_samples {
            let row_map = x_train.get_row_as_map(row_idx, &feature_names);
            row_memberships.push(self.base.get_row_memberships(&row_map));
        }

        let candidate_antecedents = self.extract_antecedents(&row_memberships, &feature_names);
        let mut generated_rules = Vec::new();

        for antecedents in candidate_antecedents {
            let rule_strength: Vec<f64> = row_memberships
                .iter()
                .map(|mems| calculate_firing_strength(&antecedents, mems, t_norm.clone()))
                .collect();

            let sum_strength: f64 = rule_strength.iter().sum();
            if sum_strength <= 0.0 {
                continue;
            }

            let support = sum_strength / (n_samples as f64);
            if support < min_support {
                continue;
            }

            let weighted_sum: f64 = y_data
                .iter()
                .zip(&rule_strength)
                .map(|(&yi, &wi)| yi * wi)
                .sum();
            let weighted_avg = weighted_sum / sum_strength;

            let weighted_var_sum: f64 = y_data
                .iter()
                .zip(&rule_strength)
                .map(|(&yi, &wi)| (yi - weighted_avg).abs() * wi)
                .sum();
            let weighted_variance = weighted_var_sum / sum_strength;

            let base_doc = (1.0 - (weighted_variance / variance_range)).clamp(0.0, 1.0);

            if !self.target_mfs.is_empty() {
                let mut best_y_label = String::new();
                let mut max_y_mu = 0.0;

                for (y_label, mf) in &self.target_mfs {
                    let mu = mf.eval(weighted_avg);
                    if mu > max_y_mu {
                        max_y_mu = mu;
                        best_y_label = y_label.clone();
                    }
                }

                if max_y_mu <= 0.0 {
                    continue;
                }

                let doc = base_doc * max_y_mu;
                if doc < min_confidence {
                    continue;
                }

                let target_center = *self.target_centers.get(&best_y_label).unwrap_or(&0.0);

                generated_rules.push(FuzzyRule::linguistic(
                    antecedents,
                    best_y_label,
                    target_center,
                    doc,
                    support,
                ));
            } else {
                let doc = base_doc;
                if doc < min_confidence {
                    continue;
                }

                generated_rules.push(FuzzyRule::numeric(
                    antecedents,
                    weighted_avg,
                    weighted_variance,
                    doc,
                    support,
                ));
            }
        }

        self.mf_values = row_memberships;
        self.rules = generated_rules.clone();
        Ok(generated_rules)
    }
}

// Persistence and evaluation
impl WMModel {
    /// Smart evaluation entry point.
    pub fn evaluate<D: TabularDataset, Y: TargetVector>(
        &mut self,
        x_test: &D,
        y_test: &Y,
        threshold: Option<f64>,
        fallback: Option<f64>,
        output_path: Option<&Path>,
    ) -> Result<EvaluationMetrics, ModelError> {
        if let Some(th) = threshold {
            let metrics =
                self.evaluate_classification(x_test, y_test, Some(th), fallback, output_path)?;
            Ok(EvaluationMetrics::Classification(metrics))
        } else {
            let metrics = self.evaluate_regression(x_test, y_test, fallback, output_path)?;
            Ok(EvaluationMetrics::Regression(metrics))
        }
    }

    /// Serializes and saves trained rules to a JSON file.
    pub fn save_rules<P: AsRef<Path>>(&self, path: P) -> Result<(), std::io::Error> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, &self.rules).map_err(std::io::Error::other)?;
        Ok(())
    }

    /// Loads trained rules from a JSON file and resets MF configuration state.
    pub fn load_rules<P: AsRef<Path>>(&mut self, path: P) -> Result<(), std::io::Error> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let rules: Vec<FuzzyRule> = serde_json::from_reader(reader)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        self.rules = rules;
        self.base.mark_unbuilt();
        Ok(())
    }
}

// Getters and setters
impl WMModel {
    pub fn rules(&self) -> Result<&[FuzzyRule], ModelError> {
        if self.rules.is_empty() {
            Err(ModelError::NoRulesLoaded)
        } else {
            Ok(&self.rules)
        }
    }

    pub fn mf_values(&self) -> Result<&[HashMap<String, FuzzyValues>], ModelError> {
        if self.mf_values.is_empty() {
            Err(ModelError::NoMFValues)
        } else {
            Ok(&self.mf_values)
        }
    }

    pub fn target_mfs(&self) -> Result<&HashMap<String, MembershipFunction>, ModelError> {
        if self.target_mfs.is_empty() {
            Err(ModelError::TargetMFUnset)
        } else {
            Ok(&self.target_mfs)
        }
    }

    pub fn mf_configs(
        &self,
    ) -> Result<&HashMap<String, HashMap<String, MembershipFunction>>, ModelError> {
        if self.base.mf_configs.is_empty() {
            Err(ModelError::UnconfiguredError)
        } else {
            Ok(&self.base.mf_configs)
        }
    }

    /// Returns a reference to the configured feature limits map if available.
    pub fn limits(&self) -> Result<&HashMap<String, Vec<f64>>, ModelError> {
        if self.base.limits.is_empty() {
            Err(ModelError::InvalidStrategy(
                "No feature limits have been configured.".into(),
            ))
        } else {
            Ok(&self.base.limits)
        }
    }
}

impl WMModel {
    /// Instantiates an unbuilt `WMModel` initialized with settings from a `ModelConfig`.
    pub fn from_config(config: ModelConfig) -> Result<Self, ModelError> {
        let mut model = WMModel::new();
        model.base.kind = Some(config.kind);

        let mut custom_labels = HashMap::new();

        // 1. Configure input features (X)
        for (feature, feat_cfg) in config.features {
            model
                .base
                .strategies
                .insert(feature.clone(), feat_cfg.strategy);

            if let Some(granularity) = feat_cfg.granularity {
                model
                    .base
                    .granularities
                    .insert(feature.clone(), granularity);
            }

            if !feat_cfg.labels.is_empty() {
                custom_labels.insert(feature, feat_cfg.labels);
            }
        }

        if !custom_labels.is_empty() {
            model.base.set_labels(custom_labels);
        }

        // 2. Configure target variable (Y) if provided
        if let Some(target_cfg) = config.target {
            model.set_target_strategy(target_cfg.strategy);

            if let Some(granularity) = target_cfg.granularity {
                model.set_target_granularity(granularity);
            }

            if !target_cfg.labels.is_empty() {
                model.set_target_labels(target_cfg.labels);
            }
        }

        Ok(model)
    }

    /// Instantiates and builds a `WMModel` directly from a JSON configuration string.
    pub fn from_json(json_str: &str) -> Result<Self, ModelError> {
        let config = ModelConfig::from_json(json_str)?;
        let mut model = Self::from_config(config)?;
        model.build()?;
        Ok(model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use neblina::membership::mb_function::MembershipOp;
    use neblina::model::builder::FuzzyBuilder;
    use neblina::types::HashDataset;
    use std::collections::HashMap;

    fn create_test_dataset(x1_values: &[f64]) -> HashDataset {
        x1_values
            .iter()
            .map(|&val| {
                let mut row = HashMap::new();
                row.insert("x1".to_string(), val);
                row
            })
            .collect()
    }

    #[test]
    fn test_builder_missing_kind_error() {
        let mut model = WMModel::new();
        model.add_granularity("x1", 3).unwrap();
        model.add_limit("x1", vec![0.0, 5.0, 10.0]).unwrap();

        let result = model.build();
        assert!(matches!(result, Err(ModelError::InvalidStrategy(_))));
    }

    #[test]
    fn test_builder_limits_granularity_mismatch() {
        let mut model = WMModel::new();
        model.set_kind("triangular").unwrap();
        model.add_granularity("x1", 3).unwrap();
        let result = model.add_limit("x1", vec![0.0, 10.0]);

        assert!(matches!(result, Err(ModelError::LimitsMismatch(2, _, 3))));
    }

    #[test]
    fn test_successful_build_with_linear_strategy() {
        let mut model = WMModel::new();
        model
            .set_kind("triangular")
            .unwrap()
            .add_granularity("x1", 3)
            .unwrap()
            .add_linear_strategy("x1", 0.0, 10.0)
            .unwrap();

        assert!(model.build().is_ok());
        assert!(model.is_configured());
        assert_eq!(
            model.limits().unwrap().get("x1").unwrap(),
            &[0.0, 5.0, 10.0]
        );
    }

    #[test]
    fn test_target_configuration_builder_methods() {
        let mut model = WMModel::new();
        model
            .set_kind("triangular")
            .unwrap()
            .add_granularity("x1", 3)
            .unwrap()
            .add_linear_strategy("x1", 0.0, 10.0)
            .unwrap()
            .add_target_granularity(3)
            .unwrap()
            .add_target_linear_strategy(0.0, 100.0)
            .unwrap()
            .set_target_labels(vec![
                "low".to_string(),
                "medium".to_string(),
                "high".to_string(),
            ]);

        assert!(model.build().is_ok());
        assert_eq!(model.target_mfs.len(), 3);
        assert!(model.target_mfs.contains_key("low"));
        assert!(model.target_mfs.contains_key("medium"));
        assert!(model.target_mfs.contains_key("high"));

        let medium_center = model.target_mfs.get("medium").unwrap().center();
        assert!((medium_center - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_generate_rules_case1_linguistic_target() {
        let mut model = WMModel::new();
        model
            .set_kind("triangular")
            .unwrap()
            .add_granularity("x1", 3)
            .unwrap()
            .add_linear_strategy("x1", 0.0, 10.0)
            .unwrap()
            .add_target_granularity(3)
            .unwrap()
            .add_target_linear_strategy(0.0, 100.0)
            .unwrap();

        let x_train: HashDataset = create_test_dataset(&[0.0, 5.0, 10.0]);
        let y_train: Vec<f64> = vec![0.0, 50.0, 100.0];

        let rules = model
            .generate_rules(&x_train, &y_train, None, TNorm::Product, 0.0, 0.0)
            .unwrap();

        assert!(!rules.is_empty());
        assert_eq!(model.rule_count(), rules.len());
        assert!(!model.target_mfs.is_empty());
    }

    #[test]
    fn test_generate_rules_case2_numeric_target() {
        let mut model = WMModel::new();
        model
            .set_kind("triangular")
            .unwrap()
            .add_granularity("x1", 3)
            .unwrap()
            .add_linear_strategy("x1", 0.0, 10.0)
            .unwrap();

        let x_train: HashDataset = create_test_dataset(&[0.0, 5.0, 10.0]);
        let y_train: Vec<f64> = vec![1.2, 5.4, 9.8];

        let rules = model
            .generate_rules(&x_train, &y_train, None, TNorm::Product, 0.0, 0.0)
            .unwrap();

        assert!(!rules.is_empty());
        assert!(model.target_mfs.is_empty());
    }

    #[test]
    fn test_predict_and_infer() {
        let mut model = WMModel::new();
        model
            .set_kind("triangular")
            .unwrap()
            .add_granularity("x1", 3)
            .unwrap()
            .add_linear_strategy("x1", 0.0, 10.0)
            .unwrap();

        let x_train: HashDataset = create_test_dataset(&[0.0, 5.0, 10.0]);
        let y_train: Vec<f64> = vec![0.0, 0.5, 1.0];

        model
            .generate_rules(&x_train, &y_train, None, TNorm::Product, 0.0, 0.0)
            .unwrap();

        let (continuous_preds, binary_preds) =
            model.predict(&x_train, Some(0.5), Some(0.0)).unwrap();

        assert_eq!(continuous_preds.len(), 3);
        assert_eq!(binary_preds.len(), 3);

        let mut sample_row = HashMap::new();
        sample_row.insert("x1".to_string(), 5.0);
        let (y_pred, activated) = model.infer(&sample_row, TNorm::Product);

        assert!(y_pred.is_some());
        assert!(!activated.is_empty());
    }

    #[test]
    fn test_evaluate_regression_and_classification() {
        let mut model = WMModel::new();
        model
            .set_kind("triangular")
            .unwrap()
            .add_granularity("x1", 3)
            .unwrap()
            .add_linear_strategy("x1", 0.0, 10.0)
            .unwrap();

        let x_test: HashDataset = create_test_dataset(&[0.0, 5.0, 10.0]);
        let y_test: Vec<f64> = vec![0.0, 0.5, 1.0];

        model
            .generate_rules(&x_test, &y_test, None, TNorm::Product, 0.0, 0.0)
            .unwrap();

        let reg_eval = model.evaluate(&x_test, &y_test, None, None, None).unwrap();
        if let EvaluationMetrics::Regression(metrics) = reg_eval {
            assert!(metrics.mse >= 0.0);
            assert!(metrics.mae >= 0.0);
        } else {
            panic!("Expected Regression metrics variant");
        }

        let clf_eval = model
            .evaluate(&x_test, &y_test, Some(0.5), None, None)
            .unwrap();
        if let EvaluationMetrics::Classification(metrics) = clf_eval {
            assert!(metrics.accuracy >= 0.0 && metrics.accuracy <= 1.0);
        } else {
            panic!("Expected Classification metrics variant");
        }
    }
}
