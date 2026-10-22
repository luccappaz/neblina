use std::fs::File;
use std::path::Path;
use std::str::FromStr;

use burn::backend::Backend;
use burn::module::Param;
use burn::optim::{AdamConfig, GradientsParams};
use burn::prelude::*;

use neblina::membership::mb_function::MembershipOp as _;
use neblina::membership::{FuzzyStrategy, MFKind};
use neblina::model::builder::HasFuzzyBase;
use neblina::model::config::ModelConfig;
use neblina::model::evaluation::FuzzyEvaluationExt as _;
use neblina::model::{FuzzyModel, TrainableModel};
use neblina::{
    dataset::{TabularDataset, TargetVector},
    error::ModelError,
    metrics::EvaluationMetrics,
    model::FuzzyBaseModel,
};

/// Represents an extracted Takagi-Sugeno fuzzy rule with premises, parameters, and formatted text.
#[derive(Debug, Clone, serde::Serialize)]
pub struct AnfisRule {
    pub rule_index: usize,
    /// Premise antecedents per feature: (Feature Name, Center, Sigma)
    pub premises: Vec<(String, f64, f64)>,
    /// Linear consequent coefficients: [p_1, p_2, ..., p_n, bias]
    pub consequent_coeffs: Vec<f64>,
    /// Formatted rule representation
    pub text_representation: String,
}

/// Low-level Burn neural network module representing the 5 layers of ANFIS (Takagi-Sugeno).
#[derive(Module, Debug)]
pub struct AnfisModule {
    /// Premise parameters - 2D Tensor [Num_Rules * Num_Inputs, 2]
    pub premise_params: Param<Tensor<2>>,

    /// Consequent linear parameters for rule outputs - 2D Tensor [Num_Rules, Num_Inputs + 1]
    pub consequent_params: Param<Tensor<2>>,
}

impl AnfisModule {
    pub fn new(num_rules: usize, num_inputs: usize, device: &Device) -> Self {
        let premise_params = Tensor::<2>::random(
            [num_rules * num_inputs, 2],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            device,
        )
        .autodiff();

        let consequent_params = Tensor::<2>::zeros([num_rules, num_inputs + 1], device).autodiff();

        Self {
            premise_params: Param::from_tensor(premise_params),
            consequent_params: Param::from_tensor(consequent_params),
        }
    }

    /// Forward pass through the 5 layers of Takagi-Sugeno ANFIS (Gaussian MFs)
    pub fn forward(&self, inputs: Tensor<2>) -> Tensor<2> {
        let [batch_size, num_inputs] = inputs.dims();
        let [total_param_rows, _] = self.premise_params.val().dims();
        let num_rules = total_param_rows / num_inputs.max(1);

        if num_rules == 0 || num_inputs == 0 {
            return Tensor::<2>::zeros([batch_size, 1], &inputs.device());
        }

        let device = inputs.device();
        let premise = self.premise_params.val();
        let centers = premise
            .clone()
            .slice([0..total_param_rows, 0..1])
            .reshape([num_rules, num_inputs]);
        let sigmas = premise
            .slice([0..total_param_rows, 1..2])
            .reshape([num_rules, num_inputs])
            .abs()
            + 1e-5;

        // Layer 1 & 2: Fuzzification & Rule Firing Strength (Product T-Norm with Gaussian MFs)
        let mut firing_strengths = Tensor::<2>::ones([batch_size, num_rules], &device);
        for i in 0..num_inputs {
            let x_col = inputs.clone().slice([0..batch_size, i..i + 1]);
            let c_col = centers.clone().slice([0..num_rules, i..i + 1]).transpose();
            let s_col = sigmas.clone().slice([0..num_rules, i..i + 1]).transpose();

            let diff = (x_col - c_col).div(s_col).powf_scalar(2.0);
            let mu = (-diff).exp();
            let mu_i = mu.slice([0..batch_size, 0..num_rules]);
            firing_strengths = firing_strengths * mu_i;
        }

        // Layer 3: Normalized Firing Strength
        let fs_sum = firing_strengths
            .clone()
            .sum_dim(1)
            .clamp(1e-8, f32::MAX as f64);
        let normalized_fs = firing_strengths.div(fs_sum);

        // Layer 4: Consequent Layer (Linear rule outputs)
        let ones = Tensor::<2>::ones([batch_size, 1], &device);
        let augmented_inputs = Tensor::cat(vec![inputs, ones], 1);

        let rule_outputs = augmented_inputs.matmul(self.consequent_params.val().transpose());

        // Layer 5: Overall Output (Weighted sum)
        (normalized_fs * rule_outputs).sum_dim(1)
    }
}

/// Adaptive Neuro-Fuzzy Inference System (ANFIS) model combining FuzzyBaseModel and Burn.
#[derive(Debug)]
pub struct AnfisModel<B: Backend> {
    pub base: FuzzyBaseModel,
    pub module: AnfisModule,
    device: Device,
    epochs: usize,
    learning_rate: f64,
    _marker: std::marker::PhantomData<B>,
}

impl<B: Backend> HasFuzzyBase for AnfisModel<B> {
    fn base_mut(&mut self) -> &mut FuzzyBaseModel {
        &mut self.base
    }
}

// ============================================================================
// Model Construction & Core Methods
// ============================================================================

impl<B: Backend> AnfisModel<B> {
    pub fn new(device: Device) -> Self {
        let module = AnfisModule::new(1, 1, &device);
        Self {
            base: FuzzyBaseModel::new(),
            module,
            device,
            epochs: 100,
            learning_rate: 0.01,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn with_epochs(&mut self, epochs: usize) -> &mut Self {
        self.epochs = epochs;
        self
    }

    pub fn with_learning_rate(&mut self, lr: f64) -> &mut Self {
        self.learning_rate = lr;
        self
    }

    /// Sets the membership function kind. Rejects Triangular with an explicit error and supports Gaussian.
    pub fn set_kind(&mut self, kind_str: &str) -> Result<&mut Self, ModelError> {
        let kind = match kind_str.to_lowercase().as_str() {
            "gaussian" => MFKind::Gaussian,
            "triangular" => {
                return Err(ModelError::InvalidStrategy(
                    "ANFIS model requires differentiable membership functions (e.g., Gaussian). Triangular MFs are not supported due to non-differentiability at their peaks.".into()
                ));
            }
            other => MFKind::from_str(other)?,
        };
        self.base.kind = Some(kind);
        Ok(self)
    }

    /// Convenience method to add a linear partition strategy for a feature.
    pub fn add_linear_strategy(
        &mut self,
        feature: &str,
        min: f64,
        max: f64,
    ) -> Result<&mut Self, ModelError> {
        let strategy = FuzzyStrategy::linear(min, max)?;
        self.base.strategies.insert(feature.to_string(), strategy);
        self.base.granularities.insert(
            feature.to_string(),
            neblina::fuzzy::granularity::Granularity::try_from(3).unwrap(),
        );
        Ok(self)
    }

    pub fn build(&mut self) -> Result<&mut Self, ModelError> {
        let kind = self.base.kind.unwrap_or(MFKind::Gaussian);
        if kind == MFKind::Triangular {
            return Err(ModelError::InvalidStrategy(
                "ANFIS model requires differentiable membership functions (e.g., Gaussian). Triangular MFs are not supported.".into()
            ));
        }

        self.base.build_inputs()?;

        let num_features = self.base.mf_configs.len();
        if num_features == 0 {
            return Err(ModelError::UnconfiguredError);
        }

        let total_rules: usize = self
            .base
            .granularities
            .values()
            .map(|g| g.count())
            .product();

        self.module = AnfisModule::new(total_rules.max(1), num_features, &self.device);

        Ok(self)
    }

    pub fn predict_tensor(&self, inputs: Tensor<2>) -> Tensor<2> {
        self.module.forward(inputs)
    }

    /// Extracts rules and formats them in natural language.
    pub fn extract_rules(&self) -> Result<Vec<AnfisRule>, ModelError> {
        let feature_names: Vec<String> = self.base.granularities.keys().cloned().collect();
        let num_inputs = feature_names.len();
        if num_inputs == 0 {
            return Err(ModelError::UnconfiguredError);
        }

        let premise = self.module.premise_params.val();
        let consequent = self.module.consequent_params.val();

        let [total_rows, _] = premise.dims();
        let num_rules = total_rows / num_inputs;

        let premise_data: Vec<f32> = premise.into_data().try_to_vec().unwrap_or_default();
        let consequent_data: Vec<f32> = consequent
            .clone()
            .into_data()
            .try_to_vec()
            .unwrap_or_default();

        let mut rules = Vec::with_capacity(num_rules);

        for r in 0..num_rules {
            let mut premises = Vec::with_capacity(num_inputs);
            let mut text_if_parts = Vec::with_capacity(num_inputs);
            let mut total_sigma = 0.0;

            for (i, fname) in feature_names.iter().enumerate() {
                let param_idx = (r * num_inputs + i) * 2;
                let center = premise_data.get(param_idx).copied().unwrap_or(0.0) as f64;
                let sigma = premise_data.get(param_idx + 1).copied().unwrap_or(1.0) as f64;

                premises.push((fname.clone(), center, sigma));
                total_sigma += sigma;

                let label = if let Some(mfs) = self.base.mf_configs.get(fname) {
                    mfs.iter()
                        .min_by(|a, b| {
                            let diff_a = (a.1.center() - center).abs();
                            let diff_b = (b.1.center() - center).abs();
                            diff_a
                                .partial_cmp(&diff_b)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(lbl, _)| lbl.as_str())
                        .unwrap_or("active")
                } else {
                    "active"
                };

                text_if_parts.push(format!("{} is {}", fname, label));
            }

            let [_, num_cols] = consequent.dims();
            let mut coeffs = Vec::with_capacity(num_cols);
            for c in 0..num_cols {
                let val = consequent_data
                    .get(r * num_cols + c)
                    .copied()
                    .unwrap_or(0.0) as f64;
                coeffs.push(val);
            }

            let avg_sigma = if num_inputs > 0 {
                total_sigma / num_inputs as f64
            } else {
                0.0
            };
            let bias = coeffs.last().copied().unwrap_or(0.0);

            let text_representation = format!(
                "IF {} THEN y = {:.4} (sigma: {:.2})",
                text_if_parts.join(" AND "),
                bias,
                avg_sigma
            );

            rules.push(AnfisRule {
                rule_index: r,
                premises,
                consequent_coeffs: coeffs,
                text_representation,
            });
        }

        Ok(rules)
    }

    /// Saves the extracted rules to a JSON file on disk.
    pub fn save_rules_json(&self, path: &Path) -> Result<(), ModelError> {
        let rules = self.extract_rules()?;
        let file = File::create(path).map_err(|e| ModelError::DatasetError(e.to_string()))?;
        serde_json::to_writer_pretty(file, &rules)
            .map_err(|e| ModelError::DatasetError(e.to_string()))?;
        Ok(())
    }
}

// ============================================================================
// Trait Implementations (`FuzzyModel` & `TrainableModel`)
// ============================================================================

impl<B: Backend> FuzzyModel for AnfisModel<B> {
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
        _fallback: Option<f64>,
    ) -> Result<Self::Output, ModelError> {
        self.build()?;

        let height = x_test.num_rows();
        if height == 0 {
            return Ok((vec![], vec![]));
        }

        let feature_names: Vec<String> = self.base.mf_configs.keys().cloned().collect();
        let mut data_matrix = Vec::with_capacity(height * feature_names.len().max(1));

        for row_idx in 0..height {
            let row_map = x_test.get_row_as_map(row_idx, &[]);
            for fname in &feature_names {
                let val = row_map.get(fname).copied().unwrap_or(0.0);
                data_matrix.push(val as f32);
            }
        }

        let num_cols = feature_names.len().max(1);
        let input_tensor = Tensor::<2>::from_data(
            burn::tensor::TensorData::new(data_matrix, vec![height, num_cols]),
            &self.device,
        );

        let pred_tensor = self.module.forward(input_tensor);
        let preds_data = pred_tensor.into_data();
        let continuous_preds: Vec<f64> = preds_data
            .try_to_vec()
            .unwrap_or_default()
            .into_iter()
            .map(|v: f32| v as f64)
            .collect();

        let threshold_val = threshold.unwrap_or(0.5);

        let binary_preds = continuous_preds
            .iter()
            .map(|&val| if val >= threshold_val { 1 } else { 0 })
            .collect();

        Ok((continuous_preds, binary_preds))
    }
}

impl<B: Backend> TrainableModel for AnfisModel<B> {
    type Config = ModelConfig;

    fn train(
        &mut self,
        x_train: &dyn TabularDataset,
        y_train: &dyn TargetVector,
        _config: &Self::Config,
    ) -> Result<(), ModelError> {
        self.build()?;

        let n_samples = x_train.num_rows();
        if n_samples == 0 || y_train.num_samples() == 0 {
            return Ok(());
        }

        if n_samples != y_train.num_samples() {
            return Err(ModelError::DatasetError(format!(
                "X and y length mismatch: {} vs {}",
                n_samples,
                y_train.num_samples()
            )));
        }

        let feature_names: Vec<String> = self.base.mf_configs.keys().cloned().collect();
        let num_cols = feature_names.len().max(1);

        let mut x_data = Vec::with_capacity(n_samples * num_cols);
        for row_idx in 0..n_samples {
            let row_map = x_train.get_row_as_map(row_idx, &[]);
            for fname in &feature_names {
                let val = row_map.get(fname).copied().unwrap_or(0.0);
                x_data.push(val as f32);
            }
        }

        let y_vec = y_train.to_vec_f64()?;
        let y_data: Vec<f32> = y_vec.into_iter().map(|v| v as f32).collect();

        let x_tensor = Tensor::<2>::from_data(
            burn::tensor::TensorData::new(x_data, vec![n_samples, num_cols]),
            &self.device,
        );
        let y_tensor = Tensor::<2>::from_data(
            burn::tensor::TensorData::new(y_data, vec![n_samples, 1]),
            &self.device,
        );

        let mut optimizer = AdamConfig::new().init();

        for _epoch in 0..self.epochs {
            let preds = self.module.forward(x_tensor.clone());
            let loss = (preds - y_tensor.clone()).powf_scalar(2.0).mean();
            let grads = loss.backward();
            let grads_params = GradientsParams::from_grads(grads, &self.module);
            self.module = optimizer.step(self.learning_rate, self.module.clone(), grads_params);
        }

        Ok(())
    }
}

// ============================================================================
// Evaluation & Persistence
// ============================================================================

impl<B: Backend> AnfisModel<B> {
    pub fn evaluate<D: TabularDataset, Y: TargetVector>(
        &mut self,
        x_test: &D,
        y_test: &Y,
        fallback: Option<f64>,
        output_path: Option<&Path>,
    ) -> Result<EvaluationMetrics, ModelError> {
        let metrics = self.evaluate_regression(x_test, y_test, fallback, output_path)?;
        Ok(EvaluationMetrics::Regression(metrics))
    }

    pub fn from_config(config: ModelConfig, device: Device) -> Result<Self, ModelError> {
        if config.kind == MFKind::Triangular {
            return Err(ModelError::InvalidStrategy(
                "ANFIS model requires differentiable membership functions (e.g., Gaussian). Triangular MFs are not supported.".into()
            ));
        }

        let mut model = AnfisModel::new(device);
        model.base.kind = Some(config.kind);

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
        }

        Ok(model)
    }
}
