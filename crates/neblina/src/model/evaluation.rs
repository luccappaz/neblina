use std::fs::File;
use std::path::Path;

use crate::dataset::{TabularDataset, TargetVector};
use crate::error::ModelError;
use crate::metrics::{ClassificationMetrics, EvaluationMetrics, RegressionMetrics};
use crate::model::FuzzyModel;

/// Extension trait providing unified regression and classification evaluation
/// for any model implementing `FuzzyModel<Output = (Vec<f64>, Vec<u32>)>`.
pub trait FuzzyEvaluationExt: FuzzyModel<Output = (Vec<f64>, Vec<u32>)> {
    /// Evaluates regression performance metrics (MSE, RMSE, MAE, R2, MAPE).
    fn evaluate_regression<D: TabularDataset, Y: TargetVector>(
        &mut self,
        x_test: &D,
        y_test: &Y,
        fallback: Option<f64>,
        output_path: Option<&Path>,
    ) -> Result<RegressionMetrics, ModelError> {
        let (y_pred, _) = self.predict(x_test, None, fallback)?;
        let y_true = y_test.to_vec_f64()?;
        let n = y_true.len() as f64;

        if n == 0.0 {
            return Ok(RegressionMetrics {
                mse: 0.0,
                rmse: 0.0,
                mae: 0.0,
                r2: 0.0,
                mape: 0.0,
            });
        }

        let y_true_mean = y_true.iter().sum::<f64>() / n;
        let mut ss_res = 0.0;
        let mut ss_tot = 0.0;
        let mut mae_sum = 0.0;
        let mut mape_sum = 0.0;

        for (&yt, &yp) in y_true.iter().zip(y_pred.iter()) {
            let diff = yt - yp;
            ss_res += diff * diff;
            ss_tot += (yt - y_true_mean).powi(2);
            mae_sum += diff.abs();

            if yt.abs() > f64::EPSILON {
                mape_sum += (diff / yt).abs();
            }
        }

        let mse = ss_res / n;
        let rmse = mse.sqrt();
        let mae = mae_sum / n;
        let r2 = if ss_tot > f64::EPSILON {
            1.0 - (ss_res / ss_tot)
        } else {
            0.0
        };
        let mape = (mape_sum / n) * 100.0;

        let metrics = RegressionMetrics {
            mse,
            rmse,
            mae,
            r2,
            mape,
        };

        if let Some(path) = output_path {
            let file = File::create(path).map_err(|e| ModelError::DatasetError(e.to_string()))?;
            serde_json::to_writer_pretty(file, &metrics)
                .map_err(|e| ModelError::DatasetError(e.to_string()))?;
            println!("✅ Regression metrics saved to: {:?}", path);
        }

        Ok(metrics)
    }

    /// Evaluates classification performance metrics (Accuracy, Precision, Recall, F1, Confusion Matrix).
    fn evaluate_classification<D: TabularDataset, Y: TargetVector>(
        &mut self,
        x_test: &D,
        y_test: &Y,
        threshold: Option<f64>,
        fallback: Option<f64>,
        output_path: Option<&Path>,
    ) -> Result<ClassificationMetrics, ModelError> {
        let (_, y_pred_bin) = self.predict(x_test, threshold, fallback)?;
        let y_true = y_test.to_vec_f64()?;

        let mut tp = 0;
        let mut tn = 0;
        let mut fp = 0;
        let mut fn_val = 0;

        for (&yt_f, &yp) in y_true.iter().zip(y_pred_bin.iter()) {
            let yt = yt_f.round() as u32;
            match (yt, yp) {
                (1, 1) => tp += 1,
                (0, 0) => tn += 1,
                (0, 1) => fp += 1,
                (1, 0) => fn_val += 1,
                _ => {}
            }
        }

        let total = (tp + tn + fp + fn_val) as f64;
        let accuracy = if total > 0.0 {
            (tp + tn) as f64 / total
        } else {
            0.0
        };
        let precision = if (tp + fp) > 0 {
            tp as f64 / (tp + fp) as f64
        } else {
            0.0
        };
        let recall = if (tp + fn_val) > 0 {
            tp as f64 / (tp + fn_val) as f64
        } else {
            0.0
        };
        let f1_score = if (precision + recall) > 0.0 {
            2.0 * (precision * recall) / (precision + recall)
        } else {
            0.0
        };

        let metrics = ClassificationMetrics {
            accuracy,
            precision,
            recall,
            f1_score,
            confusion_matrix: [[tn, fp], [fn_val, tp]],
        };

        if let Some(path) = output_path {
            let file = File::create(path).map_err(|e| ModelError::DatasetError(e.to_string()))?;
            serde_json::to_writer_pretty(file, &metrics)
                .map_err(|e| ModelError::DatasetError(e.to_string()))?;
            println!("✅ Classification metrics saved to: {:?}", path);
        }

        Ok(metrics)
    }

    /// Smart evaluation entry point routing to regression or classification based on threshold presence.
    fn evaluate<D: TabularDataset, Y: TargetVector>(
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
}

// Blanket Implementation: Any model matching the signature gets evaluation methods for free!
impl<T: FuzzyModel<Output = (Vec<f64>, Vec<u32>)>> FuzzyEvaluationExt for T {}
