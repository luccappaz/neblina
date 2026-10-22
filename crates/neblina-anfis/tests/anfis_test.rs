use burn::{backend::Flex, tensor::Device};
use neblina_anfis::anfis::AnfisModel;
use std::collections::HashMap;

use neblina::{
    membership::MFKind,
    model::{
        FuzzyModel, ModelConfig, TrainableModel, builder::FuzzyBuilder,
        evaluation::FuzzyEvaluationExt as _,
    },
    types::HashDataset,
};

type TestBackend = Flex;

#[test]
fn test_anfis_rejects_triangular() {
    let device = Device::default();
    let mut model: AnfisModel<TestBackend> = AnfisModel::new(device);

    // ANFIS must reject triangular membership functions due to non-differentiability
    let res = model.set_kind("triangular");
    assert!(
        res.is_err(),
        "Expected error when setting triangular kind on ANFIS, but got Ok"
    );
}

#[test]
fn test_anfis_builder_and_build() {
    let device = Device::default();
    let mut model: AnfisModel<TestBackend> = AnfisModel::new(device);

    model.set_kind("gaussian").unwrap();
    model.add_linear_strategy("x1", 0.0, 10.0).unwrap();
    model.add_granularity("x1", 3).unwrap();

    let build_res = model.build();
    assert!(
        build_res.is_ok(),
        "ANFIS build failed: {:?}",
        build_res.err()
    );
    assert!(model.base.is_configured());
}

#[test]
fn test_anfis_predict_batch() {
    let device = Device::default();
    let mut model: AnfisModel<TestBackend> = AnfisModel::new(device);

    model.set_kind("gaussian").unwrap();
    model.add_linear_strategy("input_a", -5.0, 5.0).unwrap();

    let mut row1 = HashMap::new();
    row1.insert("input_a".to_string(), 1.2);
    let mut row2 = HashMap::new();
    row2.insert("input_a".to_string(), -2.1);

    let dataset: HashDataset = vec![row1, row2];

    let preds = model.predict(&dataset, Some(0.5), None);
    assert!(preds.is_ok(), "Prediction failed: {:?}", preds.err());

    let (continuous, binary) = preds.unwrap();
    assert_eq!(continuous.len(), 2);
    assert_eq!(binary.len(), 2);
}

#[test]
fn test_anfis_train_and_evaluate() {
    let device = Device::default();
    let mut model: AnfisModel<TestBackend> = AnfisModel::new(device);

    model.set_kind("gaussian").unwrap();
    model.add_linear_strategy("feature1", 0.0, 10.0).unwrap();
    model.add_granularity("feature1", 3).unwrap();
    model.with_epochs(3);

    let mut row = HashMap::new();
    row.insert("feature1".to_string(), 5.0);

    let x_train: HashDataset = vec![row.clone(), row];
    let y_train: Vec<f64> = vec![3.5, 4.0];

    let config = ModelConfig {
        kind: MFKind::Gaussian,
        features: HashMap::new(),
        target: None,
    };

    let train_res = model.train(&x_train, &y_train, &config);
    assert!(
        train_res.is_ok(),
        "ANFIS training failed: {:?}",
        train_res.err()
    );

    let eval_res = model.evaluate_regression(&x_train, &y_train, None, None);
    assert!(eval_res.is_ok(), "Evaluation failed: {:?}", eval_res.err());

    let metrics = eval_res.unwrap();
    assert!(metrics.mse >= 0.0);
    assert!(metrics.rmse >= 0.0);
}
