# Neblina ANFIS 🌫️

**Neblina ANFIS** (`neblina-anfis`) is an **Adaptive Neuro-Fuzzy Inference System (ANFIS)** crate implemented in **Rust**. Built on top of the **Burn** deep learning framework and the **Neblina** fuzzy core, it provides GPU/CPU-accelerated Takagi-Sugeno neuro-fuzzy modeling and backpropagation optimization.

---

## 🚀 Features

* **5-Layer Takagi-Sugeno Architecture**: Implements the complete ANFIS neural network pipeline (fuzzification, firing strength computation, normalization, linear consequent evaluation, and weighted output aggregation).
* **Burn Framework Integration**: Leverages Burn tensor abstractions for flexible backend execution and gradient-based optimization using the Adam optimizer.
* **Differentiable Membership Functions**: Uses Gaussian membership functions ($\mu(x) = \exp(-0.5 \cdot ((x-c)/\sigma)^2)$) with dynamic parameter optimization for premise parameters ($c$ and $\sigma$).
* **Natural Language Rule Extraction**: Automatically extracts optimized premise and linear consequent parameters into human-readable Takagi-Sugeno fuzzy rules.
* **Complete Persistence & Evaluation**: Supports JSON rule export, Polars DataFrame integration, and automated regression metric evaluation ($MSE$, $RMSE$, $MAE$, $R^2$, $MAPE$).

---

## 📦 Installation

Add the crates to your `Cargo.toml`:

```toml
[dependencies]
neblina = "0.0.1"
neblina-anfis = "0.0.1"
burn = { version = "0.22.0", features = ["flex"] }

```

---

## 🛠️ Quick Start

```rust
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Define Burn backend and execution device
    type TestBackend = Flex;
    let device = Device::default();

    let mut model: AnfisModel<TestBackend> = AnfisModel::new(device);

    // 2. Configure Gaussian membership functions (Triangular is rejected due to non-differentiability)
    model.set_kind("gaussian")?;
    model.add_linear_strategy("feature1", 0.0, 10.0)?;
    model.add_granularity("feature1", 3)?;

    // 3. Set training hyperparameters
    model.with_epochs(3);
    model.build()?;

    // 4. Prepare training dataset
    let mut row = HashMap::new();
    row.insert("feature1".to_string(), 5.0);

    let x_train: HashDataset = vec![row.clone(), row];
    let y_train: Vec<f64> = vec![3.5, 4.0];

    let config = ModelConfig {
        kind: MFKind::Gaussian,
        features: HashMap::new(),
        target: None,
    };

    // 5. Train neuro-fuzzy parameters via backpropagation (Adam)
    model.train(&x_train, &y_train, &config)?;

    // 6. Evaluate regression performance
    let metrics = model.evaluate_regression(&x_train, &y_train, None, None)?;
    println!("Training MSE: {:.4}", metrics.mse);

    // 7. Extract optimized fuzzy rules
    let rules = model.extract_rules()?;
    for rule in rules {
        println!("Extracted Rule: {}", rule.text_representation);
    }

    Ok(())
}

```

---

## 🧪 Testes

Execute a suíte completa de testes unitários com o Cargo:

```bash
cargo test --all-features

```

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
