# Neblina WM 🌫️

**Neblina WM** (`nablina-wm`) is a high-performance **Rust** crate implementing the classic **Wang-Mendel** algorithm for automatic fuzzy rule generation, inference, and evaluation, built on top of the **Neblina** fuzzy logic core.

---

## 🚀 Features

* **Automatic Rule Induction**: Induces fuzzy rules from tabular datasets with configurable support and confidence (Degree of Confidence - DoC) thresholds.
* **Dual Consequent Support**:
* **Case 1**: Linguistic consequents mapped to fuzzy partitions of the target variable $y$.
* **Case 2**: Direct numeric consequents with weighted averages and variances.


* **Flexible Inference & Defuzzification**: Batch and single-sample inference supporting multiple T-Norms (`Product`, `Minimum`) and weighted-average defuzzification.
* **Robust Fallback Strategies**: Automatic nearest-rule lookup in Euclidean input space when no rules fire directly for an input sample.
* **Full Ecosystem Integration**: Seamless integration with Polars DataFrames, automated regression/classification metric evaluation, and JSON serialization/deserialization for rule persistence.

---

## 📦 Installation

Add the crate to your `Cargo.toml`:

```toml
[dependencies]
neblina = "0.0.1"
nablina-wm = "0.0.1"

```

---

## 🛠️ Quick Start

```rust
use neblina::prelude::*;
use neblina::types::HashDataset;
use nablina_wm::wm::WMModel;
use std::collections::HashMap;

fn main() -> Result<(), ModelError> {
    let mut model = WMModel::new();

    // 1. Configure input space and granularity
    model.set_kind(MFKind::Triangular)?
         .add_granularity("x1", 3)?
         .add_linear_strategy("x1", 0.0, 10.0)?;

    // 2. Build and validate internal structures
    model.build()?;

    // 3. Create a simulated training dataset
    let mut x_train: HashDataset = Vec::new();
    let mut row1 = HashMap::new();
    row1.insert("x1".to_string(), 2.0);
    x_train.push(row1);

    let mut row2 = HashMap::new();
    row2.insert("x1".to_string(), 8.0);
    x_train.push(row2);

    let y_train = vec![10.0, 90.0];

    // 4. Generate Wang-Mendel fuzzy rules automatically
    model.generate_rules(&x_train, &y_train, None, TNorm::Product, 0.0, 0.0)?;
    println!("Rules generated successfully! Total: {}", model.rule_count());

    // 5. Infer value for a new sample
    let mut sample = HashMap::new();
    sample.insert("x1".to_string(), 5.0);
    let (y_pred, _activated_rules) = model.infer(&sample, TNorm::Product);

    println!("Inferred value (y): {:?}", y_pred);
    Ok(())
}

```

---

## 🧪 Testing

Run the unit test suite with Cargo:

```bash
cargo test --all-features

```

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
