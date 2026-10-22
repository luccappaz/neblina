# Neblina 🌫️

**Neblina** is a high-performance **Rust** core library for **Fuzzy Logic and Inference Systems**, built with a strict focus on type safety, mathematical invariants, expressiveness, and seamless integration with the data science ecosystem (including native support for **Polars**).

---

## 🚀 Features

* **Flexible Membership Functions**: Out-of-the-box support for **Triangular** and **Gaussian** membership functions, implementing clean traits (`MembershipOp` and `Partitionable`).
* **Robust T-Norm Operators**: Support for `Product` and `Minimum` T-Norm combinations with short-circuit evaluation.
* **Validated Partition Strategies**: Granularity partitioning bounds ($N \ge 2$), linear uniform grids, and custom knot sequences validated at compile/runtime.
* **Data Science Integration**: Native `TabularDataset` and `TargetVector` abstractions, featuring robust serialization/deserialization via `serde` and high-performance processing via **Polars**.
* **Built-in Metrics & Utilities**: Evaluation extensions for regression (MSE, RMSE, MAE, $R^2$, MAPE) and classification metrics, alongside automated chart plotting using `plotters`.

---

## 📦 Installation

Add the crate to your `Cargo.toml`:

```toml
[dependencies]
neblina = { version = "0.0.1", features = ["polars", "plotters"] }

```

### Cargo Features

* `polars`: Enables native implementations of `TabularDataset` and `TargetVector` for Polars `DataFrame` and `Series`.

---

## 🛠️ Quick Start

```rust
use neblina::prelude::*;
use neblina::model::base::FuzzyBaseModel;
use neblina::model::traits::FuzzyBuilder;

fn main() -> Result<(), ModelError> {
    let mut model = FuzzyBaseModel::new();

    // Configure membership function kind and input strategies
    model.set_kind(MFKind::Triangular)?
         .add_linear_strategy("x1", 0.0, 100.0)?
         .add_custom_strategy("x2", vec![0.0, 25.0, 75.0, 100.0])?;

    // Build and validate internal partitions and centers
    model.build_inputs()?;

    println!("Neblina core successfully configured!");
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
