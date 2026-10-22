# Neblina 🌫️

**Neblina** é uma suíte de alta performance em **Rust** para modelagem de **Lógica Fuzzy, Sistemas de Inferência e Redes Neuro-Fuzzy**, projetada com foco estrito em segurança de tipos, expressividade, arquitetura limpa e integração nativa com o ecossistema de ciência de dados.

---

## 📦 Arquitetura da Suíte

A suíte é dividida em crates modulares para separar o núcleo matemático das estratégias de aprendizado e modelagem adaptativa:

| Crate | Descrição | Principais Recursos |
| --- | --- | --- |
| **`neblina`** | Núcleo Base & Infraestrutura | Funções de pertinência (Triangular, Gaussiana), T-Norms (`Product`, `Minimum`), estratégias de particionamento dinâmico (`Linear`, `Custom`), adaptadores de dataset genéricos (`TabularDataset`, `Polars`), plotagem de MFs, métricas de avaliação e tratamento de erros robusto via `thiserror`. |
| **`nablina-wm`** | Modelo de Wang-Mendel | Indução automática de regras fuzzy a partir de dados, suporte a **Caso 1** (consequentes linguísticos) e **Caso 2** (consequentes numéricos), defuzificação por média ponderada e busca por menor distância euclidiana. |
| **`neblina-anfis`** | Modelo ANFIS | Redes neuro-fuzzy adaptativas baseadas em **Burn** (arquitetura Takagi-Sugeno de 5 camadas), otimização por gradiente descendente (Adam) sobre funções de pertinência gaussianas diferenciáveis, extração de regras em linguagem natural e persistência em JSON. |

---

## 🚀 Instalação

Adicione os crates desejados ao seu `Cargo.toml`:

```toml
[dependencies]
neblina = { version = "0.2.0", features = ["polars"] }
nablina-wm = "0.2.0"
neblina-anfis = "0.2.0"

```

---

## 🛠️ Exemplos de Uso

### 1. Configurando o Núcleo com o Prelude (`neblina`)

```rust
use neblina::prelude::*;
use neblina::model::base::FuzzyBaseModel;
use neblina::model::traits::FuzzyBuilder;

fn main() -> Result<(), ModelError> {
    let mut model = FuzzyBaseModel::new();

    model.set_kind(MFKind::Triangular)?
         .add_linear_strategy("x1", 0.0, 100.0)?
         .add_custom_strategy("x2", vec![0.0, 25.0, 75.0, 100.0])?;

    model.build_inputs()?;

    println!("Suíte Neblina configurada com sucesso!");
    Ok(())
}

```

### 2. Indução e Inferência por Wang-Mendel (`nablina-wm`)

```rust
use neblina::prelude::*;
use neblina::types::HashDataset;
use nablina_wm::wm::WMModel;
use std::collections::HashMap;

fn main() -> Result<(), ModelError> {
    let mut model = WMModel::new();

    model.set_kind(MFKind::Triangular)?
         .add_granularity("x1", 3)?
         .add_linear_strategy("x1", 0.0, 10.0)?;

    model.build()?;

    let mut x_train: HashDataset = Vec::new();
    let mut row1 = HashMap::new();
    row1.insert("x1".to_string(), 2.0);
    x_train.push(row1);

    let mut row2 = HashMap::new();
    row2.insert("x1".to_string(), 8.0);
    x_train.push(row2);

    let y_train = vec![10.0, 90.0];

    model.generate_rules(&x_train, &y_train, None, TNorm::Product, 0.0, 0.0)?;

    let mut sample = HashMap::new();
    sample.insert("x1".to_string(), 5.0);
    let (y_pred, _activated_rules) = model.infer(&sample, TNorm::Product);

    println!("Valor inferido (y): {:?}", y_pred);
    Ok(())
}

```

### 3. Treinamento de Redes Neuro-Fuzzy (`neblina-anfis`)

```rust
use neblina::prelude::*;
use neblina::types::HashDataset;
use neblina_anfis::AnfisModel;
use burn::backend::Autodiff;
use burn::backend::ndarray::NdArray;
use std::collections::HashMap;

fn main() -> Result<(), ModelError> {
    type MyBackend = Autodiff<NdArray<f32>>;
    let device = Default::default();

    let mut model = AnfisModel::<MyBackend>::new(device);

    model.set_kind("gaussian")?
         .add_linear_strategy("x1", 0.0, 10.0)?;

    model.with_epochs(50)
         .with_learning_rate(0.01);

    model.build()?;

    let mut x_train: HashDataset = Vec::new();
    let mut row1 = HashMap::new();
    row1.insert("x1".to_string(), 2.0);
    x_train.push(row1);

    let mut row2 = HashMap::new();
    row2.insert("x1".to_string(), 8.0);
    x_train.push(row2);

    let y_train = vec![5.0, 25.0];

    model.train(&x_train, &y_train, &ModelConfig {
        kind: MFKind::Gaussian,
        features: HashMap::new(),
        target: None,
    })?;

    let rules = model.extract_rules()?;
    for rule in rules {
        println!("Regra extraída: {}", rule.text_representation);
    }

    Ok(())
}

```

---

## 🧪 Testes

Execute a suíte completa de testes unitários:

```bash
cargo test --all-features

```

---

## 📄 Licença

Distribuído sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.
