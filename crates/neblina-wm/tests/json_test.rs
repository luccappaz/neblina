// crates/wang-mendel/tests/json_test.rs

use neblina::membership::mb_function::MembershipOp;
use neblina_wm::wm::WMModel;

#[test]
fn test_load_model_from_compact_json() {
    let json_spec = r#"{
        "kind": "triangular",
        "features": {
            "x1": {
                "granularity": 3,
                "strategy": {
                    "type": "linear",
                    "min": 0.0,
                    "max": 10.0
                },
                "labels": ["low", "medium", "high"]
            },
            "x2": {
                "strategy": {
                    "type": "custom",
                    "knots": [0.0, 5.0, 10.0]
                }
            }
        }
    }"#;

    // Build and validate model directly from the JSON string
    let model = WMModel::from_json(json_spec).unwrap();

    // Verify x1 (Linear strategy with custom labels)
    let x1_mfs = model.mf_configs().unwrap().get("x1").unwrap();
    assert!(x1_mfs.contains_key("low"));
    assert!(x1_mfs.contains_key("medium"));
    assert!(x1_mfs.contains_key("high"));

    // Verify x2 (Custom strategy with inferred granularity = 3)
    let x2_mfs = model.mf_configs().unwrap().get("x2").unwrap();
    assert_eq!(x2_mfs.len(), 3);
}

#[test]
fn test_load_model_from_compact_json_with_target() {
    let json_spec = r#"{
        "kind": "triangular",
        "features": {
            "x1": {
                "granularity": 3,
                "strategy": {
                    "type": "linear",
                    "min": 0.0,
                    "max": 10.0
                },
                "labels": ["low", "medium", "high"]
            },
            "x2": {
                "strategy": {
                    "type": "custom",
                    "knots": [0.0, 5.0, 10.0]
                }
            }
        },
        "target": {
            "granularity": 3,
            "strategy": {
                "type": "linear",
                "min": 0.0,
                "max": 100.0
            },
            "labels": ["small", "moderate", "large"]
        }
    }"#;

    // Build and validate model directly from the JSON string
    let model = WMModel::from_json(json_spec).unwrap();

    // Verify x1 (Linear strategy with custom labels)
    let x1_mfs = model.mf_configs().unwrap().get("x1").unwrap();
    assert!(x1_mfs.contains_key("low"));
    assert!(x1_mfs.contains_key("medium"));
    assert!(x1_mfs.contains_key("high"));

    // Verify x2 (Custom strategy with inferred granularity = 3)
    let x2_mfs = model.mf_configs().unwrap().get("x2").unwrap();
    assert_eq!(x2_mfs.len(), 3);

    // Verify target Y (Linear strategy with custom labels)
    assert_eq!(model.target_mfs().unwrap().len(), 3);
    assert!(model.target_mfs().unwrap().contains_key("small"));
    assert!(model.target_mfs().unwrap().contains_key("moderate"));
    assert!(model.target_mfs().unwrap().contains_key("large"));

    // Verify center of the "moderate" target membership function
    let moderate_center = model
        .target_mfs()
        .unwrap()
        .get("moderate")
        .unwrap()
        .center();
    assert!((moderate_center - 50.0).abs() < f64::EPSILON);
}
