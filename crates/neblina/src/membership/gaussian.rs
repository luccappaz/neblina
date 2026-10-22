use serde::{Deserialize, Serialize};

use crate::{
    fuzzy::granularity::Granularity,
    membership::{
        FuzzyStrategy, MFKind,
        mb_function::{MembershipOp, Partitionable},
    },
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GaussianMF {
    pub center: f64,
    pub sigma: f64,
}

impl MembershipOp for GaussianMF {
    fn eval(&self, x: f64) -> f64 {
        let diff = (x - self.center) / self.sigma;
        (-0.5 * diff * diff).exp()
    }

    fn center(&self) -> f64 {
        self.center
    }

    fn support(&self) -> (f64, f64) {
        // Effective support roughly within 3 standard deviations
        (
            self.center - 3.0 * self.sigma,
            self.center + 3.0 * self.sigma,
        )
    }
}

impl Partitionable for GaussianMF {
    fn create_partition(
        _kind: MFKind,
        strategy: FuzzyStrategy,
        granularity: Granularity,
    ) -> Vec<Self> {
        let knots = strategy.resolve_knots(granularity);
        let n = knots.len();

        if n == 0 {
            return vec![];
        }

        if n == 1 {
            return vec![GaussianMF {
                center: knots[0],
                sigma: 1.0,
            }];
        }

        let mut mfs = Vec::with_capacity(n);
        for i in 0..n {
            let center = knots[i];

            let sigma = if i == 0 {
                (knots[1] - knots[0]).abs()
            } else if i == n - 1 {
                (knots[n - 1] - knots[n - 2]).abs()
            } else {
                ((knots[i + 1] - knots[i - 1]) / 2.0).abs()
            };

            let sigma = if sigma < f64::EPSILON { 1.0 } else { sigma };

            mfs.push(GaussianMF { center, sigma });
        }

        mfs
    }
}
