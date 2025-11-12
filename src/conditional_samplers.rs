use crate::block_management::NodeValue;
use rand::{Rng, RngCore};

/// Base sampler trait.
pub trait AbstractConditionalSampler {
    /// Samples a block of nodes given per-node logits.
    fn sample(&self, rng: &mut dyn RngCore, logits: &[f64]) -> Vec<NodeValue>;
}

/// Bernoulli sampler for spin nodes.
pub struct BernoulliConditional;

impl BernoulliConditional {
    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }
}

impl AbstractConditionalSampler for BernoulliConditional {
    fn sample(&self, rng: &mut dyn RngCore, logits: &[f64]) -> Vec<NodeValue> {
        logits
            .iter()
            .map(|&logit| {
                let prob = Self::sigmoid(logit);
                let draw: f64 = rng.random();
                NodeValue::Spin(draw < prob)
            })
            .collect()
    }
}

/// Softmax sampler for categorical nodes.
pub struct SoftmaxConditional {
    /// Number of categories.
    pub n_categories: usize,
}

impl SoftmaxConditional {
    pub fn new(n_categories: usize) -> Self {
        Self { n_categories }
    }

    fn softmax(logits: &[f64]) -> Vec<f64> {
        let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_vals: Vec<f64> = logits.iter().map(|v| (v - max).exp()).collect();
        let sum: f64 = exp_vals.iter().sum();
        exp_vals.iter().map(|x| x / sum).collect()
    }
}

impl AbstractConditionalSampler for SoftmaxConditional {
    fn sample(&self, rng: &mut dyn RngCore, logits: &[f64]) -> Vec<NodeValue> {
        if logits.len() % self.n_categories != 0 {
            panic!("Expected logits length divisible by n_categories");
        }
        let nodes = logits.len() / self.n_categories;
        (0..nodes)
            .map(|i| {
                let slice = &logits[i * self.n_categories..(i + 1) * self.n_categories];
                let probs = Self::softmax(slice);
                let mut cumulative = 0.0;
                let draw: f64 = rng.random();
                for (idx, &p) in probs.iter().enumerate() {
                    cumulative += p;
                    if draw <= cumulative {
                        return NodeValue::Categorical(idx as u8);
                    }
                }
                NodeValue::Categorical((self.n_categories - 1) as u8)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn bernoulli_sample_shape() {
        let sampler = BernoulliConditional;
        let mut rng = StdRng::seed_from_u64(42);
        let logits = vec![0.0, 1.0, -2.0];
        let result = sampler.sample(&mut rng, &logits);
        assert_eq!(result.len(), logits.len());
    }

    #[test]
    fn softmax_sample_categories() {
        let sampler = SoftmaxConditional::new(3);
        let mut rng = StdRng::seed_from_u64(43);
        let logits = vec![0.0, 0.0, 1.0, 0.0, -1.0, -2.0];
        let result = sampler.sample(&mut rng, &logits);
        assert_eq!(result.len(), 2);
        assert!(matches!(result[0], NodeValue::Categorical(_)));
    }
}
