use crate::block_management::{Block, BlockSpec, BlockState, GlobalState, block_state_to_global};

/// Base trait for energy-based models.
pub trait AbstractEBM {
    fn energy(&self, state: &[BlockState], blocks: &[Block]) -> f64;
}

/// A factor that contributes an energy term given a global state.
pub trait EBMFactor: crate::factor::AbstractFactor {
    fn energy(&self, global_state: &GlobalState, block_spec: &BlockSpec) -> f64;
}

/// Combines multiple factors into a single model.
pub struct FactorizedEBM {
    factors: Vec<Box<dyn EBMFactor>>,
}

impl FactorizedEBM {
    pub fn new(factors: Vec<Box<dyn EBMFactor>>) -> Self {
        // Groups energy contributors into one combined model for convenience.
        Self { factors }
    }
}

impl AbstractEBM for FactorizedEBM {
    fn energy(&self, state: &[BlockState], blocks: &[Block]) -> f64 {
        let spec = BlockSpec::new(blocks.to_vec());
        let global = block_state_to_global(&state.iter().collect::<Vec<_>>());
        self.factors
            .iter()
            .map(|factor| factor.energy(&global, &spec))
            .sum()
    }
}
