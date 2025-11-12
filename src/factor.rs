use crate::block_management::{Block, BlockState};
use crate::block_sampling::{BlockGibbsSpec, BlockSamplingProgram};
use crate::conditional_samplers::AbstractConditionalSampler;
use crate::interaction::InteractionGroup;
use rand::RngCore;

/// Factor that can emit interaction groups influencing a Gibbs program.
pub trait AbstractFactor {
    fn interaction_groups(&self) -> Vec<InteractionGroup>;
}

/// Basic weighted factor helper storing nodes and weights.
pub struct WeightedFactor {
    pub node_groups: Vec<Block>,
    pub weights: Vec<f64>,
}

impl WeightedFactor {
    pub fn new(weights: Vec<f64>, node_groups: Vec<Block>) -> Self {
        // Pairs each block with its corresponding weight tensor for downstream evaluators.
        Self {
            weights,
            node_groups,
        }
    }
}

pub struct FactorSamplingProgram {
    pub inner: BlockSamplingProgram,
}

impl FactorSamplingProgram {
    /// Assemble a block sampling program from factors.
    pub fn new(
        gibbs_spec: BlockGibbsSpec,
        samplers: Vec<Box<dyn AbstractConditionalSampler>>,
        factors: Vec<Box<dyn AbstractFactor>>,
    ) -> Self {
        let mut interactions = Vec::new();
        for factor in factors {
            interactions.extend(factor.interaction_groups());
        }
        // Flatten interactions from all factors before constructing the shared sampling loop.
        let inner = BlockSamplingProgram::new(gibbs_spec, samplers, interactions);
        Self { inner }
    }

    pub fn sample_blocks(
        &self,
        rng: &mut dyn RngCore,
        state_free: &mut [BlockState],
        state_clamp: &[BlockState],
    ) {
        self.inner.sample_blocks(rng, state_free, state_clamp);
    }
}
