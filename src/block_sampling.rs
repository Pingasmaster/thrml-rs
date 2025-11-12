use crate::block_management::{Block, BlockSpec, BlockState, GlobalState, block_state_to_global};
use crate::conditional_samplers::AbstractConditionalSampler;
use crate::interaction::InteractionGroup;
use rand::RngCore;
use std::collections::HashMap;

/// Specification for Gibbs sampling with free and clamped blocks.
pub struct BlockGibbsSpec {
    pub block_spec: BlockSpec,
    pub free_blocks: Vec<Block>,
    pub clamped_blocks: Vec<Block>,
    pub sampling_order: Vec<Vec<usize>>,
    block_index_map: HashMap<Block, usize>,
}

impl BlockGibbsSpec {
    pub fn new(free_super_blocks: Vec<Vec<Block>>, clamped_blocks: Vec<Block>) -> Self {
        let mut free_blocks = Vec::new();
        let mut sampling_order = Vec::new();

        for super_block in free_super_blocks {
            let mut group = Vec::new();
            for block in super_block {
                group.push(free_blocks.len());
                free_blocks.push(block);
            }
            if !group.is_empty() {
                sampling_order.push(group);
            }
        }

        let mut all_blocks = free_blocks.clone();
        all_blocks.extend(clamped_blocks.clone());
        let block_spec = BlockSpec::new(all_blocks);

        let block_index_map = free_blocks
            .iter()
            .cloned()
            .enumerate()
            .map(|(index, block)| (block, index))
            .collect();

        Self {
            block_spec,
            free_blocks,
            clamped_blocks,
            sampling_order,
            block_index_map,
        }
    }
}

/// Holds the sampler state for each block sampling program.
pub struct BlockSamplingProgram {
    pub gibbs_spec: BlockGibbsSpec,
    pub samplers: Vec<Box<dyn AbstractConditionalSampler>>,
    per_block_interactions: Vec<Vec<InteractionGroup>>,
}

impl BlockSamplingProgram {
    pub fn new(
        gibbs_spec: BlockGibbsSpec,
        samplers: Vec<Box<dyn AbstractConditionalSampler>>,
        interaction_groups: Vec<InteractionGroup>,
    ) -> Self {
        if samplers.len() != gibbs_spec.free_blocks.len() {
            panic!("Sampler count must match number of free blocks");
        }

        let mut per_block_interactions: Vec<Vec<InteractionGroup>> =
            Vec::with_capacity(gibbs_spec.free_blocks.len());
        per_block_interactions.resize_with(gibbs_spec.free_blocks.len(), Vec::new);
        for group in interaction_groups {
            let block_index = gibbs_spec
                .block_index_map
                .get(&group.head_nodes)
                .expect("Interaction head block is not free");
            per_block_interactions[*block_index].push(group);
        }

        Self {
            gibbs_spec,
            samplers,
            per_block_interactions,
        }
    }

    fn build_global_state(
        &self,
        state_free: &[BlockState],
        state_clamp: &[BlockState],
    ) -> GlobalState {
        let mut combined = Vec::with_capacity(state_free.len() + state_clamp.len());
        combined.extend(state_free.iter());
        combined.extend(state_clamp.iter());
        block_state_to_global(&combined)
    }

    fn sample_single_block(
        &self,
        rng: &mut dyn RngCore,
        state_free: &mut [BlockState],
        state_clamp: &[BlockState],
        block_index: usize,
    ) {
        let global_state = self.build_global_state(state_free, state_clamp);
        let block = &self.gibbs_spec.free_blocks[block_index];
        let mut logits = vec![0.0; block.len()];

        for group in &self.per_block_interactions[block_index] {
            let contributions = group.evaluate(&global_state, &self.gibbs_spec.block_spec);
            for (i, value) in contributions.iter().enumerate() {
                logits[i] += value;
            }
        }

        let sampler = &self.samplers[block_index];
        let new_values = sampler.sample(rng, &logits);
        state_free[block_index] = BlockState::new(new_values);
    }

    pub fn sample_blocks(
        &self,
        rng: &mut dyn RngCore,
        state_free: &mut [BlockState],
        state_clamp: &[BlockState],
    ) {
        for group in &self.gibbs_spec.sampling_order {
            for &block_index in group {
                self.sample_single_block(rng, state_free, state_clamp, block_index);
            }
        }
    }
}

/// Represents a sampling schedule.
pub struct SamplingSchedule {
    pub n_warmup: usize,
    pub n_samples: usize,
    pub steps_per_sample: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_management::{BlockState, NodeValue};
    use crate::conditional_samplers::BernoulliConditional;
    use crate::interaction::{InteractionGroup, SpinBiasEvaluator};
    use crate::pgm::SpinNode;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use std::sync::Arc;

    #[test]
    fn block_sampling_updates_states() {
        let node = SpinNode::new().into();
        let block = Block::new(vec![node]);
        let spec = BlockGibbsSpec::new(vec![vec![block.clone()]], vec![]);
        let interaction = InteractionGroup::new(
            block.clone(),
            vec![],
            Box::new(SpinBiasEvaluator::new(Arc::new(vec![5.0]), block.len())),
        );
        let program = BlockSamplingProgram::new(
            spec,
            vec![Box::new(BernoulliConditional)],
            vec![interaction],
        );

        let mut rng = StdRng::seed_from_u64(42);
        let mut state_free = vec![BlockState::zeros(block.kind(), block.len())];
        let state_clamped = vec![];

        program.sample_blocks(&mut rng, &mut state_free, &state_clamped);

        assert_eq!(state_free[0].len(), 1);
        assert!(matches!(state_free[0].values[0], NodeValue::Spin(_)));
    }
}
