use crate::block_management::from_global_state;
use crate::block_management::{Block, BlockState, NodeValue, block_state_to_global};
use crate::block_sampling::BlockSamplingProgram;
use crate::pgm::Node;
use std::sync::Arc;

/// Observer trait that can inspect sampling programs at each iteration.
pub trait AbstractObserver {
    type Carry;
    type Output;

    fn init(&self) -> Self::Carry;

    fn observe(
        &self,
        program: &BlockSamplingProgram,
        state_free: &[BlockState],
        state_clamp: &[BlockState],
        carry: Self::Carry,
        iteration: usize,
    ) -> (Self::Carry, Self::Output);
}

/// Observer that returns the raw states of blocks.
pub struct StateObserver {
    pub blocks_to_sample: Vec<Block>,
}

impl StateObserver {
    pub fn new(blocks: Vec<Block>) -> Self {
        Self {
            blocks_to_sample: blocks,
        }
    }
}

impl AbstractObserver for StateObserver {
    type Carry = ();
    type Output = Vec<BlockState>;

    fn init(&self) -> Self::Carry {
        ()
    }

    fn observe(
        &self,
        program: &BlockSamplingProgram,
        state_free: &[BlockState],
        state_clamp: &[BlockState],
        carry: Self::Carry,
        _iteration: usize,
    ) -> (Self::Carry, Self::Output) {
        let mut combined = Vec::with_capacity(state_free.len() + state_clamp.len());
        combined.extend(state_free.iter());
        combined.extend(state_clamp.iter());
        let global_state = block_state_to_global(&combined);
        let sampled_blocks = from_global_state(
            &global_state,
            &program.gibbs_spec.block_spec,
            &self.blocks_to_sample,
        );
        (carry, sampled_blocks)
    }
}

/// Observer that accumulates moments defined over nodes.
pub struct MomentAccumulatorObserver {
    moment_spec: Vec<Vec<Vec<Node>>>,
    f_transform: Arc<dyn Fn(&NodeValue) -> f64 + Send + Sync>,
}

impl MomentAccumulatorObserver {
    pub fn new<F>(moment_spec: Vec<Vec<Vec<Node>>>, f_transform: F) -> Self
    where
        F: Fn(&NodeValue) -> f64 + Send + Sync + 'static,
    {
        Self {
            moment_spec,
            f_transform: Arc::new(f_transform),
        }
    }
}

impl AbstractObserver for MomentAccumulatorObserver {
    type Carry = Vec<Vec<f64>>;
    type Output = Vec<Vec<f64>>;

    fn init(&self) -> Self::Carry {
        self.moment_spec
            .iter()
            .map(|group| vec![0.0; group.len()])
            .collect()
    }

    fn observe(
        &self,
        program: &BlockSamplingProgram,
        state_free: &[BlockState],
        state_clamp: &[BlockState],
        mut carry: Self::Carry,
        _iteration: usize,
    ) -> (Self::Carry, Self::Output) {
        let mut combined = Vec::with_capacity(state_free.len() + state_clamp.len());
        combined.extend(state_free.iter());
        combined.extend(state_clamp.iter());
        let global_state = block_state_to_global(&combined);
        let spec = &program.gibbs_spec.block_spec;

        for (group_index, group) in self.moment_spec.iter().enumerate() {
            for (moment_index, nodes) in group.iter().enumerate() {
                let mut product = 1.0;
                for node in nodes {
                    let idx = spec.node_location[node];
                    product *= (self.f_transform)(&global_state[idx]);
                }
                carry[group_index][moment_index] += product;
            }
        }

        (carry.clone(), carry)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_management::{Block, BlockState};
    use crate::block_sampling::BlockGibbsSpec;
    use crate::pgm::SpinNode;

    #[test]
    fn state_observer_returns_values() {
        let node = SpinNode::new();
        let block = Block::new(vec![node.into()]);
        let spec = BlockGibbsSpec::new(vec![vec![block.clone()]], vec![]);
        let interaction = Vec::new();
        let program = BlockSamplingProgram::new(
            spec,
            vec![Box::new(crate::conditional_samplers::BernoulliConditional)],
            interaction,
        );

        let observer = StateObserver::new(vec![block.clone()]);
        let carry = observer.init();
        let global_state = BlockState::zeros(block.kind(), block.len());
        let (_carry, state) = observer.observe(&program, &[global_state.clone()], &[], carry, 0);
        assert_eq!(state.len(), 1);
        assert_eq!(state[0].len(), 1);
    }

    #[test]
    fn moment_observer_accumulates() {
        let node = SpinNode::new();
        let block = Block::new(vec![node.into()]);
        let spec = BlockGibbsSpec::new(vec![vec![block.clone()]], vec![]);
        let program = BlockSamplingProgram::new(
            spec,
            vec![Box::new(crate::conditional_samplers::BernoulliConditional)],
            Vec::new(),
        );

        let spec = vec![vec![vec![node.into()]]];
        let observer = MomentAccumulatorObserver::new(spec, |value| match value {
            NodeValue::Spin(true) => 1.0,
            NodeValue::Spin(false) => -1.0,
            _ => 0.0,
        });
        let carry = observer.init();
        let (_carry, sums) = observer.observe(
            &program,
            &[BlockState::zeros(block.kind(), block.len())],
            &[],
            carry,
            0,
        );
        assert_eq!(sums.len(), 1);
        assert_eq!(sums[0].len(), 1);
    }
}
