use crate::block_management::{Block, BlockState, NodeValue, block_state_to_global};
use crate::block_sampling::BlockSamplingProgram;

pub trait AbstractObserver {
    fn observe(
        &mut self,
        program: &BlockSamplingProgram,
        state_free: &[BlockState],
        state_clamp: &[BlockState],
    ) -> Vec<NodeValue>;
}

pub struct StateObserver {
    pub blocks_to_sample: Vec<Block>,
}

impl StateObserver {
    pub fn new(blocks: Vec<Block>) -> Self {
        Self {
            blocks_to_sample: blocks,
        }
    }

    fn collect_state(
        &self,
        program: &BlockSamplingProgram,
        state_free: &[BlockState],
        state_clamp: &[BlockState],
    ) -> Vec<NodeValue> {
        let mut combined = Vec::with_capacity(state_free.len() + state_clamp.len());
        combined.extend(state_free.iter());
        combined.extend(state_clamp.iter());
        let global_state = block_state_to_global(&combined);

        self.blocks_to_sample
            .iter()
            .flat_map(|block| {
                block
                    .iter()
                    .map(|node| {
                        let idx = program
                            .gibbs_spec
                            .block_spec
                            .node_location
                            .get(node)
                            .expect("Node missing from global state");
                        global_state[*idx].clone()
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }
}

impl AbstractObserver for StateObserver {
    fn observe(
        &mut self,
        program: &BlockSamplingProgram,
        state_free: &[BlockState],
        state_clamp: &[BlockState],
    ) -> Vec<NodeValue> {
        self.collect_state(program, state_free, state_clamp)
    }
}

pub struct MomentAccumulatorObserver {
    pub blocks_to_sample: Vec<Block>,
    pub sums: Vec<f64>,
}

impl MomentAccumulatorObserver {
    pub fn new(blocks: Vec<Block>) -> Self {
        let sums = vec![0.0; blocks.iter().map(|b| b.len()).sum()];
        Self {
            blocks_to_sample: blocks,
            sums,
        }
    }

    pub fn accumulate(
        &mut self,
        program: &BlockSamplingProgram,
        state_free: &[BlockState],
        state_clamp: &[BlockState],
    ) {
        let observed = StateObserver::new(self.blocks_to_sample.clone()).collect_state(
            program,
            state_free,
            state_clamp,
        );
        for (i, value) in observed.into_iter().enumerate() {
            if let NodeValue::Spin(bit) = value {
                self.sums[i] += if bit { 1.0 } else { -1.0 };
            }
        }
    }
}

impl AbstractObserver for MomentAccumulatorObserver {
    fn observe(
        &mut self,
        program: &BlockSamplingProgram,
        state_free: &[BlockState],
        state_clamp: &[BlockState],
    ) -> Vec<NodeValue> {
        self.accumulate(program, state_free, state_clamp);
        Vec::new()
    }
}
