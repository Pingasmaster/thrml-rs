use rand::SeedableRng;
use rand::rngs::StdRng;
use std::sync::Arc;
use thrml::NodeValue;
use thrml::block_management::{Block, BlockState};
use thrml::block_sampling::{BlockGibbsSpec, BlockSamplingProgram};
use thrml::conditional_samplers::BernoulliConditional;
use thrml::interaction::{InteractionGroup, SpinBiasEvaluator};
use thrml::pgm::SpinNode;

#[test]
// Ensure block sampling can run a sweep and produce valid spin nodes.
fn sampling_program_runs_without_error() {
    let node = SpinNode::new().into();
    let block = Block::new(vec![node]);
    let spec = BlockGibbsSpec::new(vec![vec![block.clone()]], vec![]);
    let interaction = InteractionGroup::new(
        block.clone(),
        vec![],
        Box::new(SpinBiasEvaluator::new(Arc::new(vec![1.0]), block.len())),
    );
    let program = BlockSamplingProgram::new(
        spec,
        vec![Box::new(BernoulliConditional)],
        vec![interaction],
    );

    let mut rng = StdRng::seed_from_u64(7);
    let mut state_free = vec![BlockState::zeros(block.kind(), block.len())];
    let state_clamped = vec![];

    program.sample_blocks(&mut rng, &mut state_free, &state_clamped);
    assert_eq!(state_free[0].len(), block.len());
    assert!(matches!(state_free[0].values[0], NodeValue::Spin(_)));
}
