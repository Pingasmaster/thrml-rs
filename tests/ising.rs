use rand::SeedableRng;
use rand::rngs::StdRng;
use thrml::block_management::Block;
use thrml::block_management::BlockState;
use thrml::models::ising::{IsingEBM, IsingSamplingProgram, hinton_init};
use thrml::pgm::SpinNode;

#[test]
fn ising_sampling_can_run() {
    let nodes = vec![SpinNode::new(), SpinNode::new()];
    let edges = vec![];
    let biases = vec![0.0, 0.0];
    let weights = vec![];
    let ebm = IsingEBM::new(
        nodes.iter().map(|node| (*node).into()).collect(),
        edges,
        biases,
        weights,
        1.0,
    );

    let block = Block::new(vec![nodes[0].into(), nodes[1].into()]);
    let free_blocks = vec![vec![block.clone()]];

    let program = IsingSamplingProgram::new(&ebm, free_blocks, vec![]);

    let mut rng = StdRng::seed_from_u64(1);
    let mut state_free = hinton_init(&mut rng, &ebm, &[block.clone()]);
    let state_clamp: Vec<BlockState> = vec![];

    program.sample_blocks(&mut rng, &mut state_free, &state_clamp);
    assert_eq!(state_free.len(), 1);
}
