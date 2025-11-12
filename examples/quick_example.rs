use rand::SeedableRng;
use rand::rngs::StdRng;
use std::sync::Arc;
use thrml::SamplingSchedule;
use thrml::block_management::{Block, BlockState, block_state_to_global};
use thrml::block_sampling::{BlockGibbsSpec, BlockSamplingProgram};
use thrml::conditional_samplers::{AbstractConditionalSampler, BernoulliConditional};
use thrml::interaction::{InteractionGroup, SpinBiasEvaluator};
use thrml::pgm::SpinNode;

fn build_chain_nodes(n: usize) -> Vec<SpinNode> {
    (0..n).map(|_| SpinNode::new()).collect()
}

fn main() {
    let nodes = build_chain_nodes(5);
    let node_refs: Vec<_> = nodes.iter().cloned().map(|node| node.into()).collect();

    let even_block = Block::new(
        node_refs
            .iter()
            .enumerate()
            .filter(|(idx, _)| idx % 2 == 0)
            .map(|(_, node)| *node)
            .collect(),
    );
    let odd_block = Block::new(
        node_refs
            .iter()
            .enumerate()
            .filter(|(idx, _)| idx % 2 == 1)
            .map(|(_, node)| *node)
            .collect(),
    );

    let free_super_blocks = vec![vec![even_block.clone()], vec![odd_block.clone()]];
    let spec = BlockGibbsSpec::new(free_super_blocks.clone(), vec![]);

    let even_bias = vec![0.0; even_block.len()];
    let odd_bias = vec![0.0; odd_block.len()];
    let interactions = vec![
        InteractionGroup::new(
            even_block.clone(),
            vec![],
            Box::new(SpinBiasEvaluator::new(
                Arc::new(even_bias),
                even_block.len(),
            )),
        ),
        InteractionGroup::new(
            odd_block.clone(),
            vec![],
            Box::new(SpinBiasEvaluator::new(Arc::new(odd_bias), odd_block.len())),
        ),
    ];

    let samplers: Vec<Box<dyn AbstractConditionalSampler>> = spec
        .free_blocks
        .iter()
        .map(|_| Box::new(BernoulliConditional) as Box<dyn AbstractConditionalSampler>)
        .collect();
    let program = BlockSamplingProgram::new(spec, samplers, interactions);

    let mut rng = StdRng::seed_from_u64(42);
    let mut state_free = vec![
        BlockState::zeros(even_block.kind(), even_block.len()),
        BlockState::zeros(odd_block.kind(), odd_block.len()),
    ];
    let state_clamp: Vec<BlockState> = vec![];

    let schedule = SamplingSchedule {
        n_warmup: 100,
        n_samples: 10,
        steps_per_sample: 2,
    };

    for _ in 0..schedule.n_warmup {
        program.sample_blocks(&mut rng, &mut state_free, &state_clamp);
    }

    let mut samples = Vec::with_capacity(schedule.n_samples);
    for _ in 0..schedule.n_samples {
        for _ in 0..schedule.steps_per_sample {
            program.sample_blocks(&mut rng, &mut state_free, &state_clamp);
        }
        let refs: Vec<&BlockState> = state_free.iter().collect();
        samples.push(block_state_to_global(&refs));
    }

    println!("Collected {} global samples", samples.len());
    if let Some(first) = samples.first() {
        println!("First global state: {:?}", first);
    }
}
