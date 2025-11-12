use rand::SeedableRng;
use rand::rngs::StdRng;
use std::time::Instant;
use thrml::SamplingSchedule;
use thrml::block_management::{Block, BlockState, block_state_to_global};
use thrml::models::ising::{IsingEBM, IsingSamplingProgram, hinton_init};
use thrml::pgm::SpinNode;

fn main() {
    let num_spins = 64;
    let beta = 0.75;
    let node_refs = (0..num_spins)
        .map(|_| SpinNode::new().into())
        .collect::<Vec<_>>();
    let full_block = Block::new(node_refs.clone());

    let ebm = IsingEBM::new(
        node_refs.clone(),
        vec![],
        vec![0.05; node_refs.len()],
        vec![],
        beta,
    );

    let free_super_blocks = vec![vec![full_block.clone()]];
    let mut rng = StdRng::seed_from_u64(7);
    let program = IsingSamplingProgram::new(&ebm, free_super_blocks.clone(), vec![]);

    let init_blocks: Vec<Block> = free_super_blocks
        .iter()
        .flat_map(|group| group.clone())
        .collect();
    let mut state_free = hinton_init(&mut rng, &ebm, &init_blocks);
    let state_clamp: Vec<BlockState> = vec![];

    let schedule = SamplingSchedule {
        n_warmup: 500,
        n_samples: 200,
        steps_per_sample: 4,
    };

    let start = Instant::now();
    for _ in 0..schedule.n_warmup {
        program.sample_blocks(&mut rng, &mut state_free, &state_clamp);
    }

    for _ in 0..schedule.n_samples {
        for _ in 0..schedule.steps_per_sample {
            program.sample_blocks(&mut rng, &mut state_free, &state_clamp);
        }
    }
    let elapsed = start.elapsed();

    let refs: Vec<&BlockState> = state_free.iter().collect();
    let final_state = block_state_to_global(&refs);
    let snippet: Vec<_> = final_state.iter().take(16).cloned().collect();

    println!(
        "Heavy Rust run finished: {} spins, {} samples, {} steps/sample in {:?}",
        num_spins, schedule.n_samples, schedule.steps_per_sample, elapsed,
    );
    println!("Sample snippet (first 16 nodes): {:?}", snippet);
}
