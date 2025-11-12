# THRML (Rust)

THRML is now a Rust-native crate for building and sampling probabilistic graphical models.
It keeps the spirit of the original library (blocked Gibbs sampling, discrete EBMs, Ising models)
while leveraging Rust's safety and performance.

## Key modules

- `block_management`: define nodes, blocks, and conversions between block-local and global states.
- `block_sampling`: build Gibbs sampling programs with per-block samplers.
- `conditional_samplers`: Bernoulli and softmax conditionals for spin and categorical nodes.
- `interaction`/`factor`: define static interaction structures and assemble them into sampling programs.
- `models::ising`: spin-based Ising models plus a simple sampling harness.

## Installation

```bash
cargo install --path .
```

Or add `thrml = { path = "." }` to your `Cargo.toml` dependencies and import the crate with `use thrml::...`.

## Example

```rust
use rand::SeedableRng;
use rand::rngs::StdRng;
use thrml::block_management::{Block, BlockState};
use thrml::models::ising::{IsingEBM, IsingSamplingProgram, hinton_init};
use thrml::pgm::SpinNode;

let nodes = vec![SpinNode::new(), SpinNode::new()];
let ebm = IsingEBM::new(
    nodes.iter().map(|n| (*n).into()).collect(),
    vec![],
    vec![0.0, 0.0],
    vec![],
    1.0,
);
let free_blocks = vec![vec![Block::new(vec![nodes[0].into(), nodes[1].into()])]];
let program = IsingSamplingProgram::new(&ebm, free_blocks, vec![]);
let mut rng = StdRng::seed_from_u64(42);
let mut state_free = hinton_init(&mut rng, &ebm, &[Block::new(vec![nodes[0].into(), nodes[1].into()])]);
program.sample_blocks(&mut rng, &mut state_free, &[]);
assert_eq!(state_free.len(), 1);
```

## Tests

Run `cargo test` to execute the Rust unit and integration tests.
