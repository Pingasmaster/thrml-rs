# THRML (Rust)

THRML is now a Rust-native crate for building and sampling probabilistic graphical models.
It keeps the spirit of the original library (blocked Gibbs sampling, discrete EBMs, Ising models)
while leveraging Rust's safety and performance.

> **Disclaimer:** This crate is an independent reimplementation of the original THRML project and
> is not affiliated with or maintained by the original authors or company. Please do not contact
> them about issues arising with this Rust port. This work was done for fun and experimentation;
> there is no promise of professional support, and future maintenance will only happen if there is
> significant community interest.

## Migration notes

- The Python `thrml` module, its JAX/EQX dependencies, and MkDocs site were replaced with a single
  Rust crate. The new crate exposes Blocks → Interactions → Factors → SamplingPrograms workflows
  while providing explicit state management and RNG-driven samplers.
- All Python tests and notebooks were superseded by Rust unit/integration tests (`tests/*.rs` and
  `src/...::tests`). The README example now targets the Rust API directly.
- Observers now follow a carry/output trait, and discrete EBMs combine spin/categorical evaluators built
  on top of ndarray-backed weight tensors.

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
