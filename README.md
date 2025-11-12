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

See examples/. There's an implementation of the classic example shown in the python version of thrml and a heavier benchmark so you can better see the speed difference on a CPU between this and the original thrml.

Both benchmark are completely equivalent.

## Apple-to-apple comparisons

- **Quick example (Rust vs Python)** keeps the same 5-spin Ising chain, two-color Gibbs blocks, and first-sample inspection. Run the Rust side with `cargo run --example quick_example` and the Python side from the legacy repository via `python run_readme_example.py`.
- **Heavy benchmark (Rust vs Python)** now matches exactly in scale and schedule: 16000 spins, 12500 warmup sweeps, 10000 samples, and 15 steps per sample. This was made specifically to have a heavy benchmark to see the speed difference between the old python and the new rust implementation. See below how to run.

Heavy benchmark:

Run `cargo build --release` first to build the latest version of this rust crate.

1. Run the Rust heavy example: `time cargo run --release --example heavy_example`
2. Run the python version:

```bash
tee heavy.py <<'EOF'
import time
import jax
import jax.numpy as jnp
from thrml import Block, SamplingSchedule, SpinNode, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init


def main() -> None:
    node_count = 16000
    nodes = [SpinNode() for _ in range(node_count)]
    edges = []
    biases = jnp.full((node_count,), 0.05)
    beta = jnp.array(1.0)
    model = IsingEBM(nodes, edges, biases, [], beta)

    free_blocks = [Block(nodes[::2]), Block(nodes[1::2])]
    program = IsingSamplingProgram(model, free_blocks, clamped_blocks=[])

    key = jax.random.key(123)
    k_init, k_samp = jax.random.split(key, 2)
    init_state = hinton_init(k_init, model, free_blocks, ())
    schedule = SamplingSchedule(n_warmup=12500, n_samples=10000, steps_per_sample=15)

    start = time.perf_counter()
    samples = sample_states(k_samp, program, schedule, init_state, [], [Block(nodes)])
    elapsed = time.perf_counter() - start

    sample_tensor = samples[0]
    print("Heavy Python run: 16000 spins, 10000 samples, 15 steps/sample")
    print(f"Sample tensor shape: {sample_tensor.shape}")
    print(f"Elapsed wall-clock: {elapsed:.4f}s")


if __name__ == "__main__":
    main()
EOF
```

Run the python version from the original repo:

```
time python heavy.py
```

3. Compare the printed wall-clock times (real times) for a direct apples-to-apples runtime comparison.

## Tests

Run `cargo test` to execute the Rust unit and integration tests.
