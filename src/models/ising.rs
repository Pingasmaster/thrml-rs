use crate::block_management::{Block, BlockState, GlobalState, NodeValue, block_state_to_global};
use crate::block_sampling::{BlockGibbsSpec, SamplingSchedule};
use crate::conditional_samplers::AbstractConditionalSampler;
use crate::factor::FactorSamplingProgram;
use crate::models::discrete_ebm::{SpinEBMFactor, SpinGibbsConditional};
use rand::{Rng, RngCore};
use std::collections::HashMap;

fn run_sampling_chain(
    rng: &mut dyn RngCore,
    program: &IsingSamplingProgram,
    schedule: &SamplingSchedule,
    state_free: &mut [BlockState],
    state_clamp: &[BlockState],
) -> Vec<GlobalState> {
    for _ in 0..schedule.n_warmup {
        program.sample_blocks(rng, state_free, state_clamp);
    }

    let mut samples = Vec::with_capacity(schedule.n_samples);
    for _ in 0..schedule.n_samples {
        for _ in 0..schedule.steps_per_sample {
            program.sample_blocks(rng, state_free, state_clamp);
        }
        samples.push(global_state_for(state_free, state_clamp));
    }
    samples
}

pub type Edge = (crate::pgm::Node, crate::pgm::Node);

pub struct IsingEBM {
    pub nodes: Vec<crate::pgm::Node>,
    pub edges: Vec<Edge>,
    pub biases: Vec<f64>,
    pub weights: Vec<f64>,
    pub beta: f64,
    node_index: HashMap<crate::pgm::Node, usize>,
}

impl IsingEBM {
    pub fn new(
        nodes: Vec<crate::pgm::Node>,
        edges: Vec<Edge>,
        biases: Vec<f64>,
        weights: Vec<f64>,
        beta: f64,
    ) -> Self {
        if biases.len() != nodes.len() {
            panic!("Bias vector length must match node count");
        }
        if weights.len() != edges.len() {
            panic!("Weights vector length must match edge count");
        }
        let node_index = nodes
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, node)| (node, i))
            .collect();
        Self {
            nodes,
            edges,
            biases,
            weights,
            beta,
            node_index,
        }
    }

    pub fn bias_factor(&self) -> SpinEBMFactor {
        SpinEBMFactor::new(vec![Block::new(self.nodes.clone())], self.biases.clone())
    }

    pub fn edge_factor(&self) -> SpinEBMFactor {
        let head = Block::new(self.edges.iter().map(|(a, _)| *a).collect());
        let tail = Block::new(self.edges.iter().map(|(_, b)| *b).collect());
        SpinEBMFactor::new(vec![head, tail], self.weights.clone())
    }

    pub fn to_factors(&self) -> Vec<Box<dyn crate::factor::AbstractFactor>> {
        let mut result =
            vec![Box::new(self.bias_factor()) as Box<dyn crate::factor::AbstractFactor>];
        if !self.edges.is_empty() {
            result.push(Box::new(self.edge_factor()));
        }
        result
    }

    pub fn energy(&self, global_state: &GlobalState) -> f64 {
        let spins: Vec<f64> = self
            .nodes
            .iter()
            .map(|node| match global_state[self.node_index[node]] {
                NodeValue::Spin(true) => 1.0,
                NodeValue::Spin(false) => -1.0,
                NodeValue::Categorical(_) => panic!("Expected spin node"),
            })
            .collect();

        let bias_energy: f64 = self
            .biases
            .iter()
            .zip(spins.iter())
            .map(|(b, s)| b * s)
            .sum();
        let edge_energy: f64 = self
            .weights
            .iter()
            .zip(self.edges.iter())
            .map(|(w, (a, b))| {
                let ia = self.node_index[a];
                let ib = self.node_index[b];
                w * spins[ia] * spins[ib]
            })
            .sum();
        -self.beta * (bias_energy + edge_energy)
    }
}

pub struct IsingSamplingProgram {
    pub inner: FactorSamplingProgram,
}

impl IsingSamplingProgram {
    pub fn new(ebm: &IsingEBM, free_blocks: Vec<Vec<Block>>, clamped_blocks: Vec<Block>) -> Self {
        let spec = BlockGibbsSpec::new(free_blocks, clamped_blocks);
        let samplers: Vec<Box<dyn AbstractConditionalSampler>> = spec
            .free_blocks
            .iter()
            .map(|_| Box::new(SpinGibbsConditional::new()) as Box<dyn AbstractConditionalSampler>)
            .collect();
        let factors = ebm.to_factors();
        let program = FactorSamplingProgram::new(spec, samplers, factors);
        Self { inner: program }
    }

    pub fn sample_blocks(
        &self,
        rng: &mut dyn RngCore,
        state_free: &mut [BlockState],
        state_clamp: &[BlockState],
    ) {
        self.inner.sample_blocks(rng, state_free, state_clamp);
    }
}

pub struct IsingTrainingSpec {
    pub ebm: IsingEBM,
    pub program_positive: IsingSamplingProgram,
    pub program_negative: IsingSamplingProgram,
    pub schedule_positive: SamplingSchedule,
    pub schedule_negative: SamplingSchedule,
}

pub fn hinton_init(rng: &mut dyn RngCore, model: &IsingEBM, blocks: &[Block]) -> Vec<BlockState> {
    blocks
        .iter()
        .map(|block| {
            let values = block
                .iter()
                .map(|node| {
                    let bias = model.biases[model.node_index[node]];
                    let p = 1.0 / (1.0 + (-model.beta * bias).exp());
                    let draw: f64 = rng.random();
                    NodeValue::Spin(draw < p)
                })
                .collect();
            BlockState::new(values)
        })
        .collect()
}

pub fn estimate_moments(
    _rng: &mut dyn RngCore,
    _first_moment_nodes: &[crate::pgm::Node],
    _second_moment_edges: &[Edge],
    _program: &IsingSamplingProgram,
    _schedule: &SamplingSchedule,
    _init_state: &[BlockState],
    _clamped_data: &[BlockState],
) -> (Vec<f64>, Vec<f64>) {
    let mut state = _init_state.to_vec();
    let clamp = _clamped_data;
    let samples = run_sampling_chain(_rng, _program, _schedule, &mut state, clamp);

    let spec = &_program.inner.inner.gibbs_spec.block_spec;

    let mut node_sums = vec![0.0; _first_moment_nodes.len()];
    let mut edge_sums = vec![0.0; _second_moment_edges.len()];

    for global in samples.iter() {
        for (i, node) in _first_moment_nodes.iter().enumerate() {
            let value = spin_value(global[spec.node_location[node]].clone());
            node_sums[i] += value;
        }

        for (i, (a, b)) in _second_moment_edges.iter().enumerate() {
            let va = spin_value(global[spec.node_location[a]].clone());
            let vb = spin_value(global[spec.node_location[b]].clone());
            edge_sums[i] += va * vb;
        }
    }

    let denom = _schedule.n_samples as f64;
    (
        node_sums.into_iter().map(|sum| sum / denom).collect(),
        edge_sums.into_iter().map(|sum| sum / denom).collect(),
    )
}

pub fn estimate_kl_grad(
    _rng: &mut dyn RngCore,
    _spec: &IsingTrainingSpec,
    _bias_nodes: &[crate::pgm::Node],
    _weight_edges: &[Edge],
    _data: &[BlockState],
    _conditioning_values: &[BlockState],
    _init_state_positive: &[BlockState],
    _init_state_negative: &[BlockState],
) -> (
    Vec<f64>,
    Vec<f64>,
    (Vec<f64>, Vec<f64>),
    (Vec<f64>, Vec<f64>),
) {
    let (pos_nodes, pos_edges) = estimate_moments(
        _rng,
        _bias_nodes,
        _weight_edges,
        &_spec.program_positive,
        &_spec.schedule_positive,
        _init_state_positive,
        _data,
    );
    let (neg_nodes, neg_edges) = estimate_moments(
        _rng,
        _bias_nodes,
        _weight_edges,
        &_spec.program_negative,
        &_spec.schedule_negative,
        _init_state_negative,
        _conditioning_values,
    );

    let beta = _spec.ebm.beta;
    let grad_b = pos_nodes
        .iter()
        .zip(neg_nodes.iter())
        .map(|(p, n)| -beta * (p - n))
        .collect();
    let grad_w = pos_edges
        .iter()
        .zip(neg_edges.iter())
        .map(|(p, n)| -beta * (p - n))
        .collect();

    (
        grad_w,
        grad_b,
        (pos_nodes, pos_edges),
        (neg_nodes, neg_edges),
    )
}

fn global_state_for(state_free: &[BlockState], state_clamp: &[BlockState]) -> GlobalState {
    let refs: Vec<&BlockState> = state_free.iter().chain(state_clamp.iter()).collect();
    block_state_to_global(&refs)
}

fn spin_value(value: NodeValue) -> f64 {
    match value {
        NodeValue::Spin(true) => 1.0,
        NodeValue::Spin(false) => -1.0,
        NodeValue::Categorical(_) => panic!("Expected spin value"),
    }
}
