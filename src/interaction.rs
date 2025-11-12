use crate::block_management::{Block, BlockSpec, GlobalState, NodeValue, gather_block_state};
use std::sync::Arc;

/// Trait for evaluation logic inside an interaction group.
pub trait InteractionEvaluator: Send + Sync {
    fn evaluate(
        &self,
        global_state: &GlobalState,
        spec: &BlockSpec,
        tail_nodes: &[Block],
    ) -> Vec<f64>;
}

fn value_to_index(value: &NodeValue, dim: usize) -> usize {
    match value {
        NodeValue::Spin(true) => 1,
        NodeValue::Spin(false) => 0,
        NodeValue::Categorical(v) => {
            let idx = *v as usize;
            if idx >= dim {
                panic!("Categorical value {} >= dimension {}", idx, dim);
            }
            idx
        }
    }
}

pub struct TensorEvaluator {
    weights: Arc<Vec<f64>>,
    head_len: usize,
    tail_dims: Vec<usize>,
    tail_stride: usize,
}

impl TensorEvaluator {
    pub fn new(weights: Arc<Vec<f64>>, head_len: usize, tail_dims: Vec<usize>) -> Self {
        let tail_stride = tail_dims
            .iter()
            .copied()
            .fold(1, |acc, dim| acc * dim_or_one(dim));
        if weights.len() != head_len * tail_stride {
            panic!(
                "Weights length ({}) must equal head_len ({}) * tail_stride ({})",
                weights.len(),
                head_len,
                tail_stride
            );
        }
        Self {
            weights,
            head_len,
            tail_dims,
            tail_stride,
        }
    }
}

impl InteractionEvaluator for TensorEvaluator {
    fn evaluate(
        &self,
        global_state: &GlobalState,
        spec: &BlockSpec,
        tail_nodes: &[Block],
    ) -> Vec<f64> {
        if tail_nodes.len() != self.tail_dims.len() {
            panic!(
                "Tail block count mismatch {} vs {}",
                tail_nodes.len(),
                self.tail_dims.len()
            );
        }

        let tail_states: Vec<Vec<NodeValue>> = tail_nodes
            .iter()
            .map(|block| gather_block_state(block, global_state, spec))
            .collect();

        let mut out = vec![0.0; self.head_len];
        for i in 0..self.head_len {
            let mut tail_index = 0;
            let mut stride = 1;
            for (dim, state_batch) in self.tail_dims.iter().zip(tail_states.iter()) {
                let value = &state_batch[i];
                let idx = value_to_index(value, *dim);
                tail_index += idx * stride;
                stride *= dim_or_one(*dim);
            }
            out[i] = self.weights[i * self.tail_stride + tail_index];
        }
        out
    }
}

fn dim_or_one(dim: usize) -> usize {
    if dim == 0 { 1 } else { dim }
}

pub struct InteractionGroup {
    pub head_nodes: Block,
    pub tail_nodes: Vec<Block>,
    evaluator: Box<dyn InteractionEvaluator>,
}

impl InteractionGroup {
    pub fn new(
        head_nodes: Block,
        tail_nodes: Vec<Block>,
        evaluator: Box<dyn InteractionEvaluator>,
    ) -> Self {
        let head_len = head_nodes.len();
        for tail in &tail_nodes {
            if tail.len() != head_len {
                panic!("All tail node blocks must have the same length as head_nodes");
            }
        }
        Self {
            head_nodes,
            tail_nodes,
            evaluator,
        }
    }

    pub fn evaluate(&self, global_state: &GlobalState, spec: &BlockSpec) -> Vec<f64> {
        self.evaluator
            .evaluate(global_state, spec, &self.tail_nodes)
    }
}

pub struct SpinBiasEvaluator {
    inner: TensorEvaluator,
}

impl SpinBiasEvaluator {
    pub fn new(weights: Arc<Vec<f64>>, head_len: usize) -> Self {
        Self {
            inner: TensorEvaluator::new(weights, head_len, vec![]),
        }
    }
}

impl InteractionEvaluator for SpinBiasEvaluator {
    fn evaluate(
        &self,
        global_state: &GlobalState,
        spec: &BlockSpec,
        tail_nodes: &[Block],
    ) -> Vec<f64> {
        self.inner.evaluate(global_state, spec, tail_nodes)
    }
}

pub struct SpinPairwiseEvaluator {
    inner: TensorEvaluator,
}

impl SpinPairwiseEvaluator {
    pub fn new(weights: Arc<Vec<f64>>, head_len: usize) -> Self {
        Self {
            inner: TensorEvaluator::new(weights, head_len, vec![2]),
        }
    }
}

impl InteractionEvaluator for SpinPairwiseEvaluator {
    fn evaluate(
        &self,
        global_state: &GlobalState,
        spec: &BlockSpec,
        tail_nodes: &[Block],
    ) -> Vec<f64> {
        self.inner.evaluate(global_state, spec, tail_nodes)
    }
}

pub struct CategoricalPairwiseEvaluator {
    inner: TensorEvaluator,
}

impl CategoricalPairwiseEvaluator {
    pub fn new(weights: Arc<Vec<f64>>, head_len: usize, cardinality: usize) -> Self {
        Self {
            inner: TensorEvaluator::new(weights, head_len, vec![cardinality]),
        }
    }
}

impl InteractionEvaluator for CategoricalPairwiseEvaluator {
    fn evaluate(
        &self,
        global_state: &GlobalState,
        spec: &BlockSpec,
        tail_nodes: &[Block],
    ) -> Vec<f64> {
        self.inner.evaluate(global_state, spec, tail_nodes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_management::{Block, BlockState, block_state_to_global};
    use crate::pgm::SpinNode;

    #[test]
    fn tensor_evaluator_handles_categorical() {
        let head = Block::new(vec![SpinNode::new().into(), SpinNode::new().into()]);
        let tail = Block::new(vec![SpinNode::new().into(), SpinNode::new().into()]);
        let weights = Arc::new(vec![0.1, 0.2, 0.3, 0.4]);
        let interaction = InteractionGroup::new(
            head.clone(),
            vec![tail.clone()],
            Box::new(CategoricalPairwiseEvaluator::new(
                weights.clone(),
                head.len(),
                2,
            )),
        );

        let head_state = BlockState::zeros(head.kind(), head.len());
        let tail_state =
            BlockState::new(vec![NodeValue::Categorical(1), NodeValue::Categorical(0)]);
        let spec = BlockSpec::new(vec![head.clone(), tail.clone()]);
        let global = block_state_to_global(&[&head_state, &tail_state]);

        let result = interaction.evaluate(&global, &spec);
        assert_eq!(result[0], 0.2);
        assert_eq!(result[1], 0.3);
    }
}
