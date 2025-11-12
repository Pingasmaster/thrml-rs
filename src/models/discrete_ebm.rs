use crate::block_management::{Block, NodeValue};
use crate::conditional_samplers::{AbstractConditionalSampler, BernoulliConditional};
use crate::factor::AbstractFactor;
use crate::interaction::{
    CategoricalPairwiseEvaluator, InteractionGroup, SpinBiasEvaluator, SpinPairwiseEvaluator,
};
use rand::RngCore;
use std::sync::Arc;

pub struct SpinEBMFactor {
    node_groups: Vec<Block>,
    weights: Vec<f64>,
}

impl SpinEBMFactor {
    pub fn new(node_groups: Vec<Block>, weights: Vec<f64>) -> Self {
        if node_groups.is_empty() {
            panic!("SpinEBMFactor requires at least one node group");
        }
        Self {
            node_groups,
            weights,
        }
    }
}

impl AbstractFactor for SpinEBMFactor {
    fn interaction_groups(&self) -> Vec<InteractionGroup> {
        let head = self.node_groups[0].clone();
        let head_len = head.len();
        match self.node_groups.len() {
            1 => {
                if self.weights.len() != head_len {
                    panic!("Spin bias weights length must match head size")
                }
                vec![InteractionGroup::new(
                    head,
                    vec![],
                    Box::new(SpinBiasEvaluator::new(
                        Arc::new(self.weights.clone()),
                        head_len,
                    )),
                )]
            }
            2 => {
                if self.weights.len() != head_len * 2 {
                    panic!("Spin pairwise weights must be head_len * 2")
                }
                vec![InteractionGroup::new(
                    head,
                    vec![self.node_groups[1].clone()],
                    Box::new(SpinPairwiseEvaluator::new(
                        Arc::new(self.weights.clone()),
                        head_len,
                    )),
                )]
            }
            _ => panic!("SpinEBMFactor supports at most two node groups"),
        }
    }
}

pub struct SpinGibbsConditional(BernoulliConditional);

impl SpinGibbsConditional {
    pub fn new() -> Self {
        Self(BernoulliConditional)
    }
}

impl AbstractConditionalSampler for SpinGibbsConditional {
    fn sample(&self, rng: &mut dyn RngCore, logits: &[f64]) -> Vec<NodeValue> {
        self.0.sample(rng, logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BlockSpec;
    use crate::block_management::{Block, BlockState, block_state_to_global};
    use crate::pgm::CategoricalNode;

    #[test]
    fn categorical_factor_evaluates_indices() {
        let head_nodes = vec![CategoricalNode::new().into(), CategoricalNode::new().into()];
        let tail_nodes = vec![CategoricalNode::new().into(), CategoricalNode::new().into()];
        let head = Block::new(head_nodes.clone());
        let tail = Block::new(tail_nodes.clone());

        let cardinality = 3;
        let weights = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let factor = CategoricalEBMFactor::new(
            vec![head.clone(), tail.clone()],
            weights.clone(),
            cardinality,
        );
        let groups = factor.interaction_groups();

        let head_state =
            BlockState::new(vec![NodeValue::Categorical(0), NodeValue::Categorical(1)]);
        let tail_state =
            BlockState::new(vec![NodeValue::Categorical(2), NodeValue::Categorical(1)]);
        let spec = BlockSpec::new(vec![head.clone(), tail.clone()]);
        let global = block_state_to_global(&[&head_state, &tail_state]);

        let result = groups[0].evaluate(&global, &spec);
        assert_eq!(result[0], 0.3);
        assert_eq!(result[1], 0.5);
    }
}

pub struct CategoricalEBMFactor {
    node_groups: Vec<Block>,
    weights: Vec<f64>,
    cardinality: usize,
}

impl CategoricalEBMFactor {
    pub fn new(node_groups: Vec<Block>, weights: Vec<f64>, cardinality: usize) -> Self {
        if node_groups.len() != 2 {
            panic!("CategoricalEBMFactor currently supports exactly two node groups");
        }
        Self {
            node_groups,
            weights,
            cardinality,
        }
    }
}

impl AbstractFactor for CategoricalEBMFactor {
    fn interaction_groups(&self) -> Vec<InteractionGroup> {
        let head = self.node_groups[0].clone();
        let head_len = head.len();
        if self.weights.len() != head_len * self.cardinality {
            panic!("Categorical factor weights must equal head_len * cardinality")
        }
        vec![InteractionGroup::new(
            head,
            vec![self.node_groups[1].clone()],
            Box::new(CategoricalPairwiseEvaluator::new(
                Arc::new(self.weights.clone()),
                head_len,
                self.cardinality,
            )),
        )]
    }
}
