use crate::block_management::{Block, NodeValue};
use crate::conditional_samplers::{AbstractConditionalSampler, BernoulliConditional};
use crate::factor::AbstractFactor;
use crate::interaction::{
    CategoricalPairwiseEvaluator, InteractionGroup, SpinBiasEvaluator, SpinPairwiseEvaluator,
};
use rand::RngCore;
use std::sync::Arc;

/// A discrete EBM factor that mixes spin and categorical interactions.
pub struct DiscreteEBMFactor {
    spin_factor: Option<SpinEBMFactor>,
    cat_factor: Option<CategoricalEBMFactor>,
}

impl DiscreteEBMFactor {
    pub fn new(
        spin_node_groups: Vec<Block>,
        cat_node_groups: Vec<Block>,
        spin_weights: Vec<f64>,
        cat_weights: Vec<f64>,
        cat_cardinality: usize,
    ) -> Self {
        let spin_factor = if spin_node_groups.is_empty() {
            None
        } else {
            // Build spin factor only if nodes were explicitly provided.
            Some(SpinEBMFactor::new(spin_node_groups, spin_weights))
        };
        let cat_factor = if cat_node_groups.is_empty() {
            None
        } else {
            // Build categorical factor only when there are categorical node groups.
            Some(CategoricalEBMFactor::new(
                cat_node_groups,
                cat_weights,
                cat_cardinality,
            ))
        };
        Self {
            spin_factor,
            cat_factor,
        }
    }
}
/// Factor covering spin-only interactions (bias or pairwise).
pub struct SpinEBMFactor {
    node_groups: Vec<Block>,
    weights: Vec<f64>,
}

impl SpinEBMFactor {
    pub fn new(node_groups: Vec<Block>, weights: Vec<f64>) -> Self {
        if node_groups.is_empty() {
            panic!("SpinEBMFactor requires at least one node group");
        }
        // Node groups drive either bias-only or pairwise interactions depending on count.
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
        // One group -> bias-only, two -> pairwise head/tail interaction.
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

/// Gibbs sampler for spin blocks built on top of a Bernoulli conditional.
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

/// Factor covering categorical head/tail interactions using logits per category.
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
        // Each head entry expects one logit per category for conditional sampling.
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
        // Use categorical evaluator that sees full logits for each head/tail pair.
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

#[cfg(test)]
mod discrete_tests {
    use super::*;
    use crate::block_management::Block;
    use crate::pgm::{CategoricalNode, SpinNode};

    #[test]
    fn discrete_factor_produces_spin_and_cat_groups() {
        let spin_nodes = vec![SpinNode::new().into(), SpinNode::new().into()];
        let cat_nodes = vec![CategoricalNode::new().into(), CategoricalNode::new().into()];
        let tail_cat_nodes = vec![CategoricalNode::new().into(), CategoricalNode::new().into()];
        let spin_block = Block::new(spin_nodes.clone());
        let cat_block = Block::new(cat_nodes.clone());
        let cat_tail_block = Block::new(tail_cat_nodes.clone());

        let factor = DiscreteEBMFactor::new(
            vec![spin_block.clone()],
            vec![cat_block.clone(), cat_tail_block.clone()],
            vec![0.1, -0.1],
            vec![0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            3,
        );

        let groups = factor.interaction_groups();
        assert_eq!(groups.len(), 2);
    }
}

impl AbstractFactor for DiscreteEBMFactor {
    fn interaction_groups(&self) -> Vec<InteractionGroup> {
        let mut groups = Vec::new();
        if let Some(factor) = &self.spin_factor {
            groups.extend(factor.interaction_groups());
        }
        if let Some(factor) = &self.cat_factor {
            groups.extend(factor.interaction_groups());
        }
        groups
    }
}
