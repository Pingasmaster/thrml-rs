use crate::pgm::{Node, NodeKind};
use std::collections::HashMap;

/// Value stored for a node in the state.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum NodeValue {
    Spin(bool),
    Categorical(u8),
}

impl NodeValue {
    /// Returns the kind that produced this value.
    pub fn kind(&self) -> NodeKind {
        match self {
            NodeValue::Spin(_) => NodeKind::Spin,
            NodeValue::Categorical(_) => NodeKind::Categorical,
        }
    }
}

/// State of a block of nodes.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BlockState {
    pub values: Vec<NodeValue>,
}

impl BlockState {
    /// Constructs a block state with a value for each node.
    pub fn new(values: Vec<NodeValue>) -> Self {
        Self { values }
    }

    /// Creates an empty state filled with zeros for a given kind.
    pub fn zeros(kind: NodeKind, len: usize) -> Self {
        let values = match kind {
            NodeKind::Spin => vec![NodeValue::Spin(false); len],
            NodeKind::Categorical => vec![NodeValue::Categorical(0); len],
        };
        BlockState { values }
    }

    /// Returns the number of nodes represented by this state.
    pub fn len(&self) -> usize {
        self.values.len()
    }
}

/// Convenience type alias for the global state representation.
pub type GlobalState = Vec<NodeValue>;

/// Describes a collection of nodes that can be sampled together.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Block {
    nodes: Vec<Node>,
    kind: NodeKind,
}

impl Block {
    /// Creates a new block from nodes. All nodes must share the same `NodeKind`.
    pub fn new(nodes: Vec<Node>) -> Self {
        let kind = nodes.first().expect("Block cannot be empty").kind();
        assert!(
            nodes.iter().all(|node| node.kind() == kind),
            "All nodes in a block must share a type"
        );
        Self { nodes, kind }
    }

    /// Returns the kind of node stored in this block.
    pub fn kind(&self) -> NodeKind {
        self.kind
    }

    /// Returns the number of nodes in this block.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns an iterator over the nodes.
    pub fn iter(&self) -> impl Iterator<Item = &Node> {
        self.nodes.iter()
    }

    /// Returns the node at the provided relative index.
    pub fn get(&self, index: usize) -> Node {
        self.nodes[index]
    }
}

/// Gather the values of a block by indexing into a global state vector.
pub fn gather_block_state(
    block: &Block,
    global_state: &GlobalState,
    spec: &BlockSpec,
) -> Vec<NodeValue> {
    block
        .iter()
        .map(|node| global_state[spec.node_location[node]].clone())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pgm::{CategoricalNode, SpinNode};

    #[test]
    fn block_state_mapping_roundtrip() {
        let nodes = vec![SpinNode::new().into(), SpinNode::new().into()];
        let other_nodes = vec![SpinNode::new().into(), SpinNode::new().into()];
        let block = Block::new(nodes.clone());
        let other = Block::new(other_nodes);
        let spec = BlockSpec::new(vec![block.clone(), other.clone()]);

        let state = vec![
            BlockState::new(vec![NodeValue::Spin(true), NodeValue::Spin(false)]),
            BlockState::new(vec![NodeValue::Spin(true), NodeValue::Spin(true)]),
        ];

        let state_refs: Vec<&BlockState> = state.iter().collect();
        let global = block_state_to_global(&state_refs);
        let recovered = from_global_state(&global, &spec, &[block]);
        assert_eq!(recovered.len(), 1);
        assert_eq!(
            recovered[0].values,
            vec![NodeValue::Spin(true), NodeValue::Spin(false)]
        );
    }

    #[test]
    fn categorical_blocks_can_be_zeros() {
        let nodes = vec![CategoricalNode::new().into(), CategoricalNode::new().into()];
        let block = Block::new(nodes);
        let empty = BlockState::zeros(block.kind(), block.len());
        assert!(matches!(empty.values[0], NodeValue::Categorical(0)));
    }
}

/// Specification that maps between block and global representations.
#[derive(Clone, Debug)]
pub struct BlockSpec {
    pub blocks: Vec<Block>,
    pub node_location: HashMap<Node, usize>,
    pub global_order: Vec<Node>,
}

impl BlockSpec {
    /// Creates a new `BlockSpec` from the provided blocks.
    pub fn new(blocks: Vec<Block>) -> Self {
        let mut node_location = HashMap::new();
        let mut global_order = Vec::new();

        for block in &blocks {
            for node in block.iter() {
                let index = global_order.len();
                if node_location.insert(*node, index).is_some() {
                    panic!("Node referenced by multiple blocks");
                }
                global_order.push(*node);
            }
        }

        Self {
            blocks,
            node_location,
            global_order,
        }
    }
}

/// Converts block-local states into a global state.
pub fn block_state_to_global(states: &[&BlockState]) -> GlobalState {
    states
        .iter()
        .flat_map(|state| state.values.iter().cloned())
        .collect()
}

/// Extracts states for a subset of blocks from a global state.
pub fn from_global_state(
    global: &GlobalState,
    spec: &BlockSpec,
    blocks: &[Block],
) -> Vec<BlockState> {
    blocks
        .iter()
        .map(|block| {
            let values = block
                .iter()
                .map(|node| {
                    let idx = spec
                        .node_location
                        .get(node)
                        .expect("missing node in global state");
                    global[*idx].clone()
                })
                .collect();
            BlockState::new(values)
        })
        .collect()
}

/// Returns the global indices corresponding to the nodes in `block`.
pub fn get_node_locations(block: &Block, spec: &BlockSpec) -> Vec<usize> {
    block
        .iter()
        .map(|node| *spec.node_location.get(node).expect("node missing in spec"))
        .collect()
}

/// Creates zero-initialized block states for each block.
pub fn make_empty_block_state(blocks: &[Block]) -> Vec<BlockState> {
    blocks
        .iter()
        .map(|block| BlockState::zeros(block.kind(), block.len()))
        .collect()
}

/// Verifies that the provided states align with the block specification.
pub fn verify_block_state(blocks: &[Block], states: &[BlockState]) {
    if blocks.len() != states.len() {
        panic!("Number of states does not match number of blocks");
    }

    for (block, state) in blocks.iter().zip(states.iter()) {
        if block.len() != state.len() {
            panic!("Block length mismatch");
        }
        if block.kind()
            != state
                .values
                .first()
                .map(|v| v.kind())
                .unwrap_or(block.kind())
        {
            panic!("Block kind mismatch");
        }
    }
}
