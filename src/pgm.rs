use std::sync::atomic::{AtomicU64, Ordering};

static NODE_COUNTER: AtomicU64 = AtomicU64::new(0);

/// The kind of a node in a graphical model.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum NodeKind {
    /// Binary spin-valued node.
    Spin,
    /// Categorical node.
    Categorical,
}

/// A lightweight reference to an individual node.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct Node {
    id: u64,
    kind: NodeKind,
}

impl Node {
    fn new(kind: NodeKind) -> Self {
        let id = NODE_COUNTER.fetch_add(1, Ordering::SeqCst);
        Self { id, kind }
    }

    /// Returns the global identifier for the node.
    pub fn id(self) -> u64 {
        self.id
    }

    /// Returns the kind of the node.
    pub fn kind(self) -> NodeKind {
        self.kind
    }
}

impl From<Node> for NodeKind {
    fn from(node: Node) -> Self {
        node.kind()
    }
}

/// A spin-valued node reference.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct SpinNode(Node);

impl SpinNode {
    /// Creates a new spin node.
    pub fn new() -> Self {
        SpinNode(Node::new(NodeKind::Spin))
    }
}

impl From<SpinNode> for Node {
    fn from(spin: SpinNode) -> Self {
        spin.0
    }
}

/// A categorical node reference.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct CategoricalNode(Node);

impl CategoricalNode {
    /// Creates a new categorical node.
    pub fn new() -> Self {
        CategoricalNode(Node::new(NodeKind::Categorical))
    }
}

impl From<CategoricalNode> for Node {
    fn from(cat: CategoricalNode) -> Self {
        cat.0
    }
}
