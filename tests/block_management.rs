use thrml::pgm::SpinNode;
use thrml::{Block, BlockSpec, BlockState, NodeValue, block_state_to_global};

#[test]
fn roundtrip_block_state_can_be_consumed() {
    let nodes = vec![SpinNode::new().into(), SpinNode::new().into()];
    let block = Block::new(nodes.clone());
    let state = BlockState::new(vec![NodeValue::Spin(true), NodeValue::Spin(false)]);
    let spec = BlockSpec::new(vec![block.clone()]);

    let global = block_state_to_global(&[&state]);
    let recovered = thrml::block_management::from_global_state(&global, &spec, &[block]);
    assert_eq!(recovered.len(), 1);
    assert_eq!(recovered[0].values, state.values);
}

#[test]
#[should_panic]
fn cannot_duplicate_nodes_in_spec() {
    let node = SpinNode::new().into();
    let block_a = Block::new(vec![node]);
    let block_b = Block::new(vec![node]);
    let _ = BlockSpec::new(vec![block_a, block_b]);
}
