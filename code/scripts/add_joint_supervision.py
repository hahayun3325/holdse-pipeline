#!/usr/bin/env python3
"""Add ground truth joint supervision to loss."""

# Check current loss_terms.py structure
print("=== Current loss_terms.py functions ===")
with open('src/hold/loss_terms.py', 'r') as f:
    for line in f:
        if line.strip().startswith('def '):
            print(line.strip())

print("\n=== Suggested addition ===")
print("""
Add this to src/hold/loss_terms.py:

def get_joint_supervision_loss(model_outputs, gt_data, hand_node_name='right'):
    '''Supervise predicted joints with ground truth.
    
    Args:
        model_outputs: Dict with predicted joint positions
        gt_data: Dict with ground truth joint positions  
        hand_node_name: 'right' or 'left'
        
    Returns:
        Joint position loss (L2 distance)
    '''
    import torch
    
    # Get predicted joints from MANO forward pass
    pred_key = f'j3d.{hand_node_name}'
    if pred_key not in model_outputs:
        return torch.tensor(0.0, device=next(iter(model_outputs.values())).device)
    
    pred_joints = model_outputs[pred_key]  # [N, 21, 3]
    
    # Get GT joints
    gt_key = f'j3d.{hand_node_name}'
    if gt_key not in gt_data:
        return torch.tensor(0.0, device=pred_joints.device)
    
    gt_joints = gt_data[gt_key]  # [N, 21, 3]
    
    # L2 loss on joint positions
    loss = torch.nn.functional.mse_loss(pred_joints, gt_joints)
    
    return loss
""")
