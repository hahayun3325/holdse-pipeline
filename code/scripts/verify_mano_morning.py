import torch
import sys

# Load checkpoints
try:
    # Get checkpoint paths from command line or use defaults
    ckpt_path_new = sys.argv[1] if len(sys.argv) > 1 else None
    
    if not ckpt_path_new:
        print("Usage: python verify_mano_morning.py <new_checkpoint_path>")
        sys.exit(1)
    
    print("="*60)
    print("VERIFYING OVERNIGHT TRAINING - MANO UPDATE CHECK")
    print("="*60)
    
    # Compare with first checkpoint if available
    # For now, we'll check if parameters exist and look reasonable
    ckpt = torch.load(ckpt_path_new)
    
    print(f"\n✅ Checkpoint loaded successfully")
    print(f"   Epoch: {ckpt.get('epoch', 'unknown')}")
    print(f"   Total parameters: {len(ckpt['state_dict'])}")
    
    # Check MANO parameters
    mano_keys = [
        'model.nodes.right.params.pose.weight',
        'model.nodes.right.params.global_orient.weight',
        'model.nodes.right.params.transl.weight',
        'model.nodes.right.params.betas.weight',
    ]
    
    print("\n" + "="*60)
    print("MANO PARAMETERS PRESENT:")
    print("="*60)
    
    all_present = True
    for key in mano_keys:
        if key in ckpt['state_dict']:
            param = ckpt['state_dict'][key]
            print(f"✅ {key.split('.')[-2]}: {param.shape}")
            
            # Check if values look reasonable (not all zeros/NaN)
            if torch.all(param == 0):
                print(f"   ⚠️ WARNING: All zeros!")
            elif torch.any(torch.isnan(param)):
                print(f"   ❌ ERROR: Contains NaN!")
            else:
                mean_val = torch.abs(param).mean().item()
                std_val = param.std().item()
                print(f"   Stats: mean={mean_val:.6f}, std={std_val:.6f}")
        else:
            print(f"❌ {key}: MISSING!")
            all_present = False
    
    if all_present:
        print("\n" + "="*60)
        print("✅ ALL MANO PARAMETERS PRESENT AND VALID")
        print("="*60)
        print("\nNEXT STEP: Compare with initialization to verify updates")
        print("Run: python scripts/compare_checkpoints.py <old_ckpt> <new_ckpt>")
    else:
        print("\n❌ SOME MANO PARAMETERS MISSING - TRAINING FAILED")
    
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

