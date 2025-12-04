import torch
import sys
sys.path.insert(0, '/home/fredcui/Projects/holdse/code')

# Patch MANO forward to trace calls
from src.model.mano.server import GenericServer
original_forward = GenericServer.forward

call_stack = []

def traced_forward(self, *args, **kwargs):
    import traceback
    stack_str = ''.join(traceback.format_stack()[-8:-1])
    call_stack.append(stack_str)
    
    print(f"\n🔍 MANO call #{len(call_stack)}")
    print("Stack trace:")
    print(stack_str)
    print("="*60)
    
    return original_forward(self, *args, **kwargs)

GenericServer.forward = traced_forward
GenericServer.__call__ = traced_forward

# Run one step
from train import main
try:
    main()
except Exception as e:
    print(f"\n\n❌ Error after {len(call_stack)} MANO calls:")
    print(f"Error: {e}")
    print("\n\nAll MANO call locations:")
    for i, stack in enumerate(call_stack, 1):
        print(f"\n{'='*60}\nCall #{i}:\n{stack}")
