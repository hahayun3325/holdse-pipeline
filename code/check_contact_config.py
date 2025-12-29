import yaml

config_path = "confs/stage3_hold_MC1_ho3d_sds_test_2epoch_verify.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

print("=== PHASE 4 CONFIG FROM FILE ===")
phase4 = config.get('phase4', {})
print(f"enabled: {phase4.get('enabled')}")
print(f"contact_start_iter: {phase4.get('contact_start_iter')}")
print(f"contact_end_iter: {phase4.get('contact_end_iter')}")
print(f"w_contact: {phase4.get('w_contact')}")
print(f"w_penetration: {phase4.get('w_penetration')}")
print(f"w_attraction: {phase4.get('w_attraction')}")

print("\n=== CHECK IF VALUES LOADED INTO MODEL ===")
print("Run this in Python REPL after loading checkpoint:")
print("  model = HOLD.load_from_checkpoint('path/to/ckpt.ckpt')")
print("  print(f'phase4_enabled: {model.phase4_enabled}')")
print("  print(f'contact_start_iter: {model.contact_start_iter}')")
print("  print(f'w_contact: {model.w_contact}')")
