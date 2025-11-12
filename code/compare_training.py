import re

def analyze_log(filename):
    rgb_count = 0
    epoch_count = 0
    
    with open(filename, 'r') as f:
        content = f.read()
    
    rgb_count = len(re.findall(r'DEBUG RGB LOSS', content))
    epoch_matches = re.findall(r'\[Epoch (\d+)\] Avg loss:\s*([0-9]+\.[0-9]+)', content)
    
    return rgb_count, epoch_matches

print("Comparing training logs...")
print("=" * 60)

# Assuming you have both logs
print("\nCurrent training (with fix):")
rgb_count, epochs = analyze_log('production_training.log')
print(f"  RGB loss entries: {rgb_count}")
print(f"  Epochs: {len(epochs)}")
if epochs:
    print(f"  First epoch loss: {epochs[0][1]}")
    print(f"  Last epoch loss: {epochs[-1][1]}")

