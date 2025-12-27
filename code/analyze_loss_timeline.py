import re
import sys

log_file = sys.argv[1] if len(sys.argv) > 1 else "logs/stage3_1to2_hold_MC1_ho3d_infer_official_loadingUnet_20251226_215815.log"

events = []
with open(log_file, 'r') as f:
    for line_num, line in enumerate(f, 1):
        # Extract step number
        step_match = re.search(r'step[=\s]+(\d+)', line, re.IGNORECASE)
        if not step_match:
            step_match = re.search(r'Step (\d+)', line)
        
        if step_match:
            step = int(step_match.group(1))
            
            # Categorize event
            if 'LOSS COMPONENTS' in line:
                events.append((step, 'LOSS_COMPONENTS', line_num, line.strip()))
            elif 'GHOP.*added' in line or 'GHOP losses added' in line:
                events.append((step, 'GHOP_SUCCESS', line_num, line.strip()))
            elif 'GHOP.*failed' in line or 'GHOP.*ERROR' in line:
                events.append((step, 'GHOP_FAIL', line_num, line.strip()))
            elif 'ZERO LOSS' in line:
                events.append((step, 'ZERO_LOSS', line_num, line.strip()))
            elif 'Phase 3.*ACTIVATED' in line or 'SDS.*ACTIVATED' in line:
                events.append((step, 'PHASE3_START', line_num, line.strip()))
            elif 'Phase 3.*DEACTIVATED' in line:
                events.append((step, 'PHASE3_END', line_num, line.strip()))
            elif 'Phase 5.*ACTIVATED' in line:
                events.append((step, 'PHASE5_START', line_num, line.strip()))

# Sort by step
events.sort()

# Print timeline
print("LOSS CALCULATION TIMELINE")
print("=" * 80)
for step, event_type, line_num, content in events[:100]:  # First 100 events
    print(f"Step {step:5d} | {event_type:20s} | Line {line_num:7d}")
    if event_type in ['GHOP_FAIL', 'ZERO_LOSS']:
        print(f"  -> {content[:100]}")

print("\n" + "=" * 80)
print("EVENT SUMMARY:")
from collections import Counter
event_counts = Counter([e[1] for e in events])
for event_type, count in event_counts.most_common():
    print(f"  {event_type:20s}: {count:5d} occurrences")
