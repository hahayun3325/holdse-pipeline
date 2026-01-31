import re
import sys

log_file = "logs/0cea9fc97/train.log"

# Key steps where degradation occurred
key_steps = [8000, 15000, 25000, 35000, 60000]

print("Step | RGB Loss | Sem Loss | w_sem | Sem% | RGB%")
print("-" * 60)

with open(log_file) as f:
    for line in f:
        for step in key_steps:
            if f"step={step}" in line or f"step {step}" in line:
                # Extract loss values (example pattern, adjust based on actual log format)
                rgb_match = re.search(r'loss_rgb[=:]\s*([\d.]+)', line)
                sem_match = re.search(r'loss_sem[=:]\s*([\d.]+)', line)
                
                if rgb_match and sem_match:
                    rgb = float(rgb_match.group(1))
                    sem = float(sem_match.group(1))
                    
                    # Calculate w_sem based on milestone (assumed 60000)
                    milestone = 60000
                    w_sem = 1.1 - (step / milestone) * (1.1 - 0.1)
                    
                    weighted_sem = sem * w_sem
                    total = rgb + weighted_sem
                    sem_pct = (weighted_sem / total) * 100
                    rgb_pct = (rgb / total) * 100
                    
                    print(f"{step:5d} | {rgb:8.4f} | {sem:8.4f} | {w_sem:5.2f} | {sem_pct:5.1f}% | {rgb_pct:5.1f}%")
                    key_steps.remove(step)
                    break
