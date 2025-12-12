#!/bin/bash

echo "=== MANO Parameter Setup (Stage 1) ==="
echo ""

echo "1. Looking for parameter initialization in nodes:"
grep -n "nn.Parameter\|register_parameter" src/hold/hold.py | head -20

echo ""
echo "2. Finding pose parameter setup:"
grep -B 5 -A 10 "pose.*Parameter\|Parameter.*pose" src/hold/hold.py | head -30

echo ""
echo "3. Finding shape (betas) parameter setup:"
grep -B 5 -A 10 "betas.*Parameter\|Parameter.*betas" src/hold/hold.py | head -30

echo ""
echo "4. Finding translation parameter setup:"
grep -B 5 -A 10 "transl.*Parameter\|Parameter.*transl" src/hold/hold.py | head -30

echo ""
echo "5. Checking if parameters are per-frame or global:"
grep -n "num_frames\|frame.*param\|per.*frame" src/hold/hold.py | head -15
