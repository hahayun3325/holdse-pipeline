#!/bin/bash

echo "=== Implicit Network Architecture ==="
echo ""

# Find the main implicit network class
IMPLICIT_FILE=$(find src/ -name "*implicit*.py" | grep -v __pycache__ | head -1)

if [ -z "$IMPLICIT_FILE" ]; then
    echo "Searching in model/networks directory..."
    IMPLICIT_FILE=$(find src/model/networks -name "*.py" | head -1)
fi

if [ -n "$IMPLICIT_FILE" ]; then
    echo "Found: $IMPLICIT_FILE"
    echo ""
    
    echo "1. Network Classes:"
    grep -n "^class.*Network" "$IMPLICIT_FILE"
    
    echo ""
    echo "2. Forward Method:"
    grep -B 3 -A 30 "def forward" "$IMPLICIT_FILE" | head -40
    
    echo ""
    echo "3. SDF Computation:"
    grep -B 2 -A 10 "sdf\|signed.*distance" "$IMPLICIT_FILE" | head -20
else
    echo "Implicit network file not found. Checking hold.py directly..."
    grep -A 50 "implicit_network.*=" src/hold/hold.py | head -60
fi
