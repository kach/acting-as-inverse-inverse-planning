#!/bin/bash

python hoh.py a 1 1 0 0 0 &
python hoh.py a 1 5 0 0 0 ;

wait

for rho in -3 +3; do
    python hoh.py b 1 1 1 1 $rho &
    python hoh.py b 1 5 1 1 $rho &
    python hoh.py b 1 1 1 5 $rho &
    python hoh.py b 1 5 1 5 $rho &
    python hoh.py b 1 1 3 3 $rho &
    python hoh.py b 1 5 3 3 $rho &
done

# wait

for rho in -1 +1; do
    python hoh.py b 1 1 1 1 $rho &
    python hoh.py b 1 5 1 1 $rho &
    python hoh.py b 1 1 1 5 $rho &
    python hoh.py b 1 5 1 5 $rho &
    python hoh.py b 1 1 3 3 $rho &
    python hoh.py b 1 5 3 3 $rho &
done

# wait

for rho in 0; do
    python hoh.py b 1 1 1 1 $rho &
    python hoh.py b 1 5 1 1 $rho &
    python hoh.py b 1 1 1 5 $rho &
    python hoh.py b 1 5 1 5 $rho &
    python hoh.py b 1 1 3 3 $rho &
    python hoh.py b 1 5 3 3 $rho &
done

wait