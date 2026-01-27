#!/usr/bin/env python3
# coding: utf-8

import torch
import time
import math

device = "cuda"

# for me max 2_000_000_000 only 4GB vRam
N = 5_000_000
outfile = "primes.txt"

# --- CUDA szita ---
is_prime = torch.ones(N + 1, dtype=torch.bool, device=device)
is_prime[0:2] = False

torch.cuda.synchronize()
t0 = time.time()

for i in range(2, int(math.sqrt(N)) + 1):
    if is_prime[i]:
        is_prime[i*i:N+1:i] = False

torch.cuda.synchronize()
t1 = time.time()

# --- Prímek kigyűjtése ---
primes_gpu = torch.nonzero(is_prime).flatten()
primes_cpu = primes_gpu.cpu().numpy()   # GPU -> CPU

print("Prímek száma:", len(primes_cpu))
print("Számítási idő (GPU):", round(t1 - t0, 3), "s")

# --- Kiírás fájlba ---
with open(outfile, "w", encoding="utf-8") as f:
    for p in primes_cpu:
        f.write(f"{p}\n")

print(f"Kész! Kiírva ide: {outfile}")
