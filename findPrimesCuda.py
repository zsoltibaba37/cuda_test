#!/usr/bin/env python3
# coding: utf-8

import torch
import time
import math

device = "cuda"

N = 1_000_000_000
START = 200_000
outfile = "primes.txt"

# Number of odd values from 3 to N
size = (N - 1) // 2
is_prime = torch.ones(size, dtype=torch.bool, device=device)

torch.cuda.synchronize()
t0 = time.time()

limit = int(math.sqrt(N))

for i in range((limit - 1) // 2 + 1):
    if is_prime[i]:
        p = 2 * i + 3
        start = (p * p - 3) // 2
        is_prime[start::p] = False

torch.cuda.synchronize()
t1 = time.time()

# --- Convert indices back to prime numbers ---
indices = torch.nonzero(is_prime).flatten()
primes_gpu = 2 * indices + 3

# Add 2 separately
primes_gpu = torch.cat((
    torch.tensor([2], device=device),
    primes_gpu
))

# Filter from START upwards
primes_gpu = primes_gpu[primes_gpu >= START]
primes_cpu = primes_gpu.cpu().numpy()

print(f"Prime number computation completed for range {START}–{N}.")
print("Number of primes:", len(primes_cpu))
print("Computation time (GPU):", round(t1 - t0, 3), "s")

# --- Write primes to file ---
with open(outfile, "w", encoding="utf-8") as f:
    for p in primes_cpu:
        f.write(f"{p}\n")

print(f"Done! Output written to: {outfile}")

