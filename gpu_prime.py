#!/usr/bin/env python3
# coding: utf-8

import torch
import time
import math

device = "cuda"

N = 1_000_000_000

torch.cuda.synchronize()
start_total = time.perf_counter()

# Teljes boolean tömb 0..N
is_prime = torch.ones(N + 1, dtype=torch.bool, device=device)
is_prime[0:2] = False

limit = int(math.isqrt(N))

for p in range(2, limit + 1):
    if is_prime[p]:  # CUDA-n marad, nem húzza vissza CPU-ra
        start_idx = p * p
        is_prime[start_idx:N+1:p] = False

# prímek indexei
primes_gpu = torch.nonzero(is_prime).flatten()

torch.cuda.synchronize()
end_total = time.perf_counter()

print(f"Prímek száma (GPU): {primes_gpu.numel()}")
print(f"Idő (GPU, teljes): {end_total - start_total:.4f} s")
