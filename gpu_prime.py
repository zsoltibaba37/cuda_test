import time
import cupy as cp
import math

N = 500_000_000

start_total = time.perf_counter()

# GPU-n egy bool tömb: True = potenciálisan prím
is_prime = cp.ones(N + 1, dtype=cp.bool_)
is_prime[0:2] = False

limit = int(math.isqrt(N))

for p in range(2, limit + 1):
    if bool(is_prime[p]):  # ezt a bool-t visszahozza CPU-ra
        start_idx = p * p
        is_prime[start_idx:N+1:p] = False

# prímek indexei
primes_gpu = cp.nonzero(is_prime)[0]

cp.cuda.Stream.null.synchronize()
end_total = time.perf_counter()

print(f"Prímek száma (GPU): {int(primes_gpu.size)}")
print(f"Idő (GPU, teljes): {end_total - start_total:.4f} s")

