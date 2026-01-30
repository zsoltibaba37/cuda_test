import time
import math

N = 500_000_000

start = time.perf_counter()

is_prime = [True] * (N + 1)
is_prime[0] = is_prime[1] = False

for p in range(2, int(math.isqrt(N)) + 1):
    if is_prime[p]:
        step = p
        start_idx = p * p
        is_prime[start_idx:N+1:step] = [False] * len(range(start_idx, N+1, step))

primes = [i for i, v in enumerate(is_prime) if v]

end = time.perf_counter()

print(f"Prímek száma: {len(primes)}")
print(f"Idő: {end - start:.4f} s")

