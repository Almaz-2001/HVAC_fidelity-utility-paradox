

import time
import numpy as np
import requests
import torch



BOPTEST_URL = "http://web:8000"
TESTCASE = "bestest_air"
SURROGATE_PATH = "/app/outputs/surrogate_v2/rc_node_v2_best.pt"

BOPTEST_STEPS = 100       
SURROGATE_STEPS = 100000  




def benchmark_boptest():
    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})

    print("=" * 60)
    print("BOPTEST BENCHMARK")
    print("=" * 60)

    
    print("  Selecting testcase...")
    r = session.post(f"{BOPTEST_URL}/testcases/{TESTCASE}/select", json={}, timeout=300)
    testid = r.json().get("testid")
    print(f"  testid: {testid}")

    
    session.put(f"{BOPTEST_URL}/step/{testid}", json={"step": 3600}, timeout=30)

    
    session.put(f"{BOPTEST_URL}/initialize/{testid}",
                json={"start_time": 0, "warmup_period": 0}, timeout=300)

    
    for _ in range(5):
        session.post(f"{BOPTEST_URL}/advance/{testid}", json={}, timeout=60)

    # Benchmark
    print(f"  Running {BOPTEST_STEPS} steps...")
    actions = {
        "con_oveTSetCoo_activate": 1, "con_oveTSetCoo_u": 296.15,
        "con_oveTSetHea_activate": 1, "con_oveTSetHea_u": 294.15,
        "fcu_oveFan_activate": 1, "fcu_oveFan_u": 0.5,
    }

    t_start = time.perf_counter()
    for i in range(BOPTEST_STEPS):
        session.post(f"{BOPTEST_URL}/advance/{testid}", json=actions, timeout=60)
    t_elapsed = time.perf_counter() - t_start

    # Stop
    session.put(f"{BOPTEST_URL}/stop/{testid}", json={}, timeout=10)

    fps_boptest = BOPTEST_STEPS / t_elapsed
    time_per_step = t_elapsed / BOPTEST_STEPS * 1000  # ms

    print(f"\n  BOPTEST Results:")
    print(f"    Steps:          {BOPTEST_STEPS}")
    print(f"    Total time:     {t_elapsed:.2f} sec")
    print(f"    Speed:          {fps_boptest:.1f} steps/sec")
    print(f"    Per step:       {time_per_step:.1f} ms")
    print(f"    500k steps =    {500000 / fps_boptest / 3600:.1f} hours")
    print(f"    10M steps =     {10000000 / fps_boptest / 3600:.1f} hours")

    return fps_boptest




def benchmark_surrogate():
    from surrogate.rc_node_v2 import RCNeuralODEv2

    print("\n" + "=" * 60)
    print("SURROGATE BENCHMARK")
    print("=" * 60)

    
    checkpoint = torch.load(SURROGATE_PATH, map_location="cpu", weights_only=False)
    model = RCNeuralODEv2(hidden_dim=checkpoint["hidden_dim"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    
    with torch.no_grad():
        for _ in range(100):
            model(
                torch.tensor([20.0]), torch.tensor([5.0]),
                torch.tensor([12.0]), torch.tensor([180.0]),
                torch.tensor([0.5]), torch.tensor([0.3]),
            )

    
    print(f"  Running {SURROGATE_STEPS} single steps...")
    t_zone = 20.0
    t_start = time.perf_counter()
    with torch.no_grad():
        for i in range(SURROGATE_STEPS):
            t_z = torch.tensor([t_zone], dtype=torch.float32)
            t_a = torch.tensor([5.0 + 10.0 * np.sin(i / 100)], dtype=torch.float32)
            h = torch.tensor([(i % 24) * 1.0], dtype=torch.float32)
            d = torch.tensor([(i // 24) % 365 * 1.0], dtype=torch.float32)
            a0 = torch.tensor([np.random.uniform(-1, 1)], dtype=torch.float32)
            a1 = torch.tensor([np.random.uniform(-1, 1)], dtype=torch.float32)
            t_next, p = model(t_z, t_a, h, d, a0, a1)
            t_zone = float(t_next[0])
    t_elapsed = time.perf_counter() - t_start

    fps_single = SURROGATE_STEPS / t_elapsed
    time_per_step = t_elapsed / SURROGATE_STEPS * 1e6  # microseconds

    print(f"\n  Surrogate Results (single step):")
    print(f"    Steps:          {SURROGATE_STEPS:,}")
    print(f"    Total time:     {t_elapsed:.2f} sec")
    print(f"    Speed:          {fps_single:,.0f} steps/sec")
    print(f"    Per step:       {time_per_step:.1f} μs")
    print(f"    500k steps =    {500000 / fps_single:.1f} sec")
    print(f"    10M steps =     {10000000 / fps_single:.1f} sec")

    
    batch_sizes = [32, 128, 1024]
    for bs in batch_sizes:
        n_batches = SURROGATE_STEPS // bs
        t_start = time.perf_counter()
        with torch.no_grad():
            for i in range(n_batches):
                t_z = torch.randn(bs) * 5 + 20
                t_a = torch.randn(bs) * 10 + 5
                h = torch.rand(bs) * 24
                d = torch.rand(bs) * 365
                a0 = torch.rand(bs) * 2 - 1
                a1 = torch.rand(bs) * 2 - 1
                model(t_z, t_a, h, d, a0, a1)
        t_elapsed = time.perf_counter() - t_start
        fps_batch = (n_batches * bs) / t_elapsed
        print(f"    Batched (bs={bs:4d}): {fps_batch:>12,.0f} steps/sec")

    return fps_single




def main():
    print("\n" + "=" * 60)
    print("SPEED BENCHMARK: BOPTEST vs SURROGATE")
    print("=" * 60)

    try:
        fps_boptest = benchmark_boptest()
    except Exception as e:
        print(f"  BOPTEST not available: {e}")
        fps_boptest = 1.0  # fallback estimate

    fps_surrogate = benchmark_surrogate()

    speedup = fps_surrogate / fps_boptest

    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"  BOPTEST:     {fps_boptest:>12,.1f} steps/sec")
    print(f"  Surrogate:   {fps_surrogate:>12,.0f} steps/sec")
    print(f"  Speedup:     {speedup:>12,.0f}×")
    print(f"")
    print(f"  Time to train 500k steps:")
    print(f"    BOPTEST:   {500000 / fps_boptest / 3600:>8.1f} hours")
    print(f"    Surrogate: {500000 / fps_surrogate:>8.1f} seconds")
    print(f"")
    print(f"  Time to train 10M steps:")
    print(f"    BOPTEST:   {10000000 / fps_boptest / 3600:>8.1f} hours")
    print(f"    Surrogate: {10000000 / fps_surrogate:>8.1f} seconds")
    print(f"")
    print(f"  Equivalent real-time (10M steps × 1h/step):")
    print(f"    {10000000 / 8760:.0f} years of building operation")
    print("=" * 60)


if __name__ == "__main__":
    main()