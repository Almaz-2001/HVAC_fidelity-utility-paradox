from __future__ import annotations
import os
import time
from pathlib import Path
from typing import Dict, Any, Tuple
import requests
import pandas as pd

# --- КОНФИГУРАЦИЯ ---
BASE_URL = os.environ.get("BOPTEST_URL", "http://host.docker.internal:8000")
TESTCASE = os.environ.get("BOPTEST_TESTCASE", "bestest_air")
OUT_DIR = Path(os.environ.get("OUT_DIR", "/app/outputs/boptest"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "morl_log.csv"

# Параметры симуляции
N_STEPS = int(os.environ.get("N_STEPS", "48")) 
T_LOW_C, T_HIGH_C = 20.0, 26.0
ENERGY_SCALE = 1e-6

def c_to_k(x_c: float) -> float: return x_c + 273.15
def k_to_c(x_k: float) -> float: return x_k - 273.15

def post_json(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Универсальный метод для POST-запросов"""
    r = requests.post(f"{BASE_URL}{path}", json=payload, timeout=300)
    if r.status_code != 200:
        print(f"[DEBUG] Error Body: {r.text}")
    r.raise_for_status()
    return r.json()

def select_testcase(testcase: str) -> str:
    """Выбор тесткейса и получение TestID"""
    j = post_json(f"/testcases/{testcase}/select", {})
    return j["testid"]

def advance(testid: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Отправляем ТОЛЬКО управляющие сигналы. 
    Переменная 'step' удалена, так как сервер считает её ошибочной (Unexpected input).
    """
    j = post_json(f"/advance/{testid}", inputs)
    # Пытаемся достать данные из payload, если они там, иначе берем весь корень JSON
    return j.get("payload", j)

def comfort_penalty(t_room_c: float) -> float:
    if t_room_c < T_LOW_C: return -(T_LOW_C - t_room_c)
    if t_room_c > T_HIGH_C: return -(t_room_c - T_HIGH_C)
    return 0.0

def simple_policy(t_room_c: float) -> Dict[str, Any]:
    """Простая логика управления (Deadband)"""
    target = 23.0
    if t_room_c > target + 0.5:
        fan, t_coo, t_hea = 0.7, 22.0, 19.0
    elif t_room_c < target - 0.5:
        fan, t_coo, t_hea = 0.3, 25.0, 21.0
    else:
        fan, t_coo, t_hea = 0.4, 24.0, 20.0
        
    return {
        "con_oveTSetCoo_activate": 1,
        "con_oveTSetCoo_u": c_to_k(t_coo),
        "con_oveTSetHea_activate": 1,
        "con_oveTSetHea_u": c_to_k(t_hea),
        "fcu_oveFan_activate": 1,
        "fcu_oveFan_u": float(fan),
    }

def main():
    print(f"[BOPTEST] URL: {BASE_URL}")
    try:
        testid = select_testcase(TESTCASE)
        print(f"[BOPTEST] TestID: {testid}")
    except Exception as e:
        print(f"Failed to connect: {e}")
        return

    rows = []
    # 1. Инициализация (первый пустой шаг для получения начального состояния)
    neutral = {
        "con_oveTSetCoo_activate": 0, 
        "con_oveTSetHea_activate": 0, 
        "fcu_oveFan_activate": 0
    }
    print("[BOPTEST] Initializing simulation...")
    res = advance(testid, neutral)
    
    # 2. Основной цикл симуляции
    for step in range(N_STEPS):
        # Ищем температуру в ответе (может быть в корне или в ['payload'])
        t_room_k = res.get("zon_reaTRooAir_y")
        
        if t_room_k is None:
            print(f"Error: No temperature data in response. Keys: {list(res.keys())}")
            break
            
        t_room_c = k_to_c(float(t_room_k))
        
        # Получаем управляющие сигналы
        u = simple_policy(t_room_c)
        
        # Делаем шаг в симуляторе
        res = advance(testid, u)

        # Сбор данных для анализа
        p_cool = float(res.get("fcu_reaPCoo_y", 0.0))
        p_fan = float(res.get("fcu_reaPFan_y", 0.0))
        comfort = comfort_penalty(t_room_c)
        energy = -( (p_cool + p_fan) * ENERGY_SCALE )
        
        rows.append({
            "step": step,
            "zone_temp": t_room_c,
            "comfort": comfort,
            "energy": energy,
            "p_cool": p_cool,
            "time": res.get("time")
        })

        if step % 6 == 0:
            print(f"Step {step:02d}: T={t_room_c:.2f}°C | P_cool={p_cool:.1f}W")

    # Сохранение
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(OUT_CSV, index=False)
        print(f"\n[SUCCESS] Simulation finished. Results saved to {OUT_CSV}")
    else:
        print("\n[ERROR] No data collected.")

def stop_testcase(testid: str):
    requests.post(f"{BASE_URL}/stop/{testid}", timeout=10)


if __name__ == "__main__":
    try:

        main()
    finally:
        pass
    