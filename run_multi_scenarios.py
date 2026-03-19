import subprocess
import time

# 10 сценариев, охватывающих разные сезоны (время в секундах от начала года)
scenarios = {
    "Jan_Winter": 0,           # 1 января
    "Feb_Winter": 2678400,     # 1 февраля
    "Mar_Spring": 5097600,     # 1 марта
    "Apr_Spring": 7776000,     # 1 апреля
    "May_Spring": 10368000,    # 1 мая
    "Jun_Summer": 13132800,    # 1 июня
    "Jul_Summer": 15552000,    # 1 июля
    "Aug_Summer": 18144000,    # 1 августа
    "Oct_Autumn": 23328000,    # 1 октября
    "Nov_Autumn": 25920000     # 1 ноября
}

def main():
    print(f"Запуск тестирования в 10 погодных сценариях...")
    start_all = time.time()

    for name, t_start in scenarios.items():
        print(f"\n>>> ЗАПУСК: {name}")
        try:
            subprocess.run([
                "python3", "-m", "evaluation.final_test_on_boptest",
                "--start_time", str(t_start),
                "--scenario_name", name
            ], check=True)
        except Exception as e:
            print(f"Ошибка в сценарии {name}: {e}")

    print(f"\nВсе сценарии завершены за {(time.time() - start_all)/60:.2f} мин.")

if __name__ == "__main__":
    main()