import subprocess
import time


scenarios = {
    "Jan_Winter": 0,           
    "Feb_Winter": 2678400,     
    "Mar_Spring": 5097600,     
    "Apr_Spring": 7776000,    
    "May_Spring": 10368000,   
    "Jun_Summer": 13132800,    
    "Jul_Summer": 15552000,    
    "Aug_Summer": 18144000,    
    "Oct_Autumn": 23328000,    
    "Nov_Autumn": 25920000     
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