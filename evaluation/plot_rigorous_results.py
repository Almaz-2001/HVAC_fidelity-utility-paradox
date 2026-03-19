import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def calculate_stats(data_array):
    """Вычисляет среднее и 95% доверительный интервал для массива по сценариям."""
    mean_val = np.mean(data_array)
    std_val = np.std(data_array)
    # N теперь равно количеству сценариев (10)
    ci_95 = 1.96 * (std_val / np.sqrt(len(data_array)))
    return mean_val, ci_95

def plot_with_ci(ax, data_matrix, label, color, ylabel, ylim=None):
    """Рисует среднюю линию по году и область сезонной вариативности."""
    mean = np.mean(data_matrix, axis=1)
    std = np.std(data_matrix, axis=1)
    ci = 1.96 * (std / np.sqrt(data_matrix.shape[1])) 
    
    days = np.arange(len(mean)) / 24.0
    ax.plot(days, mean, label=label, color=color, lw=2)
    # Закрашенная область теперь показывает разброс между зимой и летом
    ax.fill_between(days, mean - ci, mean + ci, color=color, alpha=0.2)
    ax.set_ylabel(ylabel, fontsize=12)
    if ylim:
        ax.set_ylim(ylim)
    ax.grid(True, alpha=0.2)
    ax.legend(loc='upper right')

def main():
    output_dir = "outputs"
    temp_list, power_list, ms_list = [], [], []
    
    # Список имен сценариев, которые мы запускали
    scenario_names = [
        "Jan_Winter", "Feb_Winter", "Mar_Spring", "Apr_Spring", 
        "May_Spring", "Jun_Summer", "Jul_Summer", "Aug_Summer", 
        "Oct_Autumn", "Nov_Autumn"
    ]

    print(f"Загрузка данных из {len(scenario_names)} сезонных сценариев...")
    
    for name in scenario_names:
        path = os.path.join(output_dir, f"metrics_scenario_{name}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            temp_list.append(df["temp"].values)
            power_list.append(df["power"].values)
            ms_list.append(df["m_s"].values)
        else:
            print(f"Предупреждение: Файл для сценария {name} не найден.")

    if not temp_list:
        print("Ошибка: Данные сценариев не найдены! Запустите run_multi_scenarios.py")
        return

    # Матрицы [шаги (336), сценарии (10)]
    T = np.array(temp_list).T
    P = np.array(power_list).T
    MS = np.array(ms_list).T

    # --- 1. СТАТИСТИКА ПО СЕЗОНАМ ---
    T_LOW, T_HIGH = 21.0, 25.0
    violations_mask = (T < T_LOW) | (T > T_HIGH)
    r_time_scenarios = np.mean(violations_mask, axis=0) * 100
    r_time_mean, r_time_ci = calculate_stats(r_time_scenarios)

    energy_scenarios = np.sum(P, axis=0) / 1000.0
    energy_mean, energy_ci = calculate_stats(energy_scenarios)

    ms_final_scenarios = MS[-1, :]
    ms_mean, ms_ci = calculate_stats(ms_final_scenarios)

    # --- ВЫВОД В КОНСОЛЬ ---
    print("\n" + "="*65)
    print("ИТОГОВАЯ ГОДОВАЯ СТАТИСТИКА (10 Сезонов, Mean ± 95% CI)")
    print("="*65)
    print(f"Violation Time (r_time):  {r_time_mean:.2f}% ± {r_time_ci:.2f}%")
    print(f"Energy Consumption:       {energy_mean:.2f} ± {energy_ci:.2f} kWh/14d")
    print(f"Safety Indicator (m_s):   {ms_mean:.3f} ± {ms_ci:.3f}")
    

    # --- ВИЗУАЛИЗАЦИЯ ---
    plt.style.use('seaborn-v0_8-paper')
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Температура: Shaded area теперь показывает разницу между зимой и летом
    plot_with_ci(axs[0], T, "Seasonal Zone Temp (Avg)", "#1f77b4", "Temp [°C]", ylim=(12, 28))
    axs[0].axhline(T_LOW, color='red', linestyle='--', alpha=0.5, label="Comfort Range")
    axs[0].axhline(T_HIGH, color='red', linestyle='--', alpha=0.5)
    axs[0].set_title("Annual Robustness Analysis across 10 Weather Scenarios", fontsize=14)

    # Мощность
    plot_with_ci(axs[1], P, "HVAC Power Demand", "#ff7f0e", "Power [W]")
    
    # Безопасность
    plot_with_ci(axs[2], MS, "Safety Indicator ($m_s$)", "#9467bd", "m_s Score")
    
    axs[2].set_xlabel("Simulation Time [Days]", fontsize=12)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "annual_scenarios_plot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nГрафик сохранен как: {plot_path}")

if __name__ == "__main__":
    main()