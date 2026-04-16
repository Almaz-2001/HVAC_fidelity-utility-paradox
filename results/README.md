# Текущий срез проекта HVAC_DRL_MORL

Пакет `results/` собирает текущие валидированные результаты по surrogate, thermostatic PPO, HDRL и MORL/safety layer.

Важно:
- `reported_in_reports` означает milestone-числа из `reports/Current_HVAC_PPO_MORL.pdf` и `reports/defensePart2.pdf`.
- `measured_from_outputs` означает текущие фактические числа из `outputs/*.csv` и `outputs/*.json`.
- Для презентации текущего состояния репозитория главным источником нужно считать именно `outputs/*`.

## 1. Короткий итог по состоянию проекта

- Лучшая валидированная comfort-модель: Thermostatic PPO, mean RMSE22 = 0.867 C, mean m_s = 0.228, total energy = 3078.1 kWh.
- Энергетический reference: Standard PI, total energy = 1230.2 kWh, но RMSE22 = 3.410 C.
- Лучшая RL-модель по violation rate: HDRL, violation [21,25]C = 11.24%, но mean RMSE22 = 1.160 C.
- MORL и safety layer уже реализованы, но ещё не являются strongest validated result после direct-TSup redesign.

## 2. Суррогатный digital twin

### 2.1 Откуда взялись данные
- Сбор идёт через `data/collect_tsupply_data.py` на BOPTEST testcase `bestest_air`.
- Для dataset generation используется прямое управление `fcu_oveTSup_u` и `fcu_oveFan_u`.
- Внутренний PI-контур BOPTEST нейтрализуется фиксированием `Tset,cool = 40C` и `Tset,heat = 15C`.
- Датасет строится как 4 сезона x 4 политики x 3200 шагов = 51 200 переходов.
- Основные поля: `t_zone`, `t_amb`, `hour`, `day`, `a0_raw`, `a1_raw`, `t_zone_next`, `delta_t`, `p_total`.

### 2.2 Обучение суррогата
- `surrogate/train_surrogate_v2.py` использует `BOPTESTDatasetV2` и multi-step loss по горизонтам `[2,4]`.
- Зафиксированный report milestone: RMSE = 0.163 C, R2 = 0.991, speedup = 32313x.
- Текущий live-rollout baseline: 1h RMSE = 0.609 C, 24h rollout RMSE = 0.754 C.

### 2.3 Что означают Stage A / B / C
- Stage A: preprocessing observed trace, компенсация noise/latency/bias/scale артефактов.
- Stage B: физическая идентификация `C_zon` на возбуждённых окнах с высоким `|dT/dt|` при почти замороженном backbone.
- Stage C: мягкая калибровка heads или части модели, чтобы улучшить fit без разрушения найденной физики.

### 2.4 Текущий статус inverse calibration
- v3: calibrated RMSE = 0.4049 C, но C_zon error = 34.72%.
- v3.5 heads-only: calibrated RMSE = 0.5256 C, C_zon error = 3.07%.
- raw v3.5 rollout: 24h RMSE = 0.761 C.
- calibrated v3.5 rollout: 24h RMSE = 0.766 C.
- Вывод: физическая идентификация уже стала качественной, но free-run rollout realism для калиброванного twin ещё не улучшен.

## 3. Thermostatic PPO
- Обучается в `training/train_thermostatic.py`.
- Работает на direct-TSup surrogate с observation budget 17 признаков: физическое состояние, cyclic time, forecast, previous action, delta-T history.
- Reward ориентирован на tracking 22C, с повышенным штрафом за winter underheating и только слабой energy regularization около target.
- Валидация идёт в `evaluation/eval_thermostatic.py` по 12 месячным сценариям BOPTEST `bestest_air`.
- Текущий итог: RMSE22 = 0.867 C, MAE22 = 0.721 C, within ±1C = 72.3%, total energy = 3078.1 kWh.

## 4. HDRL
- Обучается в `training/train_hdrl.py` как два PPO-эксперта: winter и summer.
- Evaluation в `evaluation/yearly_validation_hdrl.py` использует seasonal gate и emergency heating rule.
- Текущий итог: RMSE22 = 1.160 C, violation = 11.24%, m_s = 0.252, total energy = 2956.3 kWh.
- Интерпретация: HDRL уже выглядит как structured trade-off controller, но пока не доминирует над thermostatic PPO на annual benchmark.

## 5. MORL / safety layer
- Entry point MORL: `main.py`.
- Yearly BOPTEST validation для MORL: `evaluation/yearly_validation_morl.py`.
- Surrogate-based action filtering: `layers/safety/action_filter.py`.
- Report milestone Phase 3: best m_s = 0.697 ± 0.368.
- Current raw Pareto balanced point: m_s = 0.506, total energy = 481.7 kWh.
- Current safe-MORL multi-seed snapshot: ppo_sf_ms_mean = 1.737, acceptance mean = 8.7%.
- Честный вывод: MORL нужно показывать как готовую архитектурную следующую фазу, а не как текущий strongest validated result.

## 6. Что находится в results/
- `figures/paper`: canonical figure из основных отчётов.
- `figures/controllers`: сравнение Standard PI / Thermostatic PPO / HDRL.
- `figures/surrogate`: surrogate rollout, parity и comfort-trace графики.
- `figures/calibration`: raw vs calibrated v3.5 rollout-сравнение.
- `tables/raw`: исходные CSV/JSON/TXT, на которые опирается этот snapshot.
- `tables/controller_highlights.csv`, `tables/surrogate_progress.csv`, `tables/morl_progress.csv`, `tables/code_entrypoints.csv`: компактные презентационные таблицы.
- `manifests`: список скопированных файлов и список удалённых временных графиков.

## 7. Что было очищено
- Удалены только временные PNG из промежуточных v3.5 rollout-экспериментов (`prior420_rollout_heads`, `prior420_rollout_heads_nonlinear`, `prior420_rollout_heads_mixed`).
- CSV, JSON, модели и остальные исходные результаты сохранены.
