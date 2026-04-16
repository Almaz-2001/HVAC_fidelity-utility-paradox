Context
- Direct literature overlay is only available for m_s.
- Article 7 references come from the hydronic benchmark config, so they are contextual, not apples-to-apples with bestest_air.
- All other metrics in this bundle are internal comparisons across our controllers on the same bestest_air windows.

Best controller by m_s per scenario
- Peak heat window: Thermostatic PPO | m_s=0.0529 | viol=1.8% | power=972.8 W
- Typical heat window: Thermostatic PPO | m_s=0.0573 | viol=3.3% | power=1114.7 W

Why surrogate-trained PPO is stronger than current surrogate-MPC
- Thermostatic PPO and HDRL were trained offline over many surrogate episodes and learned robust policy priors, not just one-step optimization.
- They use richer observations, including time, forecasts, previous action and delta-T history.
- HDRL adds seasonal gating and emergency logic; surrogate-MPC currently does not.
- The current surrogate-MPC trusts the surrogate inside the online optimizer, so model bias translates directly into suboptimal real actions on BOPTEST.