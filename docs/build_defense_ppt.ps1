param(
    [string]$ProjectRoot = "C:\Users\user\Desktop\HVAC_DRL_MORL",
    [string]$OutPath = "C:\Users\user\Desktop\HVAC_DRL_MORL\docs\HVAC_DRL_MORL_defense.pptx"
)

$ErrorActionPreference = "Stop"

function Get-RgbValue([int]$r, [int]$g, [int]$b) {
    return ($r + 256 * $g + 65536 * $b)
}

function Format-Num($value, [int]$digits = 1) {
    return ([double]$value).ToString("F$digits")
}

function Add-Textbox($slide, [string]$text, [double]$left, [double]$top, [double]$width, [double]$height, [int]$fontSize, [bool]$bold, [int]$rgb, [string]$fontName) {
    $shape = $slide.Shapes.AddTextbox(1, $left, $top, $width, $height)
    $shape.Line.Visible = 0
    $shape.Fill.Visible = 0
    $shape.TextFrame.WordWrap = -1
    $shape.TextFrame.AutoSize = 0
    $shape.TextFrame.MarginLeft = 2
    $shape.TextFrame.MarginRight = 2
    $shape.TextFrame.MarginTop = 2
    $shape.TextFrame.MarginBottom = 2
    $shape.TextFrame.TextRange.Text = $text
    $shape.TextFrame.TextRange.Font.Name = $fontName
    $shape.TextFrame.TextRange.Font.Size = $fontSize
    $shape.TextFrame.TextRange.Font.Bold = [int]$bold
    $shape.TextFrame.TextRange.Font.Color.RGB = $rgb
    return $shape
}

function Add-Title($slide, [string]$title) {
    $titleRgb = Get-RgbValue 32 52 95
    [void](Add-Textbox $slide $title 34 18 890 42 24 $true $titleRgb "Aptos Display")
    $bar = $slide.Shapes.AddShape(1, 34, 64, 892, 4)
    $bar.Line.Visible = 0
    $bar.Fill.ForeColor.RGB = (Get-RgbValue 78 121 167)
}

function New-BlankSlide($presentation) {
    $slide = $presentation.Slides.Add($presentation.Slides.Count + 1, 12)
    $slide.Background.Fill.ForeColor.RGB = (Get-RgbValue 255 255 255)
    return $slide
}

function Add-BulletSlide($presentation, [string]$title, [string[]]$bullets, [string]$footer = "") {
    $slide = New-BlankSlide $presentation
    Add-Title $slide $title
    $bodyText = ($bullets | ForEach-Object { "- $_" }) -join "`r"
    $body = Add-Textbox $slide $bodyText 54 92 850 360 18 $false (Get-RgbValue 40 40 40) "Aptos"
    $body.TextFrame.TextRange.ParagraphFormat.SpaceAfter = 10
    if ($footer) {
        [void](Add-Textbox $slide $footer 54 474 850 32 11 $false (Get-RgbValue 110 110 110) "Aptos")
    }
}

function Add-ImageSlide($presentation, [string]$title, [string]$imagePath, [string[]]$bullets = @(), [string]$footer = "") {
    $slide = New-BlankSlide $presentation
    Add-Title $slide $title

    if ($bullets.Count -gt 0) {
        $bodyText = ($bullets | ForEach-Object { "- $_" }) -join "`r"
        $body = Add-Textbox $slide $bodyText 44 92 330 360 17 $false (Get-RgbValue 40 40 40) "Aptos"
        $body.TextFrame.TextRange.ParagraphFormat.SpaceAfter = 8
        if (Test-Path $imagePath) {
            [void]$slide.Shapes.AddPicture($imagePath, $false, $true, 392, 96, 520, 350)
        }
    }
    else {
        if (Test-Path $imagePath) {
            [void]$slide.Shapes.AddPicture($imagePath, $false, $true, 56, 92, 840, 380)
        }
    }

    if ($footer) {
        [void](Add-Textbox $slide $footer 54 478 850 28 11 $false (Get-RgbValue 110 110 110) "Aptos")
    }
}

function Add-TitleSlide($presentation, [string]$title, [string]$subtitle, [string]$authorLine, [string]$dateLine) {
    $slide = New-BlankSlide $presentation

    $banner = $slide.Shapes.AddShape(1, 0, 0, 960, 540)
    $banner.Line.Visible = 0
    $banner.Fill.ForeColor.RGB = (Get-RgbValue 246 249 252)

    $accent = $slide.Shapes.AddShape(1, 0, 0, 960, 14)
    $accent.Line.Visible = 0
    $accent.Fill.ForeColor.RGB = (Get-RgbValue 78 121 167)

    [void](Add-Textbox $slide $title 48 92 860 98 28 $true (Get-RgbValue 25 42 86) "Aptos Display")
    [void](Add-Textbox $slide $subtitle 52 210 830 70 18 $false (Get-RgbValue 68 68 68) "Aptos")
    [void](Add-Textbox $slide $authorLine 52 360 500 24 15 $true (Get-RgbValue 44 44 44) "Aptos")
    [void](Add-Textbox $slide $dateLine 52 392 400 22 13 $false (Get-RgbValue 90 90 90) "Aptos")
}

function Add-TwoColumnSlide($presentation, [string]$title, [string[]]$leftBullets, [string[]]$rightBullets, [string]$leftHeader, [string]$rightHeader, [string]$footer = "") {
    $slide = New-BlankSlide $presentation
    Add-Title $slide $title

    [void](Add-Textbox $slide $leftHeader 48 92 390 24 16 $true (Get-RgbValue 32 52 95) "Aptos")
    [void](Add-Textbox $slide $rightHeader 492 92 390 24 16 $true (Get-RgbValue 32 52 95) "Aptos")

    $leftText = ($leftBullets | ForEach-Object { "- $_" }) -join "`r"
    $rightText = ($rightBullets | ForEach-Object { "- $_" }) -join "`r"

    $leftBox = Add-Textbox $slide $leftText 48 124 390 316 16 $false (Get-RgbValue 40 40 40) "Aptos"
    $rightBox = Add-Textbox $slide $rightText 492 124 390 316 16 $false (Get-RgbValue 40 40 40) "Aptos"
    $leftBox.TextFrame.TextRange.ParagraphFormat.SpaceAfter = 8
    $rightBox.TextFrame.TextRange.ParagraphFormat.SpaceAfter = 8

    if ($footer) {
        [void](Add-Textbox $slide $footer 54 476 850 28 11 $false (Get-RgbValue 110 110 110) "Aptos")
    }
}

$comparisonDir = Join-Path $ProjectRoot "outputs\three_model_comparison"
$summaryPath = Join-Path $comparisonDir "comparison_summary.csv"

if (-not (Test-Path $summaryPath)) {
    throw "Missing comparison summary: $summaryPath. Run python evaluation/compare_three_models.py first."
}

$summary = Import-Csv $summaryPath
$pi = $summary | Where-Object { $_.controller_key -eq "standard_pi" }
$thermo = $summary | Where-Object { $_.controller_key -eq "thermostatic" }
$hdrl = $summary | Where-Object { $_.controller_key -eq "hdrl" }

$dashboardImg = Join-Path $comparisonDir "comparison_dashboard.png"
$rmseImg = Join-Path $comparisonDir "comparison_monthly_rmse22.png"
$energyImg = Join-Path $comparisonDir "comparison_monthly_energy.png"
$tradeoffImg = Join-Path $comparisonDir "comparison_tradeoff_scatter.png"

$app = New-Object -ComObject PowerPoint.Application
$app.Visible = -1
$app.DisplayAlerts = 1

$presentation = $app.Presentations.Add()
$presentation.PageSetup.SlideWidth = 960
$presentation.PageSetup.SlideHeight = 540

Add-TitleSlide $presentation `
    "HVAC Building Energy Management via Multi-Objective Deep Reinforcement Learning" `
    "Project defense presentation based on defensePart1.pdf, defensePart2.pdf, MorlProjectFinalv3.pdf, and the latest validated results from the HVAC_DRL_MORL repository." `
    "Almaz Sapargali | DigitAlem LLP | Project IRN AP23488794" `
    "March 2026"

Add-BulletSlide $presentation "Executive Summary" @(
    "Phase 0 stabilized MORL-PPO and produced the first Pareto front with a reported 28% energy reduction.",
    "Phase 1 introduced a physics-informed RC Neural ODE surrogate with 32,313x measured speedup and RMSE = 0.163 C in the report.",
    "Phase 2 solved the inverse calibration problem and improved RMSE by 79%.",
    "Phase 3 tested eleven MORL experiments, then the project continued with a direct-Tsup redesign that produced the strongest current comfort baseline.",
    "Today the validated benchmark picture is: Standard PI = energy reference, Thermostatic PPO = comfort reference, HDRL = structured trade-off prototype."
) "All yearly numbers in the current continuation are computed on 12 representative 336-hour scenario windows."

Add-BulletSlide $presentation "Problem Statement and Research Question" @(
    "HVAC control is inherently multi-objective: comfort, energy, and safety must be balanced at the same time.",
    "Purely model-based control is often accurate but can be computationally expensive or hard to adapt.",
    "Purely model-free RL is flexible, but direct BOPTEST training is slow and sample-inefficient.",
    "The core research question is whether a physics-informed and surrogate-accelerated RL pipeline can remain fast, accurate, and structurally safe enough for deployment-oriented control."
)

Add-BulletSlide $presentation "Research Roadmap: Phases 0 to 3" @(
    "Phase 0: MORL-PPO stabilization and reward/observation debugging.",
    "Phase 1: RC Neural ODE surrogate for fast, differentiable rollouts.",
    "Phase 2: inverse-problem calibration for physical fidelity.",
    "Phase 3: structured controller experiments including reward shaping, GRU forecasts, shielding, SAC, HDRL, real weather, emergency heating, expanded action space, and direct Tsup control."
)

Add-TwoColumnSlide $presentation "Reference Papers and Our Improvement: Wang + Liao" @(
    "Wang et al. (2025): safety metric m_s = r_time + r_sev and the idea of an external safety layer for DRL HVAC control.",
    "Original limitation: MPC-based safety correction is accurate but too slow for edge deployment, around 2.67 s per decision in the report discussion.",
    "Our improvement: replace heavy online MPC prediction with a Neural ODE surrogate and short-horizon surrogate filtering.",
    "Resulting value: safety-aware architecture becomes computationally realistic for edge-oriented deployment."
) @(
    "Liao et al. (2025): hierarchical DRL logic with mode specialization across operating regimes or seasons.",
    "Original insight adopted: hierarchy helps because one flat policy is weak across all seasons.",
    "Our improvement: add a third supervisory layer, Emergency Heating Mode, for hard interception under extreme cold.",
    "This solved the winter crash regime that the learned hierarchy alone could not eliminate."
) "Wang et al. (2025)" "Liao et al. (2025)"

Add-TwoColumnSlide $presentation "Reference Papers and Our Improvement: Gao + Hedayat" @(
    "Gao et al. (2024): thermostatic benchmark focused on holding a nominal indoor target around 22 C.",
    "What we adopted: the idea of a one-objective comfort-tracking benchmark as an ablation study.",
    "Our improvement: direct-Tsup Neural ODE surrogate reaches comparable order of accuracy with a much simpler forecast mechanism than 24 h GRU processing.",
    "Current direct-Tsup thermostatic PPO result: mean RMSE22 = $(Format-Num $thermo.rmse22_mean 3) C, with the strongest current repository-level comfort performance."
) @(
    "Hedayat et al. (2025): low-level control and physically grounded HVAC learning.",
    "What we adopted: physics-aware control design and movement toward lower-level actuation variables.",
    "Our improvement: go beyond physical regularization and identify a structural interface gap caused by the built-in PI controller.",
    "Breakthrough: switch from indirect setpoint control to direct supply-air temperature control and remove the internal PI bottleneck."
) "Gao et al. (2024)" "Hedayat et al. (2025)"

Add-BulletSlide $presentation "How This Work Extends the State of the Art" @(
    "Many SOTA papers report strong performance on narrow scenarios such as peak-cooling weeks or idealized comfort windows.",
    "Our project adds three harder dimensions at the same time: computational speed, seasonal robustness, and diagnosis of control-interface mismatch.",
    "Key repository-level strengths today: Neural ODE speed, physically calibrated surrogate, explicit 12-scenario seasonal validation, and honest controller benchmarking against Standard PI, Thermostatic PPO, and HDRL.",
    "Main scientific message: the bottleneck was not only the RL algorithm; it was the control interface and the hidden PI layer inside BOPTEST."
) "Q1 positioning: this is not only a better controller study, but also a systems-level diagnosis of why certain DRL pipelines fail."

Add-BulletSlide $presentation "Phase 0: MORL-PPO Stabilization" @(
    "Objective: establish a working multi-objective PPO agent on BOPTEST with vector rewards for comfort and energy.",
    "Problem P0.1: scale dominance in the scalarized reward; solution: rescale energy by 1/Pmax.",
    "Problem P0.2: observation-space mismatch; solution: use Box[-1,1] consistently.",
    "Result from the report: 28% energy reduction and a monotonic Pareto front across comfort-energy weights."
)

Add-TwoColumnSlide $presentation "Phase 1: RC Neural ODE Surrogate" @(
    "Physics-informed residual thermal update instead of direct absolute-temperature prediction.",
    "Input features: Tzone, Tamb, hour/day cyclic terms, and action.",
    "Two heads: delta-T head and total-power head.",
    "Residual design prevents the network from canceling the HVAC signal."
) @(
    "Problem P1.1: thermal capacitance too large; fixed by data-driven estimate.",
    "Problem P1.2: parameter drift; fixed via register_buffer.",
    "Problem P1.3: network canceled HVAC; fixed by direct delta-T residual formulation."
) "Architecture" "Key fixes"

Add-BulletSlide $presentation "Phase 1 Results: Why the Surrogate Matters" @(
    "Measured speed benchmark from the report: BOPTEST 37.6 steps/s versus surrogate batch rollout 1,215,752 steps/s.",
    "That corresponds to a 32,313x speedup and changes experiment time from tens of hours to minutes.",
    "The report-level surrogate v2 result reached RMSE = 0.163 C with R2 = 0.991.",
    "This surrogate became the computational foundation for PPO, HDRL, safety rollouts, and rapid ablations."
)

Add-BulletSlide $presentation "Phase 2: Inverse Calibration" @(
    "Objective: recover thermal parameters and improve physical consistency from telemetry.",
    "Problem P2.1: latency collapsed the signal and made calibration ill-posed.",
    "Problem P2.2: linear calibration was too weak; solution: joint fine-tuning of physical parameters and neural weights.",
    "Result from the report: RMSE improved from 9.12 C to 1.91 C, a 79% reduction, with Czon recovered to 7.9% error."
)

Add-BulletSlide $presentation "Phase 3: Eleven MORL Experiments in the Report" @(
    "Experiment 1: PPO baseline with synthetic weather.",
    "Experiment 2: reward shaping; failed and amplified sim-to-real mismatch.",
    "Experiment 3: GRU weather forecast inspired by Gao et al.; partial improvement only.",
    "Experiment 4: runtime shielding inspired by Xu et al.; expensive and inaccurate on BOPTEST.",
    "Experiment 5: SAC; underperformed PPO.",
    "Experiments 6-10: HDRL, real weather, emergency heating, expanded 5-feature state, and BOPTEST fine-tuning.",
    "Experiment 11: direct Tsup control; this became the breakthrough that changed the control interface itself."
) "Source: defensePart1.pdf complete report."

Add-BulletSlide $presentation "Legacy Best Reported Phase 3 Result" @(
    "Best reported MORL/HDRL result before the current continuation: ms = 0.697 +- 0.368 over 12 monthly scenarios.",
    "The strongest reported seasonal behavior was in summer, with ms around 0.140-0.187.",
    "The report concluded that weather mismatch and extreme events were partly solved, but an action-response mismatch remained.",
    "This is why the project did not stop at the report endpoint."
)

Add-BulletSlide $presentation "Thermostatic Ablation and Direct-Tsup Breakthrough" @(
    "The full report also contains a thermostatic ablation study against Gao et al.'s benchmark logic.",
    "Key breakthrough in the report: moving from indirect thermostat-style control to direct supply-air control.",
    "Reported improvement: mean RMSE reduced from 4.70 C to 0.84 C; winter RMSE reduced from 8.89 C to 0.78 C.",
    "Interpretation in the report: the so-called structural gap was caused mainly by BOPTEST's internal PI layer, not by missing building physics."
)

Add-BulletSlide $presentation "Why Phase 3 Had to Continue" @(
    "The final report itself concluded that the remaining gap was structural.",
    "Therefore the continuation focused on changing action semantics, not just tuning PPO longer.",
    "The redesign goal was to unify surrogate training, RL training, and BOPTEST deployment under one physically meaningful interface.",
    "That interface is direct Tsup plus direct fan control."
)

Add-BulletSlide $presentation "Current Direct-Tsup Pipeline in the Repository" @(
    "Data collection: data/collect_tsupply_data.py, 51,200 BOPTEST samples from 4 seasons x 4 policies.",
    "Surrogate training: surrogate/train_surrogate_v2.py with multi-step loss and power prediction.",
    "Comfort-first baseline: training/train_thermostatic.py.",
    "Structured trade-off controller: training/train_hdrl.py.",
    "Energy reference: evaluation/standard_controller_baseline.py.",
    "Ongoing MORL and safety layer: main.py, evaluation/eval_safe_morl.py, and layers/safety/action_filter.py."
)

Add-BulletSlide $presentation "Current Standard PI Baseline: Validated Reference" @(
    "The built-in BOPTEST controller is now evaluated explicitly without any override signals.",
    "Mean fixed-target comfort versus 22 C: RMSE22 = $(Format-Num $pi.rmse22_mean 2) C.",
    "Mean fixed-band violation = $(Format-Num $pi.viol_21_25_mean 1)% and mean fixed-band m_s = $(Format-Num $pi.ms_fixed_mean 3).",
    "Total energy = $(Format-Num $pi.energy_total_kwh 1) kWh = $(Format-Num $pi.energy_total_kwh_m2 2) kWh/m2.",
    "Interpretation: this is the low-energy reference, not the main comfort baseline."
)

Add-BulletSlide $presentation "Current Thermostatic PPO Baseline: Strongest Comfort Result" @(
    "Current mean RMSE22 = $(Format-Num $thermo.rmse22_mean 3) C and mean MAE22 = $(Format-Num $thermo.mae22_mean 3) C.",
    "Within +-1 C = $(Format-Num $thermo.within_1c_mean 1)% and within +-0.5 C = $(Format-Num $thermo.within_05c_mean 1)%.",
    "Mean fixed-band violation = $(Format-Num $thermo.viol_21_25_mean 1)% and mean fixed-band m_s = $(Format-Num $thermo.ms_fixed_mean 3).",
    "Total energy = $(Format-Num $thermo.energy_total_kwh 1) kWh = $(Format-Num $thermo.energy_total_kwh_m2 2) kWh/m2.",
    "Interpretation: this is currently the strongest validated comfort controller in the repository."
)

Add-BulletSlide $presentation "Current HDRL: Structured but Not Yet Dominant" @(
    "Two seasonal PPO experts trained on the direct-Tsup surrogate with a compact 5-feature state.",
    "Mean RMSE22 = $(Format-Num $hdrl.rmse22_mean 3) C and mean fixed-band violation = $(Format-Num $hdrl.viol_21_25_mean 1)%.",
    "Mean fixed-band m_s = $(Format-Num $hdrl.ms_fixed_mean 3).",
    "Total energy = $(Format-Num $hdrl.energy_total_kwh 1) kWh = $(Format-Num $hdrl.energy_total_kwh_m2 2) kWh/m2.",
    "Interpretation: HDRL is operational and structured, but it does not yet beat the thermostatic PPO baseline on the full-year benchmark."
)

Add-ImageSlide $presentation "Visual Comparison: Three Validated Controllers" $dashboardImg @(
    "Thermostatic PPO is best on fixed-target comfort metrics.",
    "Standard PI is best on energy.",
    "HDRL has lower fixed-band violation than Thermostatic PPO, but a worse RMSE22.",
    "This confirms that the three controllers currently occupy different parts of the comfort-energy trade-off."
) "Source: outputs/three_model_comparison/comparison_dashboard.png"

Add-ImageSlide $presentation "Monthly RMSE22 Comparison" $rmseImg @(
    "Thermostatic PPO stays consistently below the PI controller across all benchmark windows.",
    "HDRL remains better than PI in most windows, but still trails Thermostatic PPO on target-tracking accuracy.",
    "This figure makes the comfort hierarchy visually clear."
) "Source: outputs/three_model_comparison/comparison_monthly_rmse22.png"

Add-ImageSlide $presentation "Monthly Energy Comparison" $energyImg @(
    "Standard PI remains far below the RL controllers in total energy.",
    "HDRL occasionally improves over Thermostatic PPO in summer, but the annual gap remains small.",
    "This figure shows why PI must be treated as the low-energy reference baseline."
) "Source: outputs/three_model_comparison/comparison_monthly_energy.png"

Add-ImageSlide $presentation "Comfort-Energy Trade-off Scatter" $tradeoffImg @(
    "The annual trade-off is now easy to interpret in one plane.",
    "Thermostatic PPO sits at the strongest comfort point.",
    "Standard PI sits at the strongest energy point.",
    "HDRL currently lands close to Thermostatic PPO in energy but worse in RMSE22, so it is not yet the winning trade-off controller."
) "Source: outputs/three_model_comparison/comparison_tradeoff_scatter.png"

Add-BulletSlide $presentation "Machine-Generated Conclusion from the Current Comparison" @(
    "Best fixed-target comfort: Thermostatic PPO, mean RMSE22 = $(Format-Num $thermo.rmse22_mean 3) C.",
    "Best energy efficiency: Standard PI, total energy = $(Format-Num $pi.energy_total_kwh 1) kWh.",
    "Best fixed-band violation among RL controllers: HDRL, but HDRL still has worse RMSE22 than Thermostatic PPO.",
    "Annual status: HDRL does not yet beat Thermostatic PPO; its annual energy is about 1.1% higher.",
    "Seasonal nuance: HDRL already reduces summer energy by about 4.3% relative to Thermostatic PPO."
) "This slide is based on outputs/three_model_comparison/comparison_conclusion.txt."

Add-BulletSlide $presentation "Why HDRL Still Trails the Thermostatic Baseline" @(
    "Thermostatic PPO is comfort-first and targets 22 C directly; HDRL optimizes comfort plus energy over the broader [21,25] C band.",
    "Thermostatic PPO uses a rich 17-feature observation with forecast and history; HDRL uses only a compact 5-feature current state.",
    "HDRL also relies on a hard ambient threshold for seasonal switching, which is too crude for shoulder seasons and mixed regimes.",
    "Therefore the current HDRL is a promising structured controller, but not yet the best annual controller."
)

Add-BulletSlide $presentation "Implemented Ideas That Were Tested and Rejected" @(
    "GRU weather forecasting inspired by Gao et al.: implemented, but Experiment 3 increased mean m_s to 0.893 and suffered from forecast mismatch.",
    "Runtime shielding inspired by Xu et al.: implemented, but Experiment 4 increased violations because shield quality depended too strongly on surrogate absolute accuracy.",
    "SAC as an off-policy alternative: implemented, but it underperformed PPO and reached around m_s = 1.063 in the report.",
    "Main lesson: not every SOTA ingredient survives contact with this simulator stack; several methods degraded performance because they amplified sim-to-real mismatch instead of reducing it."
) "These negative results are scientifically useful and strengthen the novelty claim of the final architecture."

Add-BulletSlide $presentation "MORL and Safety Layer: Current Honest Status" @(
    "MORL is already implemented in main.py and the surrogate-based safety filter is implemented in layers/safety/action_filter.py.",
    "The safety filter rolls out the Neural ODE surrogate over a short horizon and replaces unsafe actions with a fallback controller.",
    "After the direct-Tsup redesign, the main fully validated results are still Standard PI, Thermostatic PPO, and HDRL.",
    "For the defense, MORL should be presented as architecturally ready and ongoing, not as the final quantitative headline."
)

Add-TwoColumnSlide $presentation "Main Contributions vs Current Limitations" @(
    "Full pipeline from BOPTEST RL to surrogate-accelerated control.",
    "Physics-informed RC Neural ODE with strong speed and accuracy.",
    "Inverse calibration for physical fidelity.",
    "Systematic Phase 3 experiment program with explicit problem-solution logging.",
    "Direct-Tsup redesign that fixed the strongest control-interface mismatch."
) @(
    "HDRL still does not beat the comfort-first baseline on the annual trade-off.",
    "PI remains much more energy-efficient because it solves a softer scheduled-comfort task.",
    "MORL has not yet been fully re-evaluated on the direct-Tsup pipeline.",
    "The seasonal gate is still threshold-based rather than learned."
) "Contributions" "Limitations"

Add-BulletSlide $presentation "Next Steps" @(
    "Retrain MORL first on the direct-Tsup surrogate, then fine-tune on BOPTEST.",
    "Replace the hard seasonal switch with a learned gate or at least hysteresis.",
    "Give HDRL and MORL the same richer temporal context used by the Thermostatic PPO baseline.",
    "Evaluate Safe MORL on the new pipeline and extend comfort metrics toward PMV/PPD and ASHRAE 55.",
    "Prepare deployment-oriented validation for edge hardware and real-building sensor streams."
)

Add-BulletSlide $presentation "Final Defense Message" @(
    "Phase 0 stabilized the MORL formulation.",
    "Phase 1 delivered the core computational breakthrough through a Neural ODE surrogate.",
    "Phase 2 improved physical trustworthiness through inverse calibration.",
    "Phase 3 identified the real structural bottleneck and led to the direct-Tsup redesign.",
    "Today the strongest validated result is the direct-Tsup Thermostatic PPO baseline, while HDRL is a working but not yet dominant trade-off controller.",
    "The repository is now technically ready for the final MORL step."
)

Add-BulletSlide $presentation "Thank You" @(
    "Questions and discussion.",
    "Backup material can be generated from the same repository: LaTeX defense deck, comparison plots, yearly summaries, and evaluation scripts."
) "Files created in this repo: docs/project_defense_presentation.tex and docs/HVAC_DRL_MORL_defense.pptx"

$outDir = Split-Path -Parent $OutPath
if (-not (Test-Path $outDir)) {
    New-Item -ItemType Directory -Path $outDir | Out-Null
}

if (Test-Path $OutPath) {
    Remove-Item -LiteralPath $OutPath -Force
}

$presentation.SaveAs($OutPath)
$presentation.Close()
$app.Quit()

[System.Runtime.Interopservices.Marshal]::ReleaseComObject($presentation) | Out-Null
[System.Runtime.Interopservices.Marshal]::ReleaseComObject($app) | Out-Null
[GC]::Collect()
[GC]::WaitForPendingFinalizers()

Write-Output "Saved presentation: $OutPath"
