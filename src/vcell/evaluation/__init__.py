"""Evaluation: leave-one-line-out proxies of the six scored metrics (panel, proxy, mse_proxy), the 8-line harness used for
every tuning decision (harness), and the official scorer on the public H1 benchmark (h1_benchmark).

    python -m vcell evaluate proxy   --config configs/final_def.toml   # five held-out lines, seconds to minutes
    python -m vcell evaluate harness --config configs/final_def.toml   # eight held-out lines (adds the essential axis)
    python -m vcell evaluate h1      --config configs/e4_h1.toml       # generate for H1 and run vcc-h1 score (~25 min)
"""
