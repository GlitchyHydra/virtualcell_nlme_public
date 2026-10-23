"""Model: Stage 2 pooling (pool), the effect recipe (effects), the same-line rule (sources), promoter neighbours (promoter),
cell generation (generate) and packaging (package).

    python -m vcell train    --config configs/final_def.toml     -> OUT/effects_<tag>.npz
    python -m vcell generate --config configs/final_def.toml     -> data/submission/pred_<tag>.h5ad
    python -m vcell build    --config configs/final_def.toml     (train + generate in one go)
    python -m vcell package  --config configs/final_def.toml     -> data/submission/<tag>.vcc
"""
