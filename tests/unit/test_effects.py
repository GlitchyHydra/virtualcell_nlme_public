"""The effect recipe (model.effects) and the run configs (configs/*.toml)."""
import numpy as np
import pytest

from vcell import config
from vcell.model import effects

L5 = {"rung": "v3", "transform": "z", "scale": 1.5, "kappa": 1.0, "min_effect": 0.02, "bin_mult": (1.5, 1.25, 1.0, 1.0, 0.8),
      "neighbour": 0.15, "sources": ("K562", "CD4_Rest", "HCT116", "HEK293T", "H1"),
      "line_weights": {"KOLF": 0.25, "Pan_H1": 0.25}}                    # EXPERIMENT_LOG L5 (H38 + H40)


def test_steps_in_order_bins_then_cap_then_floor():
    P_row = np.array([0.5, -0.2, 0.012, np.nan, 0.3])
    col = np.array([0, 2, 3, 4, -1])                               # the last universe gene is off the axis
    keep = col >= 0
    mult = np.array([2.0, 1.0, 1.0, 1.5, 1.0, 1.0])
    nb = (np.array([0]), np.array([np.log2(0.15)]))                # gene 0 is a promoter neighbour of the target
    e = effects.target_effect(P_row, col, keep, mult, nb, 0.02, 6)
    assert e[0] == pytest.approx(np.log2(0.15))                    # 0.5 x 2 = 1.0, then capped at log2(0.15)
    assert e[2] == pytest.approx(-0.2) and e[4] == 0.0              # NaN -> 0
    assert e[3] == 0.0                                             # 0.012 x 1.5 = 0.018 < 0.02: floored after the bin
    assert e[1] == 0.0 and e[5] == 0.0


def test_bin_edges():
    m = effects.bin_multipliers(np.array([14.9, 15.0, 39.9, 40.0, 119.9, 120.0, 400.0, 1e4]), [1.5, 1.25, 1.0, 1.0, 0.8])
    assert list(m) == [1.5, 1.25, 1.25, 1.0, 1.0, 1.0, 0.8, 0.8]


def test_weight_key_sorted_and_zero_dropped():
    assert effects.weight_key({"K562": 1.0, "CD4_Rest": 1.0, "KOLF": 0.0, "H1": 1.0}) == (
        ("CD4_Rest", 1.0), ("H1", 1.0), ("K562", 1.0))


def test_context_weights_override_the_base():
    m = config.load_config("configs/final_def.toml").model
    w, e = m.weights(), m.context_weight_sets()["E"]
    assert "U87" not in w and w["K562"] == 1.0 and w["KOLF"] == 0.25
    assert e["U87"] == 4.0 and {k: v for k, v in e.items() if k != "U87"} == w


@pytest.mark.parametrize("name", ["final_def", "final_def_u87x", "l5"])
def test_configs_hold_the_l5_recipe(name):
    m = config.load_config(f"configs/{name}.toml").model
    for k, v in L5.items():
        assert getattr(m, k) == v, k
    assert set(vars(m)) == {"rung", "transform", "scale", "kappa", "min_effect", "bin_mult", "neighbour", "sources",
                            "line_weights", "context_weights"}       # no centring step (L6 lost: 0.1683 < 0.1705)


def test_final_context_weights():
    assert config.load_config("configs/final_def.toml").model.context_weights == {"E": {"U87": 4.0}}
    assert config.load_config("configs/final_def_u87x.toml").model.context_weights == {"E": {"U87x": 4.0}}
    assert config.load_config("configs/l5.toml").model.context_weights == {}
