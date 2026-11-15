"""Group 14: no order path; one account type per account; verified-spec guard; no behavioural cap
in the primary configs."""
import re
from pathlib import Path

import pytest
import yaml

from tradeworld.brokerspec import load_profile
from tradeworld.errors import ConfigError, UnverifiedSpecError
from tradeworld.interface.experiment import load_experiment, mapper_for
from tradeworld.market.instruments import InstrumentSpec, Universe

SRC = Path(__file__).resolve().parents[1] / "src" / "tradeworld"
FORBIDDEN = [r"\bimport\s+ccxt", r"\bMetaTrader5\b", r"metaapi", r"agiliumtrade", r"order_send",
             r"create_order", r"/trade\b", r"requests\.post", r"POSITION_CLOSE", r"ORDER_TYPE_BUY"]


def test_no_order_path_in_package():
    hits = []
    for f in SRC.rglob("*.py"):
        text = f.read_text()
        for pat in FORBIDDEN:
            if re.search(pat, text, flags=re.IGNORECASE):
                hits.append(f"{f.relative_to(SRC)}: {pat}")
    assert not hits, hits


def test_one_account_type_per_account():
    with pytest.raises(ConfigError, match="mix account types"):
        Universe((InstrumentSpec("EURUSDc", "EUR", "USD", "fx", account_type="exness_standard_cent"),
                  InstrumentSpec("BTCUSD", "BTC", "USD", "crypto", account_type="exness_standard")))


def test_environment_b_blocked_while_unverified():
    with pytest.raises(UnverifiedSpecError) as e:
        load_profile("environments/B_exness_cent.yaml")
    fields = e.value.fields
    assert fields["stop_out_level"] == "unknown"                      # legal entity not verified (U1)
    assert fields["margin.account_leverage"] == "unknown"
    assert fields["instruments[0].swap_long"] == "unknown"
    assert fields["instruments[3].contract_size"] == "unverified"     # XAUUSDc (U4)


def test_sensitivity_requires_candidate_values_and_tags_results():
    with pytest.raises(UnverifiedSpecError, match="no candidate values"):
        load_profile("environments/B_exness_cent.yaml", sensitivity=True)
    candidates = {"stop_out_level": 0.0, "margin": {"account_leverage": 2000},
                  "instruments": None}
    base = yaml.safe_load((SRC.parents[1] / "configs/brokers/exness/standard_cent.yaml").read_text())
    for inst in base["instruments"]:
        for k in ("swap_long", "swap_short"):
            inst[k] = {"value": 0.0, "status": "unverified"}
        inst["swap_free"] = {"value": False, "status": "unverified"}
    candidates["instruments"] = base["instruments"]
    prof = load_profile("environments/B_exness_cent.yaml", sensitivity=True, overrides=candidates)
    assert "unverified-spec" in prof.tags and prof.environment == "B"
    assert {s.account_type for s in prof.universe.specs} == {"exness_standard_cent"}
    assert "BTCUSDc" not in prof.universe.symbols and "USTECc" not in prof.universe.symbols


def test_plain_values_are_never_verified_in_b(tmp_path):
    root = SRC.parents[1] / "configs"
    cfg = tmp_path / "b.yaml"
    cfg.write_text("name: x\nenvironment: B\nstop_out_level: 0.0\nmargin: {account_leverage: 100}\n"
                   "instruments: [{symbol: EURUSDc, base: EUR, quote: USD, asset_class: fx}]\n")
    with pytest.raises(UnverifiedSpecError) as e:
        load_profile(cfg, root=root)
    assert e.value.fields["stop_out_level"] == "unverified"


def test_environment_a_loads_with_assumptions_and_no_cap():
    prof = load_profile("environments/A_fractional.yaml")
    assert prof.environment == "A" and len(prof.universe.symbols) == 6 and not prof.nbp
    exp = load_experiment("experiments/primary_uncapped.yaml")
    assert exp["exposure_cap"] is None
    assert mapper_for(exp, 6).cap is None
    bench = load_experiment("experiments/leverage_benchmarks.yaml")
    assert sorted(v["exposure_cap"] for v in bench["variants"]) == [3, 10, 50, 100]
