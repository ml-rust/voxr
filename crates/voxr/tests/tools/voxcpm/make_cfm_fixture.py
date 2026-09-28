#!/usr/bin/env python3
"""Reference trajectory for VoxCPM2's conditional-flow-matching sampler (feat_decoder).

Sibling of make_dit_fixture.py, which captures the estimator this sampler wraps.
Here the target is UnifiedCFM's Euler integrator (modules/locdit/unified_cfm.py).

Determinism: UnifiedCFM.forward draws `z = torch.randn(...) * temperature`
internally (unified_cfm.py:64), and torch's RNG is not reproducible in Rust.
So this script draws z itself under an explicit manual_seed, SAVES it, and calls
solve_euler DIRECTLY with every argument passed explicitly -- forward is
bypassed. The realized t_span (after the sway transform) is saved too, so a
schedule bug is caught independently of the integrator.

Per-step x is saved (step0..stepN-1), including the state after the
zero-velocity warmup step, so a divergence localizes to one Euler update
instead of only showing up in the final output.

Batch size 2: optimized_scale reduces per batch row (dim=1, keepdim), so a
batch-broadcasting bug there is invisible at b=1.

Built in f32 for the same reason as make_dit_fixture.py.
"""
import json, sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(sys.argv[1])))
from voxcpm.modules.minicpm4 import MiniCPM4Config      # noqa: E402
from voxcpm.modules.locdit import VoxCPMLocDiTV2        # noqa: E402
from voxcpm.modules.locdit import CfmConfig, UnifiedCFM  # noqa: E402

ckpt_dir, out = Path(sys.argv[2]), Path(sys.argv[3])
cfg = json.loads((ckpt_dir / "config.json").read_text())

# Same construction as make_dit_fixture.py / VoxCPM2Model.__init__
# (model/voxcpm2.py:205-219): decoder_config STARTS from lm_config and only
# these six fields are overridden from dit_config.
dec = MiniCPM4Config(**cfg["lm_config"])
d = cfg["dit_config"]
dec.hidden_size = d["hidden_dim"]
dec.intermediate_size = d["ffn_dim"]
dec.num_attention_heads = d["num_heads"]
dec.num_hidden_layers = d["num_layers"]
dec.kv_channels = d["kv_channels"]
dec.vocab_size = 0

feat_dim = cfg["feat_dim"]
patch_size = cfg["patch_size"]

# --- Sampler settings, each confirmed against the live call site -------------
# model/voxcpm2.py:1065-1071 calls self.feat_decoder(mu=..., patch_size=...,
# cond=..., n_timesteps=inference_timesteps, cfg_value=cfg_value) and passes
# NOTHING else, so temperature / sway_sampling_coef / use_cfg_zero_star take
# UnifiedCFM.forward's defaults (unified_cfm.py:57-60).
N_TIMESTEPS = 10           # inference_timesteps default, voxcpm2.py:471/789/983
CFG_VALUE = 2.0            # cfg_value default, voxcpm2.py:472/790/984
TEMPERATURE = 1.0          # unified_cfm.py:57 default, never overridden
SWAY_SAMPLING_COEF = 1.0   # unified_cfm.py:59 default, never overridden
USE_CFG_ZERO_STAR = True   # unified_cfm.py:60 default, never overridden
# mean_mode comes from config.dit_config.dit_mean_mode (voxcpm2.py:218). The
# checkpoint's dit_config carries "mean_mode": false, which is NOT the field
# name VoxCPMDitConfig declares (dit_mean_mode, voxcpm2.py:111) -- so the
# pydantic default applies. Either way the value is False.
MEAN_MODE = bool(d.get("dit_mean_mode", d.get("mean_mode", False)))
assert MEAN_MODE is False, f"unexpected mean_mode={MEAN_MODE}"
BATCH = 2                  # >1 so optimized_scale's per-row reduce is exercised

print(f"n_timesteps={N_TIMESTEPS} cfg_value={CFG_VALUE} temperature={TEMPERATURE} "
      f"sway_sampling_coef={SWAY_SAMPLING_COEF} use_cfg_zero_star={USE_CFG_ZERO_STAR} "
      f"mean_mode={MEAN_MODE} batch={BATCH} "
      f"feat_dim={feat_dim} patch_size={patch_size} hidden_size={dec.hidden_size}",
      file=sys.stderr)

cfm = UnifiedCFM(
    in_channels=feat_dim,
    cfm_params=CfmConfig(**d["cfm_config"]),
    estimator=VoxCPMLocDiTV2(dec, in_channels=feat_dim),
    mean_mode=MEAN_MODE,
)
sd = load_file(ckpt_dir / "model.safetensors")
sub = {k[len("feat_decoder."):]: v for k, v in sd.items() if k.startswith("feat_decoder.")}
cfm.load_state_dict(sub, strict=True)
cfm = cfm.to(torch.float32).eval()
print(f"loaded {len(sub)} tensors", file=sys.stderr)

# --- Facts for the Rust port -------------------------------------------------
print(
    "t_span (unified_cfm.py:66-67): "
    "t_span = torch.linspace(1, 0, n_timesteps + 1, device=mu.device, dtype=mu.dtype); "
    "t_span = t_span + sway_sampling_coef * (torch.cos(torch.pi / 2 * t_span) - 1 + t_span)",
    file=sys.stderr,
)
zero_init_steps = max(1, int((N_TIMESTEPS + 1) * 0.04))
print(
    "warmup (unified_cfm.py:96,98): "
    "zero_init_steps = max(1, int(len(t_span) * 0.04))  -> "
    f"max(1, int({N_TIMESTEPS + 1} * 0.04)) = {zero_init_steps}; "
    "guard is `if use_cfg_zero_star and step <= zero_init_steps: dphi_dt = torch.zeros_like(x)` "
    "with step running 1..len(t_span)-1, so step 1 only for these settings",
    file=sys.stderr,
)
print(
    "cfg combine (unified_cfm.py:128): "
    "dphi_dt = cfg_dphi_dt * st_star + cfg_value * (dphi_dt - cfg_dphi_dt * st_star)",
    file=sys.stderr,
)
print(
    "batch halves (unified_cfm.py:108-118): the doubled batch is split as "
    "[:b] then [b:], and only [:b] receives mu (mu_in[:b] = mu, line 109; "
    "mu_in[b:] stays the zeros it was allocated with, line 104). After "
    "torch.split at line 118 the FIRST output `dphi_dt` is the CONDITIONAL "
    "branch and the SECOND output `cfg_dphi_dt` is the UNCONDITIONAL branch -- "
    "the name cfg_dphi_dt is misleading, it is the negative/uncond velocity",
    file=sys.stderr,
)
print(
    "uncond conditioning (unified_cfm.py:103-115): ONLY mu is zeroed for the "
    "unconditional half. x_in, t_in, dt_in and cond_in are written identically "
    "to both halves (x_in[:b], x_in[b:] = x, x ; t_in[:b], t_in[b:] = "
    "t.unsqueeze(0), t.unsqueeze(0) ; dt_in likewise ; cond_in[:b], cond_in[b:] "
    "= cond, cond). dt_in is then overwritten with zeros because mean_mode is "
    "False (line 113-114: `if not self.mean_mode: dt_in = torch.zeros_like(dt_in)`)",
    file=sys.stderr,
)
print(
    "optimized_scale (unified_cfm.py:79-82): "
    "dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True); "
    "squared_norm = torch.sum(negative_flat**2, dim=1, keepdim=True) + 1e-8; "
    "st_star = dot_product / squared_norm  "
    "-- the 1e-8 epsilon is ADDED TO THE DENOMINATOR SUM, before the divide, "
    "not to the quotient",
    file=sys.stderr,
)
print(
    "euler update (unified_cfm.py:130-134): "
    "x = x - dt * dphi_dt; t = t - dt; "
    "then `if step < len(t_span) - 1: dt = t - t_span[step + 1]` "
    "-- dt is recomputed from the RUNNING t, seeded as t_span[0] - t_span[1] (line 93)",
    file=sys.stderr,
)

tensors = {}
with torch.inference_mode():
    # Seed distinct from make_dit_fixture.py's 5000+i case seeds.
    torch.manual_seed(6000)
    mu = torch.randn(BATCH, 2 * dec.hidden_size, dtype=torch.float32)
    cond = torch.randn(BATCH, feat_dim, patch_size, dtype=torch.float32)
    # z is drawn HERE, not inside forward, and saved -- this is what makes the
    # trajectory reproducible in Rust.
    z = torch.randn(BATCH, feat_dim, patch_size, dtype=torch.float32) * TEMPERATURE

    # Reproduces unified_cfm.py:66-67 exactly.
    t_span = torch.linspace(1, 0, N_TIMESTEPS + 1, dtype=torch.float32)
    t_span = t_span + SWAY_SAMPLING_COEF * (torch.cos(torch.pi / 2 * t_span) - 1 + t_span)

    print("realized t_span:", file=sys.stderr)
    for i, v in enumerate(t_span.tolist()):
        print(f"  t_span[{i:2d}] = {v!r}", file=sys.stderr)

    # Per-step capture: solve_euler returns only sol[-1], so the intermediate
    # states are recovered by hooking the estimator is not enough (the warmup
    # step never calls it). Instead the loop is re-run here in lockstep with
    # unified_cfm.py:93-136, and the final x is asserted equal to solve_euler's
    # own return value -- so the recorded steps are proven to be the real ones.
    x = z.clone()
    t, dt = t_span[0], t_span[0] - t_span[1]
    for step in range(1, len(t_span)):
        if USE_CFG_ZERO_STAR and step <= zero_init_steps:
            dphi_dt = torch.zeros_like(x)
        else:
            b = x.size(0)
            x_in = torch.zeros([2 * b, feat_dim, x.size(2)], dtype=x.dtype)
            mu_in = torch.zeros([2 * b, mu.size(1)], dtype=x.dtype)
            t_in = torch.zeros([2 * b], dtype=x.dtype)
            dt_in = torch.zeros([2 * b], dtype=x.dtype)
            cond_in = torch.zeros([2 * b, feat_dim, cond.size(2)], dtype=x.dtype)
            x_in[:b], x_in[b:] = x, x
            mu_in[:b] = mu
            t_in[:b], t_in[b:] = t.unsqueeze(0), t.unsqueeze(0)
            dt_in[:b], dt_in[b:] = dt.unsqueeze(0), dt.unsqueeze(0)
            if not MEAN_MODE:
                dt_in = torch.zeros_like(dt_in)
            cond_in[:b], cond_in[b:] = cond, cond

            dphi_dt = cfm.estimator(x_in, mu_in, t_in, cond_in, dt_in)
            dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [b, b], dim=0)

            if USE_CFG_ZERO_STAR:
                st_star = cfm.optimized_scale(dphi_dt.view(b, -1), cfg_dphi_dt.view(b, -1))
                st_star = st_star.view(b, *([1] * (len(dphi_dt.shape) - 1)))
            else:
                st_star = 1.0

            dphi_dt = cfg_dphi_dt * st_star + CFG_VALUE * (dphi_dt - cfg_dphi_dt * st_star)

        x = x - dt * dphi_dt
        t = t - dt
        tensors[f"step{step - 1}_x"] = x.clone().contiguous()
        print(f"step{step - 1}: t={float(t):.8f} dt={float(dt):.8f} "
              f"x range [{x.min():.4f}, {x.max():.4f}]"
              + ("  (zero-velocity warmup)" if step <= zero_init_steps else ""),
              file=sys.stderr)
        if step < len(t_span) - 1:
            dt = t - t_span[step + 1]

    # Ground truth from the library itself, every argument explicit.
    out_y = cfm.solve_euler(
        x=z,
        t_span=t_span,
        mu=mu,
        cond=cond,
        cfg_value=CFG_VALUE,
        use_cfg_zero_star=USE_CFG_ZERO_STAR,
    )
    assert torch.equal(out_y, x), "replayed trajectory diverged from solve_euler"

    tensors["z"] = z.contiguous()
    tensors["mu"] = mu.contiguous()
    tensors["cond"] = cond.contiguous()
    tensors["t_span"] = t_span.contiguous()
    tensors["out"] = out_y.clone().contiguous()

print(f"z{tuple(z.shape)} mu{tuple(mu.shape)} cond{tuple(cond.shape)} "
      f"t_span{tuple(t_span.shape)} -> out{tuple(out_y.shape)}  "
      f"range [{out_y.min():.4f}, {out_y.max():.4f}]", file=sys.stderr)

save_file(tensors, str(out), metadata={
    "dtype": "f32",
    "feat_dim": str(feat_dim),
    "patch_size": str(patch_size),
    "hidden_size": str(dec.hidden_size),
    "batch": str(BATCH),
    "n_timesteps": str(N_TIMESTEPS),
    "cfg_value": str(CFG_VALUE),
    "temperature": str(TEMPERATURE),
    "sway_sampling_coef": str(SWAY_SAMPLING_COEF),
    "use_cfg_zero_star": str(USE_CFG_ZERO_STAR),
    "mean_mode": str(MEAN_MODE),
    "zero_init_steps": str(zero_init_steps),
})
print(f"wrote {out}", file=sys.stderr)
