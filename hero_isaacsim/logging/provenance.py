"""Record the code, corpus and runtime settings associated with a HERO run."""
from __future__ import annotations

import hashlib

import json

import os

import platform

import socket

import subprocess

import sys

from datetime import datetime, timezone

from pathlib import Path

from typing import Any, Mapping

PLANT_CONTRACT = "sonic_g1_model_12_dex_v1"

ACTION_CONTRACT_HERO = "hero_residual_upper_v1"  # PPODual arms: residual upper-body action after per-joint scaling

OBS_HISTORY_LAYOUT_TRAIN = "term_major_holosoma"  # what the actor sees inside holosoma

HERO_CORPUS_MANIFEST_ENV = "HERO_CORPUS_MANIFEST"  # same variable PPODual._find_corpus_manifest reads

MB_PER_CORPUS_HOUR = 379.0

_SYNCED = False

def _run(cmd: list[str], cwd: Path | None = None) -> str | None:
    try:
        out = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True, timeout=10, check=False)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:  # noqa: BLE001 - best effort only
        return None

def repo_root() -> Path:
    """Resolve the current HERO checkout or installed package root."""
    from hero_isaacsim.paths import RELEASE_ROOT
    return RELEASE_ROOT

def git_provenance(root: Path | None = None) -> dict[str, Any]:
    root = root or repo_root()
    sha = _run(["git", "rev-parse", "HEAD"], root)
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], root)
    status = _run(["git", "status", "--porcelain", "--untracked-files=no"], root)
    return {
        "git_sha": sha,
        "git_branch": branch,
        "git_dirty": (bool(status) if status is not None else None),
        "repo_root": str(root),
    }

def vendored_holosoma_provenance(root: Path | None = None) -> dict[str, Any]:
    """Parse VENDOR.md for the source commit lines (best effort)."""
    root = root or repo_root()
    vendor_md = root / "third_party" / "holosoma" / "VENDOR.md"
    info: dict[str, Any] = {"holosoma_vendor_md": str(vendor_md) if vendor_md.exists() else None}
    if vendor_md.exists():
        text = vendor_md.read_text(errors="ignore")
        for token in ("8736d68c", "ebf4e003"):
            if token in text:
                info.setdefault("holosoma_source_commits", []).append(token)
    return info

def file_sha256(path: str | os.PathLike[str]) -> str | None:
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def corpus_provenance(manifest_path: str | os.PathLike[str] | None) -> dict[str, Any]:
    """sha256 + summary of CORPUS_MANIFEST.json (source weights, clip counts, low-posture shares) if present."""
    if manifest_path is None:
        return {"corpus_manifest": None}
    p = Path(manifest_path)
    out: dict[str, Any] = {"corpus_manifest": str(p), "corpus_manifest_sha256": file_sha256(p)}
    try:
        data = json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return out
    summary = data.get("summary") if isinstance(data, dict) else None
    if isinstance(summary, Mapping):
        for k in ("source_weights", "clip_counts", "episode_share", "low_posture_clip_share", "low_posture_episode_share", "num_clips", "total_hours"):
            if k in summary:
                out[f"corpus_{k}"] = summary[k]
        hours = summary.get("total_hours", summary.get("hours"))
        if isinstance(hours, (int, float)):
            out["corpus_total_hours"] = float(hours)
            out["corpus_est_ram_gb_per_rank"] = round(float(hours) * MB_PER_CORPUS_HOUR / 1024.0, 2)
    elif isinstance(data, dict) and "source_weights" in data:
        out["corpus_source_weights"] = data["source_weights"]
    if isinstance(data, dict):
        for k in ("subset", "created_utc", "schema"):
            if data.get(k) is not None:
                out[f"corpus_manifest_{k}"] = data[k]
        census = data.get("object_channel_census")
        if census is not None:
            out["corpus_object_channel_census"] = census
    return out

def compute_provenance() -> dict[str, Any]:
    gpus = os.environ.get("CUDA_VISIBLE_DEVICES")
    names = _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    world_size = os.environ.get("WORLD_SIZE")
    info: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "cuda_visible_devices": gpus,
        "gpu_names": names.splitlines() if names else None,
        "world_size": int(world_size) if world_size and world_size.isdigit() else 1,
        "local_rank": os.environ.get("LOCAL_RANK"),
        "usd_conversion_slot": os.environ.get("HOLOSOMA_USD_CONVERSION_SLOT"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    try:
        import torch  # noqa: WPS433

        info["torch"] = torch.__version__
    except Exception:  # noqa: BLE001
        pass
    return info

def host_provenance() -> dict[str, Any]:
    """Host RAM (GB, total / available) + CPU count.  Linux ``/proc/meminfo``; ``os.sysconf`` elsewhere (total only)."""
    info: dict[str, Any] = {"host_cpu_count": os.cpu_count(), "host_ram_total_gb": None, "host_ram_available_gb": None}
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        try:
            for line in meminfo.read_text().splitlines():
                key, _, rest = line.partition(":")
                if key in ("MemTotal", "MemAvailable"):
                    kb = float(rest.strip().split()[0])
                    info["host_ram_total_gb" if key == "MemTotal" else "host_ram_available_gb"] = round(kb / 1024.0 / 1024.0, 1)
        except Exception:  # noqa: BLE001
            pass
    if info["host_ram_total_gb"] is None:
        try:
            info["host_ram_total_gb"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024.0**3, 1)
        except (ValueError, OSError, AttributeError):
            pass
    return info

def _jsonable(value: Any) -> Any:
    """Keep only plain JSON-friendly values (config params may carry tuples / numpy scalars)."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:  # noqa: BLE001
            pass
    return str(value)

def dr_ranges_from_randomization(rand: Any) -> dict[str, Any]:
    """DR table of a ``RandomizationManagerCfg`` (kp/kd, delay, link / base / EE mass, friction, pushes, per-episode
    EE terms, object terms).  Unknown term layouts degrade to their raw params."""
    out: dict[str, Any] = {}
    setup = dict(getattr(rand, "setup_terms", {}) or {})
    reset = dict(getattr(rand, "reset_terms", {}) or {})

    def params(term_name: str, table: Mapping[str, Any]) -> dict[str, Any]:
        term = table.get(term_name)
        return dict(getattr(term, "params", {}) or {}) if term is not None else {}

    act = params("actuator_randomizer_state", setup)
    if act:
        out["pd_gain"] = {"enabled": act.get("enable_pd_gain"), "kp_range": _jsonable(act.get("kp_range")), "kd_range": _jsonable(act.get("kd_range"))}
    dly = params("setup_action_delay_buffers", setup)
    if dly:
        out["action_delay"] = {"enabled": dly.get("enabled"), "ctrl_delay_step_range": _jsonable(dly.get("ctrl_delay_step_range"))}
    mass = params("mass_randomizer", setup)
    if mass:
        out["link_mass"] = {"enabled": mass.get("enable_link_mass"), "link_mass_range": _jsonable(mass.get("link_mass_range"))}
        out["base_added_mass"] = {"enabled": mass.get("enable_base_mass"), "added_mass_range": _jsonable(mass.get("added_mass_range"))}
    fric = params("randomize_robot_rigid_body_material_startup", setup)
    if fric:
        out["friction"] = {
            "static_friction_range": _jsonable(fric.get("static_friction_range")),
            "dynamic_friction_range": _jsonable(fric.get("dynamic_friction_range")),
            "restitution_range": _jsonable(fric.get("restitution_range")),
        }
    push = params("push_randomizer_state", setup)
    if push:
        out["push"] = _jsonable(push)
    com = params("randomize_base_com_startup", setup)
    if com:
        out["base_com"] = _jsonable(com)
    for name, term in setup.items():
        func = str(getattr(term, "func", ""))
        if "randomize_ee_mass_startup" in func:
            out["ee_mass_startup"] = _jsonable(dict(term.params or {}))
        elif "object" in name or "object" in func:
            out.setdefault("object_setup_terms", {})[name] = _jsonable(dict(getattr(term, "params", {}) or {}))
    for name, term in reset.items():
        func = str(getattr(term, "func", ""))
        if "randomize_ee_mass_reset" in func:
            out["ee_mass_reset"] = _jsonable(dict(term.params or {}))
        elif "randomize_ee_com_reset" in func:
            out["ee_com_reset"] = _jsonable(dict(term.params or {}))
    out["setup_term_names"] = sorted(setup)
    out["reset_term_names"] = sorted(reset)
    out["step_term_names"] = sorted(dict(getattr(rand, "step_terms", {}) or {}))
    return out

def obs_noise_table(observation: Any) -> dict[str, Any]:
    """Observation-noise table of an ``ObservationManagerCfg``.

    ``groups[<g>].enable_noise`` is holosoma's generic noise switch -- it is applied in evaluation too
    (observation/manager.py), so HERO keeps it False; training-only noise lives in per-term ``noise_*``
    params gated by ``training_noise_active`` .  ``any_in_term_noise`` is True when any such
    param is non-zero; ``ungated_group_noise`` lists groups with ``enable_noise=True`` (should be empty)."""
    groups: dict[str, Any] = {}
    terms: dict[str, Any] = {}
    any_noise = False
    for gname, group in dict(getattr(observation, "groups", {}) or {}).items():
        groups[gname] = {"enable_noise": bool(getattr(group, "enable_noise", False)), "history_length": int(getattr(group, "history_length", 1))}
        for tname, term in dict(getattr(group, "terms", {}) or {}).items():
            entry: dict[str, Any] = {}
            generic = float(getattr(term, "noise", 0.0) or 0.0)
            if generic:
                entry["noise"] = generic
            for k, v in dict(getattr(term, "params", {}) or {}).items():
                if str(k).startswith("noise"):
                    entry[str(k)] = _jsonable(v)
            if entry:
                terms[f"{gname}/{tname}"] = entry
                any_noise = any_noise or any(bool(v) for v in entry.values())
    return {
        "groups": groups,
        "terms": terms,
        "any_in_term_noise": any_noise,
        "ungated_group_noise": [g for g, v in groups.items() if v["enable_noise"]],
        "gate": "hero_isaacsim.utils.training.training_noise_active (env.is_evaluating)",
    }

def preset_provenance(preset: Any = None) -> dict[str, Any]:
    from hero_isaacsim.config_values.experiment import DEFAULTS, observation_contract
    if isinstance(preset, str):
        preset = DEFAULTS.get(preset)
    if preset is None:
        return {}
    return {"preset_name": preset.training.name, "obs_layout": observation_contract(preset),
            "obs_noise": obs_noise_table(preset.observation),
            "dr_ranges": dr_ranges_from_randomization(preset.randomization)}

def runtime_robot_asset_provenance(env: Any) -> dict[str, Any]:
    """Describe the collision asset prepared for the simulator importer."""
    from holosoma.utils.robot_asset_metadata import runtime_robot_asset_metadata

    return runtime_robot_asset_metadata(getattr(env, "simulator", None))

def collect_provenance(
    *,
    arm: str,
    corpus_manifest: str | os.PathLike[str] | None,
    seed: int,
    global_num_envs: int | None = None,
    envs_per_rank: int | None = None,
    network_sizes: Mapping[str, Any] | None = None,
    dr_ranges: Mapping[str, Any] | None = None,
    obs_layout: Mapping[str, Any] | None = None,
    action_contract: str = ACTION_CONTRACT_HERO,
    plant_contract: str = PLANT_CONTRACT,
    extra: Mapping[str, Any] | None = None,
    preset: Any = None,
) -> dict[str, Any]:
    """Assemble the provenance block.  ``preset`` = preset key / ExperimentConfig / None (-> ``$HERO_PRESET``); its
    block lands under ``prov["preset"]`` with the headline fields (``preset_name``, ``motion_storage_device``,
    ``sampler_clip_max_probability``, ``sampler_uniform_ratio``, ``randomization_preset``, ``obs_noise``) mirrored at
    top level; ``dr_ranges`` falls back to the preset's table when the caller passes none.  ``corpus_manifest=None``
    falls back to ``$HERO_CORPUS_MANIFEST``."""
    prov: dict[str, Any] = {"arm": arm, "seed": int(seed), "plant_contract": plant_contract, "action_contract": action_contract}
    prov.update(git_provenance())
    prov.update(vendored_holosoma_provenance())
    if corpus_manifest is None:
        env_manifest = os.environ.get(HERO_CORPUS_MANIFEST_ENV)
        if env_manifest and Path(env_manifest).is_file():
            corpus_manifest = env_manifest
    prov.update(corpus_provenance(corpus_manifest))
    prov.update(compute_provenance())
    prov.update(host_provenance())
    try:
        preset_block = preset_provenance(preset)
    except Exception:  # noqa: BLE001 - never fail a run over provenance
        preset_block = {}
    if preset_block:
        prov["preset"] = preset_block
        for key in ("preset_name", "motion_storage_device", "sampler_clip_max_probability", "sampler_uniform_ratio", "randomization_preset", "obs_noise",
                    "terrain_preset", "terrain_layout_sha", "terrain_class_fracs", "terrain_mesh_type"):
            if key in preset_block:
                prov[key] = preset_block[key]
        if not dr_ranges and preset_block.get("dr_ranges"):
            dr_ranges = preset_block["dr_ranges"]
    if global_num_envs is not None:
        prov["global_num_envs"] = int(global_num_envs)
    if envs_per_rank is not None:
        prov["envs_per_rank"] = int(envs_per_rank)
    if network_sizes:
        prov["network_sizes"] = dict(network_sizes)
    if dr_ranges:
        prov["dr_ranges"] = dict(dr_ranges)
    if obs_layout:
        prov["obs_layout"] = dict(obs_layout)
    if extra:
        prov.update(dict(extra))
    return prov

def write_provenance_json(prov: Mapping[str, Any], log_dir: str | os.PathLike[str]) -> Path:
    p = Path(log_dir) / "provenance.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dict(prov), indent=2, default=str))
    return p

def sync_to_wandb(prov: Mapping[str, Any]) -> bool:
    """wandb.config.update(prov) if a run is active. Returns True if synced."""
    try:
        import wandb  # noqa: WPS433
    except Exception:  # noqa: BLE001
        return False
    run = getattr(wandb, "run", None)
    if run is None:
        return False
    try:
        run.config.update({"provenance": dict(prov)}, allow_val_change=True)
        return True
    except Exception:  # noqa: BLE001
        return False

def sync_once(prov: Mapping[str, Any], log_dir: str | os.PathLike[str] | None = None) -> bool:
    """Idempotent per process: write provenance.json (if log_dir) and push to wandb once."""
    global _SYNCED
    if _SYNCED:
        return False
    if log_dir is not None:
        try:
            write_provenance_json(prov, log_dir)
        except Exception:  # noqa: BLE001
            pass
    ok = sync_to_wandb(prov)
    _SYNCED = True
    return ok
