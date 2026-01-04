#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""SeriousGame/edt_scenario_generator.py

This repository's EDT evaluation is *scenario-level*:

- The agent acts like a manager and outputs a minimal schema once per episode.
- We apply that schema to a template scenario (e.g. `interactive`) and
  materialize a temporary BPTK repo root (contains `scenarios/` and a
  `simulation_models/` package).
- A BPTK server is started pointing at that temp root so the episode runs with
  the newly generated scenario.

Minimal schema (JSON) expected from agent:

{
  "C": <int>,            # number of consultants to keep (0..C_max)
  "R": <float>,          # revenue_risk_level (0..1)
  "P": [                # per-project decisions, aligned with template order
    0 | [1, start_step, deadline_step],
    ...
  ]
}

Notes
- We intentionally use *step indices* for start/deadline to avoid floating-point
  equality issues in the ABM (`time == start_time`).
- We do NOT let the agent modify salary, workplace costs, billing_rate, etc.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _safe_int(x: Any, default: int) -> int:
    try:
        return int(float(x))
    except Exception:
        return int(default)


def _safe_float(x: Any, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def load_scenario_document(template_repo_root: str, scenario_file: str) -> Dict[str, Any]:
    """Load a scenario JSON document from `<template_repo_root>/scenarios/<scenario_file>`."""
    p = Path(template_repo_root).resolve() / "scenarios" / scenario_file
    if not p.exists():
        raise FileNotFoundError(f"Scenario file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_sm_document(doc: Dict[str, Any], scenario_manager: str) -> Dict[str, Any]:
    """Extract scenario manager document, e.g. doc['smEDT']."""
    if scenario_manager not in doc:
        raise KeyError(f"Scenario manager '{scenario_manager}' not found in scenario doc keys={list(doc.keys())[:10]}")
    sm_doc = doc[scenario_manager]
    if not isinstance(sm_doc, dict):
        raise TypeError(f"Scenario manager doc must be dict, got {type(sm_doc)}")
    return sm_doc


def extract_scenario(sm_doc: Dict[str, Any], scenario_key: str) -> Dict[str, Any]:
    """Extract scenario body, e.g. sm_doc['scenarios']['interactive']."""
    scenarios = sm_doc.get("scenarios", {})
    if scenario_key not in scenarios:
        raise KeyError(f"Scenario '{scenario_key}' not found. Available={list(scenarios.keys())[:10]}")
    scen = scenarios[scenario_key]
    if not isinstance(scen, dict):
        raise TypeError(f"Scenario must be dict, got {type(scen)}")
    return scen


def _agent_props(agent_obj: Dict[str, Any]) -> Dict[str, Any]:
    return (agent_obj or {}).get("properties", {}) or {}


def _prop_value(props: Dict[str, Any], key: str, default: Any = None) -> Any:
    if key not in props:
        return default
    node = props.get(key) or {}
    if isinstance(node, dict) and "value" in node:
        return node.get("value")
    return default


def summarize_base_scenario(scenario: Dict[str, Any], max_steps: int = 96) -> Dict[str, Any]:
    props = scenario.get("properties", {}) or {}
    runspecs = scenario.get("runspecs", {}) or {}

    def _get_num(container, key, default):
        node = container.get(key, None)
        if isinstance(node, dict) and "value" in node:
            return node["value"]
        elif node is not None:
            return node
        return default

    # 1) prefer runspecs
    dt = float(_get_num(runspecs, "dt", _get_num(props, "dt", 0.25)))
    starttime = float(_get_num(runspecs, "starttime", _get_num(props, "starttime", 0.0)))
    stoptime = float(_get_num(runspecs, "stoptime", _get_num(props, "stoptime", starttime)))

    # --- collect template (non-follow-on) projects first
    agents = list(scenario.get("agents", []) or [])
    projects = []
    for a in agents:
        if isinstance(a, dict) and str(a.get("name", "")).lower() == "project":
            pprops = (a.get("properties", {}) or {})
            is_follow_on = bool((pprops.get("is_follow_on", {}) or {}).get("value", False)) if isinstance(pprops.get("is_follow_on", {}), dict) else False
            if not is_follow_on:
                projects.append(a)

    # 2) fallback: derive start/stoptime from projects if timeline invalid
    if dt <= 0:
        dt = 0.25
    if stoptime <= starttime:
        # try derive from projects
        p_starts = []
        p_deadlines = []
        for p in projects:
            pprops = p.get("properties", {}) or {}
            st = (pprops.get("start_time", {}) or {}).get("value", None)
            dl = (pprops.get("deadline", {}) or {}).get("value", None)
            if isinstance(st, (int, float)): p_starts.append(float(st))
            if isinstance(dl, (int, float)): p_deadlines.append(float(dl))
        if p_starts and p_deadlines:
            starttime = min(p_starts)
            stoptime = max(p_deadlines)
        else:
            # last resort: align to evaluator max_steps
            stoptime = starttime + dt * max(1, int(max_steps))

    # 3) horizon in steps; clamp to evaluator horizon
    horizon_steps = int(round((stoptime - starttime) / dt))
    horizon_steps = max(1, horizon_steps)
    horizon_steps = min(horizon_steps, max(1, int(max_steps)))  # <= max_steps

    revenue_risk_level = _safe_float(_prop_value(props, "revenue_risk_level", 1.0), 1.0)

    agents = list(scenario.get("agents", []) or [])

    consultants = []
    projects = []
    for a in agents:
        if not isinstance(a, dict):
            continue
        name = str(a.get("name", "")).lower()
        if name == "consultant":
            consultants.append(a)
        elif name == "project":
            # template projects only; follow-ons are generated at runtime
            pprops = _agent_props(a)
            is_follow_on = _prop_value(pprops, "is_follow_on", False)
            if bool(is_follow_on):
                continue
            projects.append(a)

    proj_summary = []
    for p in projects:
        pprops = _agent_props(p)
        proj_summary.append(
            {
                "name": str(_prop_value(pprops, "name", "project")),
                "start_time": _safe_float(_prop_value(pprops, "start_time", starttime), starttime),
                "deadline": _safe_float(_prop_value(pprops, "deadline", stoptime), stoptime),
                "consultants": _safe_int(_prop_value(pprops, "consultants", 1), 1),
                "contracted_effort": _safe_float(_prop_value(pprops, "contracted_effort", 0.0), 0.0),
                "contracted_probability": _safe_float(_prop_value(pprops, "contracted_probability", 1.0), 1.0),
                "extension_probability": _safe_float(_prop_value(pprops, "extension_probability", 0.0), 0.0),
                "follow_on_probability": _safe_float(_prop_value(pprops, "follow_on_probability", 0.0), 0.0),
                "billing_rate": _safe_float(_prop_value(pprops, "billing_rate", 0.0), 0.0),
            }
        )

    return {
        "starttime": starttime,
        "stoptime": stoptime,
        "dt": dt,
        "horizon_steps": horizon_steps,
        "revenue_risk_level": revenue_risk_level,
        "consultants_max": len(consultants),
        "projects": proj_summary,
    }


def normalize_schema(schema: Dict[str, Any], base_summary: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize/clamp the minimal schema."""
    cmax = int(base_summary.get("consultants_max", 0) or 0)
    horizon = int(base_summary.get("horizon_steps", 0) or 0)
    default_r = float(base_summary.get("revenue_risk_level", 1.0) or 1.0)

    out: Dict[str, Any] = {}
    out["C"] = max(0, min(cmax, _safe_int(schema.get("C", cmax), cmax)))
    r = _safe_float(schema.get("R", default_r), default_r)
    out["R"] = float(max(0.0, min(1.0, r)))

    # default P: enable all projects with their original steps
    starttime = float(base_summary.get("starttime", 0.0) or 0.0)
    dt = float(base_summary.get("dt", 0.25) or 0.25)
    proj = list(base_summary.get("projects", []) or [])
    default_P: List[Any] = []
    for p in proj:
        s_step = 0
        d_step = horizon
        if dt > 0:
            s_step = int(round((float(p.get("start_time", starttime)) - starttime) / dt))
            d_step = int(round((float(p.get("deadline", starttime)) - starttime) / dt))
        s_step = max(0, min(horizon, s_step))
        d_step = max(0, min(horizon, d_step))
        if d_step < s_step:
            d_step = s_step
        default_P.append([1, s_step, d_step])

    P = schema.get("P", None)
    if not isinstance(P, list) or len(P) != len(proj):
        P = default_P

    P_norm: List[Any] = []
    for i, dec in enumerate(P):
        if dec == 0 or dec is False or dec is None:
            P_norm.append(0)
            continue
        if isinstance(dec, list) and len(dec) == 3:
            flag = _safe_int(dec[0], 0)
            if flag != 1:
                P_norm.append(0)
                continue
            s = max(0, min(horizon, _safe_int(dec[1], default_P[i][1])))
            d = max(0, min(horizon, _safe_int(dec[2], default_P[i][2])))
            if d < s:
                d = s
            P_norm.append([1, s, d])
            continue
        P_norm.append(default_P[i])

    out["P"] = P_norm
    return out


def apply_schema_to_scenario(
    base_scenario: Dict[str, Any],
    schema: Dict[str, Any],
    base_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """Apply normalized schema to a deep-copied scenario."""
    scenario = deepcopy(base_scenario)

    # update scenario-level properties
    sprops = scenario.get("properties", {}) or {}
    if "revenue_risk_level" in sprops and isinstance(sprops["revenue_risk_level"], dict):
        sprops["revenue_risk_level"]["value"] = float(schema["R"])
    else:
        sprops["revenue_risk_level"] = {"type": "Double", "value": float(schema["R"])}
    scenario["properties"] = sprops

    agents = list(scenario.get("agents", []) or [])

    # split agents
    consultants: List[Dict[str, Any]] = []
    projects: List[Dict[str, Any]] = []
    others: List[Dict[str, Any]] = []
    for a in agents:
        if not isinstance(a, dict):
            continue
        name = str(a.get("name", "")).lower()
        if name == "consultant":
            consultants.append(a)
        elif name == "project":
            pprops = _agent_props(a)
            is_follow_on = _prop_value(pprops, "is_follow_on", False)
            if bool(is_follow_on):
                # keep follow-on template agents if any (usually none)
                others.append(a)
            else:
                projects.append(a)
        else:
            others.append(a)

    # consultants: keep only first C
    C = int(schema["C"])
    consultants = consultants[:C]

    # projects: enable/disable and adjust schedule
    P = list(schema.get("P", []) or [])
    starttime = float(base_summary.get("starttime", 0.0) or 0.0)
    stoptime = float(base_summary.get("stoptime", 0.0) or 0.0)
    dt = float(base_summary.get("dt", 0.25) or 0.25)

    new_projects: List[Dict[str, Any]] = []
    for i, proj_obj in enumerate(projects):
        dec = P[i] if i < len(P) else [1, 0, int(base_summary.get("horizon_steps", 0) or 0)]
        if dec == 0:
            continue
        # enabled
        s_step = _safe_int(dec[1], 0)
        d_step = _safe_int(dec[2], s_step)
        if d_step < s_step:
            d_step = s_step

        # convert to time (aligned)
        st = max(min(stoptime, s_step), starttime)
        dl = max(min(stoptime, d_step), starttime)

        pprops = proj_obj.get("properties", {}) or {}
        if "start_time" in pprops and isinstance(pprops["start_time"], dict):
            pprops["start_time"]["value"] = float(st)
        else:
            pprops["start_time"] = {"type": "Double", "value": float(st)}

        if "deadline" in pprops and isinstance(pprops["deadline"], dict):
            pprops["deadline"]["value"] = float(dl)
        else:
            pprops["deadline"] = {"type": "Double", "value": float(dl)}

        proj_obj["properties"] = pprops
        new_projects.append(proj_obj)

    scenario["agents"] = consultants + new_projects + others
    return scenario


def _copy_or_symlink(src: Path, dst: Path) -> None:
    """Prefer symlink to avoid heavy copies; fallback to copytree/copy2."""
    if dst.exists():
        return
    try:
        os.symlink(src.as_posix(), dst.as_posix(), target_is_directory=src.is_dir())
        return
    except Exception:
        pass
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def materialize_temp_bptk_repo(
    *,
    template_repo_root: str,
    sm_doc: Dict[str, Any],
    new_scenario_key: str,
    new_scenario: Dict[str, Any],
    out_scenario_file: str,
    keep_temp: bool = False,
) -> Tuple[str, str]:
    """Create a temporary BPTK repo root for one episode.

    The BPTK server is started with `--repo-root <temp_root>`, which must include:
      - ./scenarios/<out_scenario_file>
      - ./simulation_models (importable)

    We symlink/copy from template repo root where possible.
    """
    template_root = Path(template_repo_root).resolve()
    if not (template_root / "scenarios").exists():
        raise FileNotFoundError(f"Template repo root must contain 'scenarios/': {template_root}")
    if not (template_root / "simulation_models").exists():
        raise FileNotFoundError(f"Template repo root must contain 'simulation_models/': {template_root}")

    if keep_temp:
        temp_root = Path(tempfile.mkdtemp(prefix="edt_bptk_repo_")).resolve()
    else:
        temp_root = Path(tempfile.mkdtemp(prefix="edt_bptk_repo_")).resolve()

    # link/copy simulation_models
    _copy_or_symlink(template_root / "simulation_models", temp_root / "simulation_models")

    # write scenarios
    scen_dir = temp_root / "scenarios"
    scen_dir.mkdir(parents=True, exist_ok=True)

    out_doc = {
        # keep original meta but replace scenarios with the new one
        "type": sm_doc.get("type", "abm"),
        "name": sm_doc.get("name", "EDT"),
        "model": sm_doc.get("model", ""),
        "scenarios": {new_scenario_key: new_scenario},
    }
    full = {"smEDT": out_doc} if "scenarios" in sm_doc else {"smEDT": out_doc}

    out_path = scen_dir / out_scenario_file
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(full, f, ensure_ascii=False, indent=2)

    return temp_root.as_posix(), out_path.as_posix()


def generate_temp_repo_from_schema(
    *,
    template_repo_root: str,
    scenario_manager: str,
    scenario_file: str,
    base_scenario_key: str,
    schema: Dict[str, Any],
    new_scenario_key: str,
    out_scenario_file: str = "generated.json",
    keep_temp: bool = False,
) -> Dict[str, Any]:
    """Convenience wrapper used by evaluation.

    Returns a dict:
    {
      "temp_repo_root": <str>,
      "scenario_file": <str>,
      "scenario_key": <str>,
      "base_summary": <dict>,
      "schema": <normalized schema>
    }
    """
    doc = load_scenario_document(template_repo_root, scenario_file)
    sm_doc = extract_sm_document(doc, scenario_manager)
    base_scenario = extract_scenario(sm_doc, base_scenario_key)

    base_summary = summarize_base_scenario(base_scenario)
    schema_norm = normalize_schema(schema, base_summary)
    new_scenario = apply_schema_to_scenario(base_scenario, schema_norm, base_summary)

    temp_repo_root, scenario_path = materialize_temp_bptk_repo(
        template_repo_root=template_repo_root,
        sm_doc=sm_doc,
        new_scenario_key=new_scenario_key,
        new_scenario=new_scenario,
        out_scenario_file=out_scenario_file,
        keep_temp=keep_temp,
    )

    return {
        "temp_repo_root": temp_repo_root,
        "scenario_path": scenario_path,
        "scenario_file": out_scenario_file,
        "scenario_key": new_scenario_key,
        "base_summary": base_summary,
        "schema": schema_norm,
    }
