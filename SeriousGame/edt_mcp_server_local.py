from __future__ import annotations

"""edt_mcp_server_local.py

MCP server for the Enterprise Digital Twin (EDT).

This server exposes MCP tools that forward to a running BptkServer (REST API).
Provide the BptkServer base URL via environment variable:
  EDT_BPTK_BASE_URL=http://localhost:5000

Or via CLI override:
  python edt_mcp_server_local.py --base-url http://localhost:5000
"""
import json
import datetime
import os
import sys

# Optional CLI override for base URL (so you don't rely on env vars).
if "--base-url" in sys.argv:
    try:
        i = sys.argv.index("--base-url")
        os.environ["EDT_BPTK_BASE_URL"] = sys.argv[i + 1]
        del sys.argv[i : i + 2]  # remove so FastMCP doesn't see it
    except Exception:
        pass


import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Prefer official MCP SDK; fallback to fastmcp if you use that.
try:
    from mcp.server.fastmcp import FastMCP  # official MCP Python SDK style
except Exception:  # pragma: no cover
    from fastmcp import FastMCP  # community FastMCP

import requests

mcp = FastMCP("edt-mcp")
_MAX_TEXT = int(os.getenv("EDT_MCP_MAX_LOG_CHARS", "8000"))

_DEBUG = 0
_LOGFILE = "/mnt/data/BIT_public/FinAgent_bench-master/edt_trace.jsonl"


def _dbg(event: str, **fields):
    """Write one JSON line to stderr and optional file. Never write to stdout."""
    if not _DEBUG and not _LOGFILE:
        return
    rec = {
        "ts": datetime.datetime.now().isoformat(timespec="milliseconds"),
        "event": event,
        **fields,
    }
    line = json.dumps(rec, ensure_ascii=False, default=str)

    # stderr for interactive debugging
    if _DEBUG:
        print(line, file=sys.stderr, flush=True)

    # optional file (jsonl)
    if _LOGFILE:
        try:
            with open(_LOGFILE, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass


def _clean_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def _post_with_key_fallback(url: str, endpoint: str, payload: Dict[str, Any], headers: Dict[str, str], timeout: float = 15.0) -> requests.Response:
    """POST helper that retries with common key variants.

    BptkServer historically accepted both camelCase and snake_case for some fields.
    For maximum compatibility across versions, we attempt a couple of variants when
    we see a 400/500 class error.
    """

    variants: List[Dict[str, Any]] = [payload]

    # common renames: scenarioManager <-> scenario_manager, scenarioManagers <-> scenario_managers
    if "scenarioManager" in payload and "scenario_manager" not in payload:
        v = dict(payload)
        v["scenario_manager"] = v["scenarioManager"]
        variants.append(v)
    if "scenario_manager" in payload and "scenarioManager" not in payload:
        v = dict(payload)
        v["scenarioManager"] = v["scenario_manager"]
        variants.append(v)
    if "scenarioManagers" in payload and "scenario_managers" not in payload:
        v = dict(payload)
        v["scenario_managers"] = v["scenarioManagers"]
        variants.append(v)
    if "scenario_managers" in payload and "scenarioManagers" not in payload:
        v = dict(payload)
        v["scenarioManagers"] = v["scenario_managers"]
        variants.append(v)

    last_resp: Optional[requests.Response] = None
    for p in variants:
        resp = requests.post(f"{url}{endpoint}", json=p, headers=headers, timeout=timeout)
        last_resp = resp
        if resp.status_code < 400:
            return resp
        # Retry only on common schema errors.
        # If it's an auth error, not found, etc., don't spam retries.
        if resp.status_code in (401, 403, 404):
            return resp
    assert last_resp is not None
    return last_resp


def _safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _flatten_numeric_leaves(obj: Any, prefix: str = "") -> Iterable[Tuple[str, float]]:
    """
    Flatten a nested JSON object and yield (flat_key, numeric_value).

    Heuristic:
      - If a dict looks like { "<time>": <number>, ... } we take the last time (max key as float if possible).
      - Otherwise we recurse.
    """
    if isinstance(obj, dict):
        if obj:
            time_keys: List[Tuple[Optional[float], str]] = []
            all_time_like = True
            for k in obj.keys():
                fk = _safe_float(k)
                time_keys.append((fk, str(k)))
                if fk is None:
                    all_time_like = False

            if all_time_like:
                values_ok = True
                for v in obj.values():
                    if _safe_float(v) is None:
                        values_ok = False
                        break
                if values_ok:
                    last = max(time_keys, key=lambda t: t[0] if t[0] is not None else float("-inf"))[1]
                    val = _safe_float(obj[last])
                    if val is not None:
                        yield (prefix.rstrip("."), val)
                    return

        for k, v in obj.items():
            yield from _flatten_numeric_leaves(v, f"{prefix}{k}.")
        return

    if isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _flatten_numeric_leaves(v, f"{prefix}{i}.")
        return

    val = _safe_float(obj)
    if val is not None:
        yield (prefix.rstrip("."), val)


def _shorten_metric_key(k: str) -> str:
    """
    Shorten long flat metric keys coming from BptkServer.
    """
    parts = k.split(".")
    # fallback: keep last two segments to avoid collapsing everything to "total"
    return parts[-2] if len(parts) >= 2 else k


def _default_abm_selection() -> Dict[str, List[str]]:
    """A conservative default for ABM sessions.

    Many ABM models (including the EDT tutorial) expose business KPIs as properties
    of a bookkeeping/controller agent (often named "controlling").

    BptkServer aborts begin-session if neither equations nor (agents + metrics)
    are provided. This default keeps sessions alive even if the caller did not
    specify what to collect.
    """

    # NOTE: These names are best-effort. If the underlying model uses different
    # property names, you'll still get a running session, but reward/KPIs may be 0.
    # In that case, use edt-list-equations / add an agent-inspection endpoint and
    # override via config.
    return {
        "agents": ["controlling"],
        "agent_states": [],
        "agent_properties": [
            "cash",
            "cash_flow",
            "profit_margin",
            "overall_profit_margin",
            "avg_utilization",
            "overall_avg_utilization",
            "revenue_risk",
            "accumulated_earnings",
            "accumulated_revenue",
            "accumulated_expenses",
            "avg_billing_rate",
            "overall_avg_billing_rate",
        ],
        # For ABM sessions, BPTK requires agent_property_types if agent_properties are set.
        # "total" is the most commonly used aggregation in the official beergame training tutorial.
        "agent_property_types": ["total"],
    }


# 固定的 EDT 场景配置：smEDT / interactive，显式订阅 ABM 指标
# 固定 EDT 默认场景，但让 BptkServer 自动选择 ABM 指标
EDT_FIXED_CONFIG = {
    # 场景选择：与 EDT 教程一致
    "scenario_managers": ["smEDT"],
    "scenarios": ["interactive"],

    # 不主动指定方程（如果模型有 SD 部分，可后续再加）
    "equations": [],

    # ABM 相关先全部留空，让 auto_select_abm_metrics 去做
    "agents": [],
    "agent_states": [],
    "agent_properties": [],
    "agent_property_types": [],

    # episode 配置
    "max_steps": 96,
    "reward_key_contains": "accumulated_earnings",
    "reward_mode": "delta",

    # 让 Bptk 自动选可用的 ABM metrics（这是原始 EDT 示例的默认行为）
    "auto_select_abm_metrics": True,
}



@dataclass
class EDTConfig:
    """
    Configuration for wrapping an Enterprise Digital Twin (EDT) served via BptkServer.

    - base_url: URL of the running BptkServer (e.g., http://localhost:5000)
    - scenario_managers/scenarios/equations: forwarded to /{instance_uuid}/begin-session
    - step settings: forwarded to /{instance_uuid}/run-step as `settings`
    """
    base_url: str = field(default_factory=lambda: os.getenv("EDT_BPTK_BASE_URL", "http://localhost:5000"))
    bearer_token: Optional[str] = field(default_factory=lambda: os.getenv("EDT_BPTK_BEARER_TOKEN") or None)

    scenario_managers: List[str] = field(default_factory=lambda: ["smEDT"])
    scenarios: List[str] = field(default_factory=lambda: ["interactive"])
    equations: List[str] = field(default_factory=list)  # optional; keep empty if unsure

    # ABM/hybrid session selection.
    # BptkServer's begin_session requires at least one of:
    #   - equations (SD)
    #   - agents + (agent_states or agent_properties)
    # If you leave everything empty, BptkServer will abort the session.
    agents: List[str] = field(default_factory=list)
    agent_states: List[str] = field(default_factory=list)
    agent_properties: List[str] = field(default_factory=list)
    agent_property_types: List[str] = field(default_factory=list)

    # If True, and neither equations nor (agents+metrics) are provided, the server will
    # auto-select a minimal set of ABM metrics for the "controlling" agent.
    auto_select_abm_metrics: bool = True

    max_steps: int = 100
    instance_timeout: Dict[str, int] = field(default_factory=lambda: {"minutes": 30})
    request_timeout_sec: float = 20.0
    keep_alive_each_step: bool = True

    # Best-effort reward extraction from returned run-step payload.
    reward_key_contains: str = "accumulated_earnings"
    reward_mode: str = "delta"  # delta | level


class EDTEnv:
    def __init__(self, cfg: EDTConfig):
        self.cfg = cfg
        self.base_url = _clean_base_url(cfg.base_url)
        self.env_id = str(uuid.uuid4())
        self.instance_uuid: Optional[str] = None

        self.step_count = 0
        self.done = False

        self._last_reward_level: Optional[float] = None
        self._http = requests.Session()
        # Cache for /agents schema, useful for debugging which properties are available.
        self._agent_schema: Optional[Dict[str, Dict[str, List[str]]]] = None

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.cfg.bearer_token:
            h["Authorization"] = f"Bearer {self.cfg.bearer_token}"
        return h

    def _post(self, path: str, json_payload: Optional[dict] = None) -> requests.Response:
        url = f"{self.base_url}{path}"
        '''
        _dbg(
            "http.request",
            method="POST",
            url=url,
            path=path,
            env_id=self.env_id,
            instance_uuid=self.instance_uuid,
            payload=json_payload,
        )
        '''
        resp = self._http.post(
            url,
            json=json_payload,
            headers=self._headers(),
            timeout=self.cfg.request_timeout_sec,
        )
        text = resp.text
        if text and len(text) > _MAX_TEXT:
            text = text[:_MAX_TEXT] + "...(truncated)"
        '''
        _dbg(
            "http.response",
            method="POST",
            url=url,
            status=resp.status_code,
            env_id=self.env_id,
            instance_uuid=self.instance_uuid,
            text=text,
        )
        '''
        return resp

    def _get(self, path: str) -> requests.Response:
        url = f"{self.base_url}{path}"
        '''
        _dbg(
            "http.request",
            method="GET",
            url=url,
            path=path,
            env_id=self.env_id,
            instance_uuid=self.instance_uuid,
        )
        '''
        resp = self._http.get(
            url,
            headers=self._headers(),
            timeout=self.cfg.request_timeout_sec,
        )
        text = resp.text
        if text and len(text) > _MAX_TEXT:
            text = text[:_MAX_TEXT] + "...(truncated)"
        '''
        _dbg(
            "http.response",
            method="GET",
            url=url,
            status=resp.status_code,
            env_id=self.env_id,
            instance_uuid=self.instance_uuid,
            text=text,
        )
        '''
        return resp

    def start_instance(self) -> str:
        resp = self._post("/start-instance", json_payload={"timeout": self.cfg.instance_timeout})
        resp.raise_for_status()
        data = resp.json()
        self.instance_uuid = data.get("instance_uuid") or data.get("instance_id") or data.get("uuid")
        if not self.instance_uuid:
            raise RuntimeError(f"start-instance: could not find instance_uuid in response: {data}")
        return self.instance_uuid

    def _inspect_agents_schema(self) -> Optional[dict]:
        """Query /agents for the current scenario selection.

        This helps auto-select valid agent properties/states so that ABM sessions
        actually return data (otherwise run-step can return an empty payload).
        """

        sm = self.cfg.scenario_managers[0] if self.cfg.scenario_managers else None
        sc = self.cfg.scenarios[0] if self.cfg.scenarios else None
        if not sm or not sc:
            return None

        try:
            resp = _post_with_key_fallback(
                url=self.base_url,
                endpoint="/agents",
                payload={"scenarioManager": sm, "scenario": sc},
                headers=self._headers(),
                timeout=self.cfg.request_timeout_sec,
            )
            if resp.status_code >= 400:
                return None
            return resp.json() if resp.content else None
        except Exception:
            return None

    @staticmethod
    def _normalize_agent_schema(raw: Any) -> Dict[str, Dict[str, List[str]]]:
        """Normalize the /agents response into: {agent_type: {states:[], properties:[]}}."""

        out: Dict[str, Dict[str, List[str]]] = {}
        if raw is None:
            return out

        # Most common: {"consultant": {"states": [...], "properties": [...]}, ...}
        if isinstance(raw, dict):
            # Some servers wrap it with a top-level key
            if "agents" in raw and isinstance(raw["agents"], (dict, list)):
                raw = raw["agents"]

        if isinstance(raw, dict):
            for agent_type, info in raw.items():
                if not isinstance(agent_type, str):
                    continue
                states: List[str] = []
                props: List[str] = []
                if isinstance(info, dict):
                    for k in ("states", "agent_states", "agentStates"):
                        if k in info and isinstance(info[k], list):
                            states = [str(x) for x in info[k]]
                            break
                    for k in ("properties", "agent_properties", "agentProperties"):
                        if k in info:
                            if isinstance(info[k], list):
                                props = [str(x) for x in info[k]]
                            elif isinstance(info[k], dict):
                                props = [str(x) for x in info[k].keys()]
                            break
                elif isinstance(info, list):
                    # e.g. properties only
                    props = [str(x) for x in info]
                out[agent_type] = {"states": states, "properties": props}
            return out

        # Alternate: list of dict entries
        if isinstance(raw, list):
            for item in raw:
                if not isinstance(item, dict):
                    continue
                agent_type = item.get("name") or item.get("type") or item.get("agent")
                if not isinstance(agent_type, str):
                    continue
                states = item.get("states") or item.get("agent_states") or []
                props = item.get("properties") or item.get("agent_properties") or []
                out[agent_type] = {
                    "states": [str(x) for x in (states if isinstance(states, list) else [])],
                    "properties": [str(x) for x in (props if isinstance(props, list) else [])],
                }
        return out

    def begin_session(self) -> None:
        if not self.instance_uuid:
            raise RuntimeError("begin_session called before start_instance")

        # --- Ensure we specify *something* to simulate/collect.
        # For ABM/hybrid models, BptkServer requires agents + (agent_states or agent_properties)
        # OR equations. Otherwise it will abort the session.
        equations = list(self.cfg.equations or [])
        agents = list(self.cfg.agents or [])
        agent_states = list(self.cfg.agent_states or [])
        agent_properties = list(self.cfg.agent_properties or [])
        agent_property_types = list(self.cfg.agent_property_types or [])

        if (not equations) and (not agents or (not agent_states and not agent_properties)):
            if self.cfg.auto_select_abm_metrics:
                # Auto-pick a valid ABM selection by consulting /agents.
                # This avoids the "session starts but returns no data" failure mode.
                preferred_props = [
                    "accumulated_earnings",
                    "accumulated_revenue",
                    "accumulated_expenses",
                    "cash",
                    "cash_flow",
                    "profit_margin",
                    "overall_profit_margin",
                    "avg_utilization",
                    "overall_avg_utilization",
                    "avg_billing_rate",
                    "overall_avg_billing_rate",
                    "revenue_risk",
                ]

                schema_raw = self._inspect_agents_schema()
                schema = self._normalize_agent_schema(schema_raw)
                # Cache schema for later debugging (e.g., which properties actually exist).
                self._agent_schema = schema

                agent_type = "controlling" if "controlling" in schema else (next(iter(schema.keys()), "controlling"))
                agents = [agent_type]
                avail_props = schema.get(agent_type, {}).get("properties", [])
                avail_states = schema.get(agent_type, {}).get("states", [])

                # Debug: log the controlling agent's available properties/states,
                # and whether fixed_cost / revenue_risk_level are present at all.
                _dbg(
                    "edt.agents.schema",
                    env_id=self.env_id,
                    scenario_managers=self.cfg.scenario_managers,
                    scenarios=self.cfg.scenarios,
                    agent_type=agent_type,
                    states=avail_states,
                    properties=avail_props,
                    has_fixed_cost=("fixed_cost" in avail_props),
                    has_revenue_risk_level=("revenue_risk_level" in avail_props),
                    has_revenue_risk=("revenue_risk" in avail_props),
                )

                picked = [p for p in preferred_props if p in set(avail_props)]
                if not picked and avail_props:
                    picked = avail_props[: min(12, len(avail_props))]

                # Provide at least one of agent_states/properties.
                agent_properties = picked
                agent_states = avail_states
                agent_property_types = ["total"]

                # If /agents is unavailable or returns nothing, fall back to a conservative hard-coded selection.
                if not agent_properties and not agent_states:
                    fallback = _default_abm_selection()
                    agents = fallback["agents"]
                    agent_states = fallback["agent_states"]
                    agent_properties = fallback["agent_properties"]
                    agent_property_types = fallback["agent_property_types"]
            else:
                raise RuntimeError(
                    "begin_session: you must provide either `equations` (SD) or `agents` plus at least one of "
                    "`agent_states`/`agent_properties` (ABM)."
                )

        # If the caller specified agent_properties but no property types, default to "total".
        if agents and agent_properties and not agent_property_types:
            agent_property_types = ["total"]

        # IMPORTANT: BptkServer's REST endpoints have historically accepted a mix of
        # snake_case and camelCase field names (e.g. /equations uses `scenarioManager`).
        # To maximize compatibility across versions, we send both variants for the key
        # fields used by begin_session.
        payload: Dict[str, Any] = {
            # scenario selection
            "scenario_managers": self.cfg.scenario_managers,
            "scenarioManagers": self.cfg.scenario_managers,
            "scenarios": self.cfg.scenarios,

            # what to simulate / record
            "equations": equations,
            "agents": agents,
            "agent_states": agent_states,
            "agentStates": agent_states,
            "agent_properties": agent_properties,
            "agentProperties": agent_properties,
            "agent_property_types": agent_property_types,
            "agentPropertyTypes": agent_property_types,
        }
        resp = self._post(f"/{self.instance_uuid}/begin-session", json_payload=payload)
        # Fail-fast: BptkServer may return a JSON error even when HTTP status is 200.
        # We treat any payload containing an "error" key as fatal for the episode.
        try:
            data = resp.json() if resp.content else None
        except Exception:
            data = None
        if resp.status_code >= 400:
            raise RuntimeError(f"begin-session failed: HTTP {resp.status_code} {data or resp.text}")
        if isinstance(data, dict) and data.get("error"):
            raise RuntimeError(f"begin-session failed: {data}")
        # otherwise: success

    def end_session(self) -> None:
        if not self.instance_uuid:
            return
        resp = self._post(f"/{self.instance_uuid}/end-session")
        if resp.status_code not in (200, 204):
            # tolerate teardown errors
            try:
                resp.raise_for_status()
            except Exception:
                pass

    def keep_alive(self) -> None:
        if not self.instance_uuid:
            return
        resp = self._post(f"/{self.instance_uuid}/keep-alive")
        if resp.status_code not in (200, 204):
            try:
                resp.raise_for_status()
            except Exception:
                pass

    def step(self, settings: Optional[dict] = None) -> Dict[str, Any]:
        if self.done:
            return {"error": "episode_done", "done": True}

        if not self.instance_uuid:
            raise RuntimeError("step called before init (instance_uuid missing)")

        # Debug: log the outgoing settings before hitting /run-step
        '''
        _dbg(
            "edt.step.call",
            env_id=self.env_id,
            instance_uuid=self.instance_uuid,
            step=self.step_count,
            settings=settings,
        )
        '''
        payload: Optional[Dict[str, Any]] = None
        if settings is not None:
            payload = {"settings": settings}

        resp = self._post(f"/{self.instance_uuid}/run-step", json_payload=payload)
        # Fail-fast on bad HTTP
        if resp.status_code >= 400:
            raise RuntimeError(f"run-step failed: HTTP {resp.status_code} {resp.text}")
        step_data = resp.json() if resp.content else {}
        # Fail-fast on in-band error payloads
        if isinstance(step_data, dict) and step_data.get("error"):
            raise RuntimeError(f"run-step failed: {step_data}")

        if self.cfg.keep_alive_each_step:
            self.keep_alive()

        self.step_count += 1
        self.done = self.step_count >= int(self.cfg.max_steps)

        # reward (best-effort)
        flat = dict(_flatten_numeric_leaves(step_data))
        level = None
        for k, v in flat.items():
            if self.cfg.reward_key_contains in k:
                level = float(v)
                break

        reward = 0.0
        if level is not None:
            if self.cfg.reward_mode == "level":
                reward = float(level)
            else:
                if self._last_reward_level is not None:
                    reward = float(level - self._last_reward_level)
                self._last_reward_level = float(level)

        # Debug: log a subset of metrics after the step so we can check whether
        # settings had any visible effect on key properties.
        flat_metrics = {k.split('.')[-2]: v for k, v in list(flat.items())[:200]}
        debug_keys = []
        for k in flat.keys():
            if any(x in k for x in
                   ("fixed_cost", "revenue_risk_level", "revenue_risk", "cash", "accumulated_earnings", "accumulated_expenses")):
                debug_keys.append(k)
        debug_snapshot = {k: flat_metrics[k] for k in debug_keys}

        _dbg(
            "edt.step.result",
            env_id=self.env_id,
            instance_uuid=self.instance_uuid,
            step=self.step_count,
            reward=reward,
            reward_level=level,
            metrics=debug_snapshot,
        )

        obs = {
            "step": self.step_count,
            "instance_uuid": self.instance_uuid,
            # cap to keep payload size stable for MCP
            "flat_metrics": flat_metrics,
        }

        info = {"reward_level": level, "http_status": resp.status_code}
        return {"obs": obs, "reward": reward, "done": self.done, "info": info}

    def get_session_results(self, flat: bool = True) -> Dict[str, Any]:
        if not self.instance_uuid:
            raise RuntimeError("get_session_results called before init")
        resp = self._get(f"/{self.instance_uuid}/flat-session-results" if flat else f"/{self.instance_uuid}/session-results")
        resp.raise_for_status()
        return resp.json()


# ---------------------------
# MCP tools (BeerGame-compatible ergonomics)
# ---------------------------

_ENVS: Dict[str, EDTEnv] = {}


@mcp.tool(name="edt-list-scenarios", description="List available scenarios from the connected BptkServer.")
def edt_list_scenarios(base_url: Optional[str] = None, bearer_token: Optional[str] = None) -> dict:
    url = _clean_base_url(base_url or os.getenv("EDT_BPTK_BASE_URL", "http://localhost:5000"))
    headers = {"Content-Type": "application/json"}
    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"
    resp = requests.get(f"{url}/scenarios", headers=headers, timeout=15)
    resp.raise_for_status()
    return resp.json()


@mcp.tool(name="edt-list-equations", description="List available equations/constants for a scenario manager + scenario.")
def edt_list_equations(
    scenario_manager: str,
    scenario: str,
    base_url: Optional[str] = None,
    bearer_token: Optional[str] = None,
) -> dict:
    url = _clean_base_url(base_url or os.getenv("EDT_BPTK_BASE_URL", "http://localhost:5000"))
    headers = {"Content-Type": "application/json"}
    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"
    resp = _post_with_key_fallback(
        url=url,
        endpoint="/equations",
        payload={"scenarioManager": scenario_manager, "scenario": scenario},
        headers=headers,
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


@mcp.tool(
    name="edt-list-agents",
    description="List agents (types), their states, and available properties for an ABM/hybrid scenario.",
)
def edt_list_agents(
    scenario_manager: str,
    scenario: str,
    base_url: Optional[str] = None,
    bearer_token: Optional[str] = None,
) -> dict:
    url = _clean_base_url(base_url or os.getenv("EDT_BPTK_BASE_URL", "http://localhost:5000"))
    headers = {"Content-Type": "application/json"}
    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"

    resp = _post_with_key_fallback(
        url=url,
        endpoint="/agents",
        payload={"scenarioManager": scenario_manager, "scenario": scenario},
        headers=headers,
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


@mcp.tool(
    name="init-edt-env",
    description=(
        "Initialize an EDT episode (creates BptkServer instance + begin-session). "
        "Config is determined by the evaluation framework (test_samples), "
        "NOT by free-form LLM text."
    ),
)
def init_edt_env(config: Optional[dict] = None) -> dict:
    """
    初始化 EDT 环境。

    注意：
    - 这里不再“让 LLM 决定 cfg”，而是：
        * 上层评测脚本（例如 SeriousGame/run_edt.py + evaluate_edt_set）
          把 test_samples[i]['config'] 作为 config 传入；
        * 若 config 为空，则使用 EDT_FIXED_CONFIG。
    - 也就是说，这里的 config 完全由 Python 侧控制，不经由自然语言解析。
    """
    merged_cfg: Dict[str, Any] = dict(EDT_FIXED_CONFIG)
    if config:
        # 允许上层在固定模板基础上做少量覆盖（例如 max_steps/agents 等）
        merged_cfg.update(config)

    cfg = EDTConfig(**merged_cfg)
    env = EDTEnv(cfg)
    env.start_instance()
    env.begin_session()
    _ENVS[env.env_id] = env

    return {
        "env_id": env.env_id,
        "instance_uuid": env.instance_uuid,
        # 首次 obs 暂时只返回 step + instance_uuid，后续可扩展
        "obs": {"step": 0, "instance_uuid": env.instance_uuid},
        "config": {**cfg.__dict__, "base_url": env.base_url},
    }



@mcp.tool(name="step-edt-env", description="Advance one timestep in EDT. Optionally provide `settings` forwarded to BptkServer run-step.")
def step_edt_env(env_id: str, settings: Optional[dict] = None) -> dict:
    # _dbg("tool.call", tool="step-edt-env", env_id=env_id, settings=settings)
    env = _ENVS[env_id]
    out = env.step(settings=settings)
    '''
    _dbg(
        "tool.return",
        tool="step-edt-env",
        env_id=env_id,
        step=out.get("obs", {}).get("step"),
        done=out.get("done"),
        reward=out.get("reward"),
        info=out.get("info"),
        flat_metric_keys=list((out.get("obs") or {}).get("flat_metrics", {}).keys())[:30],
        out=out,
    )
    '''
    return out


@mcp.tool(name="get-edt-state", description="Get current environment state (step counter, instance_uuid, config).")
def get_edt_state(env_id: str) -> dict:
    env = _ENVS[env_id]
    return {
        "env_id": env.env_id,
        "instance_uuid": env.instance_uuid,
        "step_count": env.step_count,
        "done": env.done,
        "config": env.cfg.__dict__,
    }


'''
@mcp.tool(name="get-edt-metrics", description="Get accumulated session results (flat by default) for an EDT episode.")
def get_edt_metrics(env_id: str, flat: bool = True) -> dict:
    _dbg("tool.call", tool="get-edt-metrics", env_id=env_id, flat=flat)
    env = _ENVS[env_id]
    results = env.get_session_results(flat=flat)

    # best-effort: surface a few common KPI names used in the EDT tutorial
    flat_vals = dict(_flatten_numeric_leaves(results))
    kpi_candidates = [
        "consultant_demand",
        "consultant_capacity_fte",
        "avg_utilization",
        "overall_avg_utilization",
        "avg_consulting_fee",
        "overall_avg_consulting_fee",
        "cash_flow",
        "cash",
        "revenue_risk",
        "profit_margin",
        "overall_profit_margin",
        "accumulated_earnings",
        "accumulated_revenue",
        "accumulated_expenses",
    ]
    kpis: Dict[str, float] = {}
    for cand in kpi_candidates:
        for k, v in flat_vals.items():
            if cand in k:
                kpis[cand] = float(v)
                break

    return {"kpis": kpis, "results": results}
'''


@mcp.tool(name="close-edt-env", description="End session and free the EDT environment handle.")
def close_edt_env(env_id: str) -> dict:
    env = _ENVS.pop(env_id, None)
    if env is not None:
        try:
            env.end_session()
        except Exception:
            pass
    return {"ok": True}


if __name__ == "__main__":
    # STDIO is the default transport. Do not print to stdout except protocol messages.
    mcp.run()
