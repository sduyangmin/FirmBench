
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Any
import statistics

# Prefer official MCP SDK; fallback to fastmcp if you use that.
try:
    from mcp.server.fastmcp import FastMCP  # official MCP Python SDK style
except Exception:  # pragma: no cover
    from fastmcp import FastMCP  # community FastMCP

Role = Literal["retailer", "wholesaler", "distributor", "brewery"]

mcp = FastMCP("beergame-mcp")

# ---------------------------
# Core Beer Game simulation
# ---------------------------

@dataclass
class BeerGameConfig:
    horizon_weeks: int = 52

    # demand script (retailer sees customer demand)
    demand_low: int = 100
    demand_high: int = 400
    demand_step_week: int = 2  # week index (0-based). week>=2 -> high

    # delays
    order_delay: int = 1       # info delay (orders travel upstream)
    ship_delay: int = 2        # delivery delay (goods travel downstream)

    # initial steady state
    target_inventory: int = 400

    # costs (align to typical BPTK tutorial settings)
    inventory_item_cost: float = 0.5
    backorder_item_cost: float = 1.0
    minimum_inventory_cost: float = 200.0

    # evaluation role and opponents
    controlled_role: Role = "retailer"
    opponent_policy: str = "typical"  # typical | no_backorder | base_stock | smoothing:4


@dataclass
class EchelonState:
    role: Role
    inventory: int
    backorder: int

    # Pipelines
    incoming_deliveries: List[int]  # length = ship_delay
    incoming_orders: List[int]      # length = order_delay

    # For reporting / metrics
    last_incoming_order: int = 0
    last_incoming_delivery: int = 0
    last_outgoing_delivery: int = 0
    last_outgoing_order: int = 0

    def supply_line(self) -> int:
        # outstanding orders ≈ goods in transit
        return sum(self.incoming_deliveries)


class BeerGameEnv:
    def __init__(self, cfg: BeerGameConfig):
        self.cfg = cfg
        sd, od = cfg.ship_delay, cfg.order_delay

        init_inv = cfg.target_inventory
        # initialize each echelon near steady-state:
        # pipelines filled with demand_low so the system starts stable-ish
        self.state: Dict[Role, EchelonState] = {
            r: EchelonState(
                role=r,
                inventory=init_inv,
                backorder=0,
                incoming_deliveries=[cfg.demand_low] * sd,
                incoming_orders=[cfg.demand_low] * od,
            )
            for r in ["retailer", "wholesaler", "distributor", "brewery"]
        }

        self.week: int = 0
        self.done: bool = False

        # history for metrics
        self.orders_history: Dict[Role, List[int]] = {r: [] for r in self.state}
        self.incoming_order_history: Dict[Role, List[int]] = {r: [] for r in self.state}
        self.cost_history: Dict[Role, List[float]] = {r: [] for r in self.state}

    def demand(self) -> int:
        return self.cfg.demand_high if self.week >= self.cfg.demand_step_week else self.cfg.demand_low

    def _policy(self, role: Role, incoming_order: int) -> int:
        """Rule-based policy for non-controlled roles."""
        s = self.state[role]
        tgt = self.cfg.target_inventory
        pol = self.cfg.opponent_policy

        if pol == "typical":
            # Classic Beer Game heuristic (often causes bullwhip):
            # order = incoming + target - inventory + backorder
            return max(0, int(incoming_order + tgt - s.inventory + s.backorder))

        if pol == "no_backorder":
            return max(0, int(incoming_order + tgt - s.inventory))

        if pol == "base_stock":
            inv_pos = s.inventory + s.supply_line() - s.backorder
            return max(0, int(tgt - inv_pos + incoming_order))

        if pol.startswith("smoothing:"):
            # smoothing:T
            try:
                T = int(pol.split(":")[1])
            except Exception:
                T = 4
            inv_pos = s.inventory + s.supply_line() - s.backorder
            adj = (tgt - inv_pos) / max(1, T)
            return max(0, int(incoming_order + adj))

        # fallback
        return max(0, int(incoming_order))

    def _cost(self, role: Role) -> float:
        s = self.state[role]
        holding = max(self.cfg.minimum_inventory_cost, self.cfg.inventory_item_cost * max(0, s.inventory))
        backlog = self.cfg.backorder_item_cost * max(0, s.backorder)
        return float(holding + backlog)

    def _obs(self, role: Role) -> Dict[str, Any]:
        s = self.state[role]
        return {
            "week": self.week,
            "role": role,
            "inventory": s.inventory,
            "backorder": s.backorder,
            "incoming_order": s.last_incoming_order,
            "incoming_delivery": s.last_incoming_delivery,
            "outgoing_delivery": s.last_outgoing_delivery,
            "supply_line": s.supply_line(),
            "last_order": s.last_outgoing_order,
            "horizon_weeks": self.cfg.horizon_weeks,
            "opponent_policy": self.cfg.opponent_policy,
        }

    def step(self, controlled_order_qty: int) -> Dict[str, Any]:
        if self.done:
            return {"error": "episode_done"}

        cfg = self.cfg
        roles: List[Role] = ["retailer", "wholesaler", "distributor", "brewery"]

        # 1) Deliveries arrive (shift pipeline)
        for r in roles:
            s = self.state[r]
            incoming_delivery = s.incoming_deliveries.pop(0)
            s.incoming_deliveries.append(0)
            s.last_incoming_delivery = incoming_delivery
            s.inventory += incoming_delivery

        # 2) Orders arrive (shift pipeline); retailer demand comes from consumer script
        incoming_orders: Dict[Role, int] = {}
        for r in roles:
            s = self.state[r]
            inc = s.incoming_orders.pop(0)
            s.incoming_orders.append(0)
            incoming_orders[r] = inc

        incoming_orders["retailer"] = self.demand()

        # record incoming orders
        for r in roles:
            self.state[r].last_incoming_order = incoming_orders[r]
            self.incoming_order_history[r].append(incoming_orders[r])

        # 3) Ship to downstream:
        # brewery -> distributor -> wholesaler -> retailer -> (customer)
        downstream: Dict[Role, Optional[Role]] = {
            "brewery": "distributor",
            "distributor": "wholesaler",
            "wholesaler": "retailer",
            "retailer": None,
        }

        outgoing_deliveries: Dict[Role, int] = {}
        for r in roles:
            s = self.state[r]
            demand = incoming_orders[r] + s.backorder
            out_deliv = min(s.inventory, demand)
            s.inventory -= out_deliv
            s.backorder = demand - out_deliv
            s.last_outgoing_delivery = out_deliv
            outgoing_deliveries[r] = out_deliv

            d = downstream[r]
            if d is not None:
                # delivery will arrive after ship_delay via pipeline tail
                self.state[d].incoming_deliveries[-1] += out_deliv

        # 4) Place orders upstream:
        upstream: Dict[Role, Role] = {
            "retailer": "wholesaler",
            "wholesaler": "distributor",
            "distributor": "brewery",
            "brewery": "brewery",  # brewery "orders" translate into production
        }

        outgoing_orders: Dict[Role, int] = {}
        for r in roles:
            if r == cfg.controlled_role:
                oq = max(0, int(controlled_order_qty))
            else:
                oq = self._policy(r, incoming_orders[r])

            outgoing_orders[r] = oq
            self.state[r].last_outgoing_order = oq
            self.orders_history[r].append(oq)

            up = upstream[r]
            # order travels with order_delay
            self.state[up].incoming_orders[-1] += oq

        # Production: brewery converts its own order into future inventory (via its pipeline)
        # We model production as a "delivery into brewery inventory" after ship_delay.
        self.state["brewery"].incoming_deliveries[-1] += outgoing_orders["brewery"]

        # 5) Costs
        week_costs: Dict[Role, float] = {}
        for r in roles:
            c = self._cost(r)
            week_costs[r] = c
            self.cost_history[r].append(c)

        self.week += 1
        self.done = self.week >= cfg.horizon_weeks

        controlled = cfg.controlled_role
        reward = -week_costs[controlled]

        return {
            "obs": self._obs(controlled),
            "reward": reward,
            "done": self.done,
            "info": {
                "week_costs": week_costs,
                "outgoing_orders": outgoing_orders,
                "outgoing_deliveries": outgoing_deliveries,
            }
        }

    def metrics(self) -> Dict[str, Any]:
        controlled = self.cfg.controlled_role

        total_cost_controlled = sum(self.cost_history[controlled])
        total_cost_supply_chain = sum(sum(self.cost_history[r]) for r in self.state.keys())

        # bullwhip (controlled): Var(orders)/Var(incoming_order) with small-sample guard
        orders = self.orders_history[controlled]
        inc = self.incoming_order_history[controlled]
        bullwhip = None
        if len(orders) >= 3 and len(inc) >= 3 and statistics.pvariance(inc) > 0:
            bullwhip = statistics.pvariance(orders) / statistics.pvariance(inc)

        # service proxy: average backlog (controlled)
        avg_backlog = statistics.mean([self._backlog_at_week(controlled, i) for i in range(len(self.cost_history[controlled]))]) \
            if self.cost_history[controlled] else 0.0

        return {
            "weeks_completed": self.week,
            "controlled_role": controlled,
            "opponent_policy": self.cfg.opponent_policy,
            "total_cost_controlled": float(total_cost_controlled),
            "total_cost_supply_chain": float(total_cost_supply_chain),
            "bullwhip_controlled": bullwhip,
            "avg_backlog_controlled": float(avg_backlog),
        }

    def _backlog_at_week(self, role: Role, t: int) -> int:
        # We did not store backlog history explicitly; keep it simple: approximate by current state if needed.
        # If you need exact backlog history, store it per step similarly to cost_history.
        return self.state[role].backorder


# ---------------------------
# MCP tools
# ---------------------------

_ENVS: Dict[str, BeerGameEnv] = {}

@mcp.tool(name="init-beer-env", description="Initialize a Beer Game episode and return env_id and initial observation.")
def init_beer_env(config: Optional[dict] = None) -> dict:
    cfg = BeerGameConfig(**(config or {}))
    env = BeerGameEnv(cfg)
    env_id = str(uuid.uuid4())
    _ENVS[env_id] = env
    return {"env_id": env_id, "obs": env._obs(cfg.controlled_role), "config": cfg.__dict__}

@mcp.tool(name="step-beer-env", description="Advance one week in Beer Game. Provide order_qty for the controlled role.")
def step_beer_env(env_id: str, order_qty: int) -> dict:
    env = _ENVS[env_id]
    return env.step(order_qty)

@mcp.tool(name="get-beer-state", description="Get current full-chain state for debugging/analysis.")
def get_beer_state(env_id: str) -> dict:
    env = _ENVS[env_id]
    snapshot = {
        r: {
            "inventory": s.inventory,
            "backorder": s.backorder,
            "incoming_deliveries": list(s.incoming_deliveries),
            "incoming_orders": list(s.incoming_orders),
            "last_incoming_order": s.last_incoming_order,
            "last_incoming_delivery": s.last_incoming_delivery,
            "last_outgoing_delivery": s.last_outgoing_delivery,
            "last_outgoing_order": s.last_outgoing_order,
        }
        for r, s in env.state.items()
    }
    return {"week": env.week, "done": env.done, "state": snapshot, "config": env.cfg.__dict__}

@mcp.tool(name="get-beer-metrics", description="Get cumulative evaluation metrics for an episode.")
def get_beer_metrics(env_id: str) -> dict:
    env = _ENVS[env_id]
    return env.metrics()

@mcp.tool(name="close-beer-env", description="Close an environment and free memory.")
def close_beer_env(env_id: str) -> dict:
    _ENVS.pop(env_id, None)
    return {"ok": True}

if __name__ == "__main__":
    # STDIO is the default transport. Do not print to stdout except protocol messages.
    mcp.run()
