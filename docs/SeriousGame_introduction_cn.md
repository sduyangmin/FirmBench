# EDT 评测简介（Scenario-level）

## 1. 严肃游戏概述
EDT（Enterprise Digital Twin）是一个企业经营仿真环境：公司拥有一定数量的 **consultant（顾问资源）** 与多个 **project（项目）**。仿真按离散时间步推进，空闲顾问会自动加入可执行项目并完成工作；项目带来收入，公司同时承担固定开销与人员成本，最终形成利润、现金、利用率等经营指标。Agent通过一次性战略配置（团队规模、项目组合、排期、风险偏好）改变企业经营轨迹，从而反应出其决策能力。

## 2. 原始 scenario 关键字段与 Agent 可调项
每个scenario决定企业仿真环境的相关参数。Agent调控模板scenario，即interactive.json来首先相关指标的最大化。

### 2.1 `interactive.json` 模板场景（关键字段节选）

- 时间步长 `dt`：0.25
- 默认风险参数 `revenue_risk_level`：1.0
- 模板可用顾问上限 `consultants_max`：6
- 项目字段（按模板顺序；只列核心经营相关字段）：
- **0. Project 1**：需求顾问数=2, 合同工作量=48.0, billing_rate(不可控)=18000.0, start_time=1.0, deadline=26.0, 合同概率=1.0, 延期概率=0.75, 后续项目概率=0.25
- **1. Project 2**：需求顾问数=2, 合同工作量=48.0, billing_rate(不可控)=18000.0, start_time=1.0, deadline=26.0, 合同概率=1.0, 延期概率=0.75, 后续项目概率=0.25
- **2. Project 3**：需求顾问数=2, 合同工作量=48.0, billing_rate(不可控)=18000.0, start_time=1.0, deadline=26.0, 合同概率=1.0, 延期概率=0.75, 后续项目概率=0.25

### 2.2 Agent 动作设计（一次性生成 scenario）
本基准采用 **scenario-level** 动作：Agent **仅在 episode 开始前行动一次**，输出极简 JSON schema；评测器据此在 `interactive.json` 基础上生成新的 scenario 文件，然后让仿真完整跑完（中途不再干预）。

最小动作 schema：
```json
{
  "C": <int>,                // 保留的顾问数量（0..C_max）
  "R": <float>,              // revenue_risk_level（0..1）
  "P": [                     // 按模板项目顺序对齐
    0 | [1, start_step, deadline_step],
    ...
  ]
}
```

含义与约束：
- `C`：团队规模（顾问人数）。人数越多潜在交付越快，但闲置会增加成本压力。
- `P[i]`：项目是否执行与排期（`start_step/deadline_step` 为步索引）。
- `R`：风险/扩展偏好，影响延期/后续项目等事件的触发，从而改变收入与成本轨迹。
- **不可调**：工资、固定成本、billing_rate 等结构性参数（避免“无脑提价”等退化策略）。

## 3. 评测指标（终局报告）
环境每个 step 都会返回经营指标；评测在 episode 结束时汇总输出终局值，核心包括：

**accumulated_earnings**：累计收益/累计净利润（主 KPI）

**accumulated_revenue**：累计收入

**accumulated_expenses**：累计支出

**cash**：终局现金余额（状态量）

**cash_flow**：现金流（过程量/当期量，具体口径取决于环境实现）。现金流有时间上的延迟。

**profit_margin**：当期利润率（更偏当期/末步）

**overall_profit_margin**：整体利润率（累计利润）

**avg_utilization / overall_avg_utilization**：当前资源利用率/全期平均资源利用率

**revenue_risk**：收入风险指标

## 4. Baseline vs A-mem(GPT-5)（两次运行结果对比）
KPI提升：
- `accumulated_earnings`：**51,000 → 78,000**（Δ 27,000；52.9%）
- 现金从 -3,000 改善到 24,000
- overall_profit_margin 从约 9.94% 提升到约 14.44%

对比表（终局值）：
| 指标 | Baseline（直接运行 interactive.json） | A-mem(GPT-5) 调整后（生成新 scenario） | 变化 |
|---|---:|---:|---:|
| `accumulated_earnings` | 51,000 | 78,000 | 27,000 |
| `accumulated_revenue` | 513,000 | 540,000 | 27,000 |
| `accumulated_expenses` | 462,000 | 462,000 | 0 |
| `cash` | -3,000 | 24,000 | 27,000 |
| `cash_flow` | 5,000 | 5,000 | 0 |
| `profit_margin` | 0.1852 | 0.1852 | 0.0000 |
| `overall_profit_margin` | 0.0994 | 0.1444 | 0.0450 |
| `avg_utilization` | 1.0000 | 1.0000 | 0.0000 |
| `overall_avg_utilization` | 0.9048 | 0.9524 | 0.0476 |
| `revenue_risk` | 0.0000 | 0.0000 | 0.0000 |
| `total_reward` | 51,000 | 78,000 | 27,000 |

# BeerGame 评测简介

## 1. 严肃游戏概述

Beer Game（啤酒分销游戏）是经典的供应链动态仿真：下游产生需求并向上游逐级传导，链路存在信息不对称与交付/生产延迟。参与者（或智能体）需要在每个离散时间步（通常按“周”）决定向上游的订货量。由于延迟与库存/缺货耦合，策略容易诱发 **牛鞭效应（Bullwhip Effect）**：订单波动被放大，导致库存与缺货在时间上剧烈振荡。

在本评测框架中，一个 scenario 对应一个 episode；环境会指定一个 **controlled_role**（如 retailer / wholesaler / distributor / factory），Agent 仅控制该角色，其余角色由环境内置策略/规则驱动（取决于 MCP server 实现）。

---

## 2. Agent 动作与输入

### 2.1 动作（Action）

- **唯一动作：** `order_qty`（向上游下单数量）
- 类型：整数；通常会在 agent 侧做非负与上限裁剪（例如 `0..max_order_qty`）。

每个 step 框架都会调用一次 `policy_fn(obs, ctx) -> int`，将返回的 `order_qty` 作为当步动作提交给环境。

### 2.2 观测（Observation）

`obs` 为环境返回的结构化状态（dict）。典型字段包括（以你们 agent 的 query 构造逻辑为代表）：

- `role`：当前受控角色
- `week`：当前周/时间步
- `inventory`：库存水平
- `backorder`：缺货/积压
- `incoming_order`：来自下游的订单（需求信号）
- `supply_line`：在途订单/生产管道（尚未到货）
- `last_order`：上一周向上游下的订单等

实际字段以 beergame MCP server 的返回为准。

---

## 3. 评测流程（Episode 执行逻辑）

1. **new_episode**：创建环境，获得 `env_id` 与初始 `obs`，读取 `controlled_role`。  
2. **循环 step**：直到 done。每步：
   - `order_qty = policy_fn(obs, ctx)`
   - 调用环境 step 工具推进仿真并返回下一步 `obs`。  
3. **metrics**：episode 结束后调用 metrics 工具拉取本局指标。  
4. **trace（可选）**：若 server 支持 trace，则拉取轨迹用于计算 bullwhip（见下）。  
5. **聚合**：对所有 episodes 做均值/方差等统计，生成 rollup 指标，并保留每局原始 `episode_metrics` 以便诊断与复现。

---

## 4. 评测指标（含义与解读）

> 说明：下列“*_controlled”字段表示“受控角色”的指标；若 server 未提供对应字段，评测器会回退到不带 controlled 的通用字段（例如 `total_cost_controlled` 缺失则用 `total_cost`）。

### 4.1 成本类（通常越低越好）

- **`avg_total_cost_controlled` / `std_total_cost_controlled`**  
  受控角色在每个 episode 的总成本（`total_cost_controlled` 或回退 `total_cost`）的均值/标准差。  
  业务含义：该角色在整段仿真中的总成本，通常由：
  - 库存持有成本（inventory holding）
  - 缺货/积压惩罚（backorder penalty）
  - 可能包含订购/运输等成本  
    具体成本口径以 MCP server 的 metrics 实现为准；评测器不重定义成本，只做汇总统计。

### 4.2 缺货/服务水平压力（通常越低越好）

- **`avg_total_backlog_controlled` / `std_total_backlog_controlled`**  
  受控角色 backlog（`total_backlog_controlled` 或回退 `total_backlog`）的均值/标准差。  
  backlog 一般表示未满足需求造成的积压量（缺货导致的未交付订单）。该值越大，通常意味着服务水平越差、罚金更高，也更容易诱发策略补货振荡。

### 4.3 库存压力（通常需要“适中”；该指标本身越高通常对应更高资金占用与持有成本）

- **`avg_total_inventory_controlled` / `std_total_inventory_controlled`**  
  受控角色库存（`total_inventory_controlled` 或回退 `total_inventory`）的均值/标准差。  
  inventory 代表库存水平（或累计口径，取决于 server 定义）。库存过高一般意味着持有成本与资金占用增加；库存过低则可能导致 backlog 上升。

### 4.4 牛鞭效应（Bullwhip，通常越低越好）

- **`avg_bullwhip_controlled` / `std_bullwhip_controlled`**  
  评测器在获得 trace 后计算牛鞭效应，典型定义为：
  $$
  \text{bullwhip} := \frac{\mathrm{Var}(\text{orders})}{\mathrm{Var}(\text{demand})}
  $$

  - `demand`：来自 trace 中每步的需求信号
  - `orders`：来自 trace 中受控角色的 outgoing order（对上游下单）  
    直观含义：订单波动相对于需求波动被放大的倍数。值越大表示策略越不稳定、对上游扰动越强；接近 1 表示订单波动与需求波动接近。

