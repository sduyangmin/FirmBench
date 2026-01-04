# FirmBench: A Benchmark for Strategic Reasoning and Decision-Making of LLM Agents in Enterprise Simulation

FirmBench is a comprehensive benchmarking platform specifically designed to evaluate the **strategic reasoning and decision-making capabilities** of Large Language Model (LLM) agents in complex enterprise scenarios. Unlike traditional static QA benchmarks, FirmBench introduces interactive consulting cases and dynamic serious games to authentically simulate information asymmetry, sequential feedback, and long-term planning challenges inherent in corporate decision-making.

---

## Core Components

### 1. Structured Reasoning
This module comprises 10 curated financial and logical reasoning datasets to assess the foundational cognitive abilities of agents. Tasks range from extracting key information from financial reports to complex numerical calculations and the application of professional domain knowledge.
- **Datasets included**: FinQA, TAT-QA, ConvFinQA, finer, FinKnow, FormulaEval, and more.
- **Evaluation Domains**: Information Extraction, Numerical Calculation, Domain Knowledge.
- **Run scripts**: `run_scripts/StructuredReasoning/online/`

### 2. Consulting
This module simulates management consulting case interviews. Agents act as interviewees and must engage in multi-turn interactions with an LLM-based interviewer to proactively acquire hidden information before providing structured business recommendations.
- **Features**: Emphasizes structured problem-solving, information acquisition, quantitative analysis, and business intuition.
- **Scoring Dimensions**: Structure, Quantitative Reasoning, Business Sense, Communication.
- **Run scripts**: `run_scripts/Consulting/online/`

### 3. Serious Game
This module includes two dynamic simulation environments to evaluate agents' long-term strategic planning under uncertainty and delayed feedback.
- **Beer Game**: A classic supply chain management simulation where agents manage inventory to minimize total costs while dealing with the bullwhip effect and demand fluctuations.
- **Enterprise Digital Twin (EDT)**: A high-fidelity business process simulation where agents act as CEOs, making cross-functional operational decisions to maximize accumulated earnings.
- **Run scripts**: `run_scripts/SeriousGame/`

---

## Agent-based Taxonomy

FirmBench proposes an agent-centric classification framework. Using an expert model, all samples are re-annotated based on required cognitive capabilities and difficulty scores, enabling fine-grained performance diagnosis beyond traditional dataset boundaries.

- **Information Extraction (IE)**: Accurately retrieving facts from documents.
- **Numerical Calculation (NC)**: Mathematical reasoning involving arithmetic logic or code synthesis.
- **Domain Knowledge (DK)**: Specialized financial terminology, standards (e.g., XBRL), and industry common sense.
- **Complex Reasoning (CR)**: Handling situational ambiguity, strategic choices, and interactive decision-making.

---

## Quick Start

### 0. Preparation
Ensure your Python environment is version 3.10+ and install dependencies:
```bash
pip install -r requirements.txt
```
Configure your LLM API (supports various providers; see script parameters for details).

### 1. Task Reclassification
Before running experiments, categorize and score the difficulty of the task samples:
```bash
bash run_scripts/StructuredReasoning/data_process/reclassify_test.sh
```

### 2. Run Agent Experiments
You can choose to run specific agents or the entire benchmark suite:
- **Run all Structured Reasoning experiments**:
  ```bash
  bash run_scripts/StructuredReasoning/online/run_all_agents_online.sh
  ```
- **Run Consulting task experiments**:
  ```bash
  bash run_scripts/Consulting/online/run_all_online.sh
  ```
- **Run Serious Game experiments**:
  ```bash
  python SeriousGame/run_beergame.py
  python SeriousGame/run_edt.py
  ```

### 3. Result Synthesis and Evaluation
Once experiments are completed, use the capability evaluation script to generate statistical tables:
```bash
bash run_scripts/StructuredReasoning/online/capability_eval/online/capability_eval_online.sh
```

---

## FirmAgent (Agent-of-Agents)

FirmAgent is our proposed adaptive routing ensemble framework. By analyzing the execution traces of various base agents across diverse tasks, it learns and synthesizes routing rules to dynamically assign new samples to the most specialized agent.

For more details, please refer to: [docs/AOA.md](docs/AOA.md)

---

## Citation

If you use FirmBench in your research, please cite our paper:

```bibtex

```
