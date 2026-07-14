# Agent From Scratch

A starter template and workspace for building advanced AI agents using **LangGraph** and **LangChain**. This project builds up from a basic tool-calling conversational agent to a production-grade Email Assistant complete with Human-in-the-loop (HITL) overrides, dynamic memory management, and automatic LangSmith evaluation.

---

## 📂 Project Structure

* **`src/langgraph101.py`**: The basic starter graph implementing a simple loop of model-driven tool calling with a `write_email` tool.
* **`src/agent.py`**: The core Email Assistant workflow. It triages incoming emails (`ignore`, `respond`, `notify`) using structured outputs and runs an agent workflow to respond if needed.
* **`src/agent_hitl.py`**: Email Assistant with human feedback loops. Prompts manual reviews via LangGraph `interrupt` for approval/editing/ignoring of actions and triage alerts.
* **`src/agent_memory.py`**: A memory-capable Email Assistant. Uses LangGraph's `BaseStore` to persist, retrieve, and dynamically update user preferences based on human-in-the-loop feedback/edits.
* **`src/prompts.py`**: Prompts and system instructions for triage nodes, agents, background information, and default preferences.
* **`src/eval/`**: Evaluation suite:
  * `email_dataset.py`: Comprehensive test datasets with ground-truth triage labels and tool criteria.
  * `evaluate_triage.py`: LangSmith evaluation workflow running triage accuracy tests.
  * `prompts.py`: LLM-as-a-judge system prompts for evaluating results.
* **`langgraph.json`**: Studio configuration mapping endpoints to defined graphs.
* **`requirements.txt`**: Package dependencies.

---

## 🛠️ Graph Hierarchy & Core Agents

The project contains four separate graphs configured in `langgraph.json`:

### 1. Simple Agent (`agent` / `src/langgraph101.py`)
A basic state graph that invokes a model to compose emails.
```
          [ START ]
              │
              ▼
        ┌───────────┐
        │ call_llm  │◄────────┐
        └─────┬─────┘         │
              │               │
              ├───────────────┤ (run_tool)
              ▼               │
      [ should_continue? ]────┘
              │
              ▼ (END)
           [ END ]
```

---

### 2. Email Assistant (`agent_2` / `src/agent.py`)
An end-to-end triaged assistant. It parses the incoming email details:
1. **Triage Router**: Runs a structured-output classification.
   * `ignore` -> Direct to `END`.
   * `notify` -> Direct to `END` (intended for FYI/Announcements).
   * `respond` -> Routes to the Response Agent which utilizes tools (`write_email`, `schedule_meeting`, `check_calendar_availability`) and exits via the `Done` tool.

---

### 3. Email Assistant + Human-in-the-Loop (`agent_hitl` / `src/agent_hitl.py`)
Introduces human review hooks using LangGraph `interrupts`:
1. **Triage Interrupt**: When a triage decision is categorised as `notify`, the system interrupts so the user can choose to ignore the alert or reply with feedback (dynamically routing to the response agent).
2. **Tool Execution Interrupt**: Intercepts tool calls (`write_email`, `schedule_meeting`, `Question`) before execution. The user can:
   * **Accept**: Run the tool with proposed arguments.
   * **Edit**: Modify arguments (e.g. rewrite an email draft, adjust calendar dates) before execution.
   * **Ignore**: Abandon the path.
   * **Response**: Give feedback to the LLM to trigger a regeneration.

---

### 4. Email Assistant + Memory (`agent_memory` / `src/agent_memory.py`)
Integrates a dynamic personalization layer:
* **BaseStore Integration**: Standardized state queries read user's configuration profiles (`triage_preferences`, `response_preferences`, `cal_preferences`).
* **Self-Updating Memory**: When hitl interactions take place (e.g., the user edits a draft or changes a triage choice), a background LLM process summarizes the user's implicit preferences and appends/updates them into the store.
* Preferences persist across sessions, refining the agent's behavior dynamically based on usage history.

---

## 🧪 Evaluation

We use LangSmith to evaluate the performance of our triage logic and email replies:
* Run the evaluator:
  ```bash
  python src/eval/evaluate_triage.py
  ```
* This compares the agent's classifications on the `email_inputs` found in `src/eval/email_dataset.py` against the reference answers.

---

## 🚀 Setup & Running

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Create a `.env` file in the root directory (based on `example.env`):
```env
OPENAI_API_BASE=https://api.openai.com/v1     # Or your proxy/LLM gateway URL
OPENAI_API_KEY=your-api-key
LANGSMITH_API_KEY=your-langsmith-key          # Optional: For evaluation tracking
```

### 3. Run Locally & Tests
To run the starter graph:
```bash
python src/langgraph101.py
```

### 4. Deploy or View in LangGraph Studio
Simply drag the project folder or open it in **LangGraph Studio** using `langgraph.json` to visually debug, step-through interrupts, and verify state modifications.
