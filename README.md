# AgentFromScratch 🤖✍️

A minimal, clean starter template for building AI agents using **LangGraph** and **LangChain**. This project implements a basic conversational agent capable of dynamically deciding when to call system tools (such as composing and sending emails).

---

## 📂 Project Structure

* **`src/langgraph101.py`**: The core workflow script. It defines:
  * A `write_email` custom tool.
  * A `call_llm` node that invokes a model bound to the tool.
  * A conditional check (`should_continue`) that determines whether to transition to the tool execution node or exit.
  * An interactive execution block at the bottom to test the workflow locally.
* **`langgraph.json`**: Graph deployment configuration for LangGraph Studio.
* **`requirements.txt`**: Package dependencies.
* **`example.env`**: Specimen setup for project environment variables.

---

## 💡 System Design

The agent runs a dynamic control flow using a LangGraph `StateGraph`:

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

1. **State Node**: Stores conversation history via `MessagesState`.
2. **LLM Node (`call_llm`)**: Bound to available tools, calling them as needed.
3. **Conditional Router (`should_continue`)**: Checks the last message's metadata. If it contains tool calls, routes to `run_tool`, otherwise stops at `END`.
4. **Tool Node (`run_tool`)**: Executes the system function and returns results back to the workflow's state.

---

## 🛠️ Setup & Running

### 1. Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables
Create a `.env` file in the root directory (you can copy from `example.env`):
```env
OPENAI_API_KEY=your_openai_api_key
OPENAI_API_BASE=your_openai_api_base_url
```

### 3. Run the Agent Locally
Run the entry-point script to test the agent workflow:
```bash
python src/langgraph101.py
```
This will trigger the graph with a prompt asking the agent to send an email to a sample address and document the step-by-step state changes.

### 4. Deploy or Open in LangGraph Studio
You can use `langgraph.json` directly to load the graph visually in LangGraph Studio or deploy it dynamically online.
