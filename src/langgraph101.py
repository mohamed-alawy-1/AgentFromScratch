import os
from typing import Literal
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.graph import MessagesState, StateGraph, END, START
from dotenv import load_dotenv
load_dotenv()

@tool
def write_email(to: str, subject: str, body: str) -> str:
    """
    Write an email to the specified recipient with the given subject and body.

    Args:
        to (str): The recipient's email address.
        subject (str): The subject of the email.
        body (str): The body content of the email.

    Returns:
        str: A confirmation message indicating that the email has been sent.
    """
    return f"Email sent to {to} with subject '{subject}' and body '{body}'."

llm = ChatOpenAI(
    model="claude-sonnet-4-6",
    base_url=os.getenv("OPENAI_API_BASE"),
    api_key=os.getenv("OPENAI_API_KEY")
)
model_with_tools = llm.bind_tools([write_email], tool_choice="any")

def call_llm(state: MessagesState) -> MessagesState:
  
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def call_tool(state: MessagesState) -> MessagesState:
    
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        observation = write_email.invoke(tool_call["args"])
        result.append({"role": "tool", "content": observation, "tool_call_id": tool_call["id"]})
    return {"messages": result}

def should_continue(state: MessagesState) -> Literal["run_tool", END]:
    
    # last message
    messages = state["messages"]
    last_message = messages[-1]
    
    if last_message.tool_calls:
        return "run_tool"

    return END

workflow = StateGraph(MessagesState)
workflow.add_node("call_llm", call_llm)
workflow.add_node("run_tool", call_tool)

workflow.add_edge(START, "call_llm")
workflow.add_conditional_edges("call_llm", should_continue, {"run_tool": "run_tool", END: END})
workflow.add_edge("run_tool", END)

app = workflow.compile()

if __name__ == "__main__":
    from langchain_core.messages import HumanMessage

    inputs = {"messages": [HumanMessage(content="Send an email to ahmed@example.com about the project update, saying everything is ready.")]}
    print("Running workflow...")
    output = app.invoke(inputs)

    for message in output["messages"]:
        print(f"\n[{message.type.upper()}]: {message.content}")