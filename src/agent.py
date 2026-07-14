import os
from typing import Literal, TypedDict
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from src.prompts import triage_system_prompt, triage_user_prompt, agent_system_prompt, default_background, default_triage_instructions, default_response_preferences, default_cal_preferences, AGENT_TOOLS_PROMPT
from src.schemas import State, RouterSchema, StateInput
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langgraph.graph import MessagesState
from dotenv import load_dotenv
load_dotenv()

@tool
def write_email(to: str, subject: str, content: str) -> str:
    """Write and send an email."""
    return f"Email sent to {to} with subject '{subject}' and content: {content}"

@tool
def schedule_meeting(attendees: list[str], subject: str, duration_minutes: int, preferred_day: datetime, start_time: int) -> str:
    """Schedule a calendar meeting."""
    date_str = preferred_day.strftime("%A, %B %d, %Y")
    return f"Meeting '{subject}' scheduled on {date_str} at {start_time} for {duration_minutes} minutes with {len(attendees)} attendees"

@tool
def check_calendar_availability(day: str) -> str:
    """Check calendar availability for a given day."""
    return f"Available times on {day}: 9:00 AM, 2:00 PM, 4:00 PM"

class Done(BaseModel):
      """E-mail has been sent."""
      done: bool

class State(MessagesState):
    email_input: dict
    classification_decision: Literal["ignore", "respond", "notify"]

class RouterSchema(BaseModel):
    """Analyze the unread email and route it according to its content."""

    reasoning: str = Field(
        description="Step-by-step reasoning behind the classification."
    )
    classification: Literal["ignore", "respond", "notify"] = Field(
        description="The classification of an email: 'ignore' for irrelevant emails, "
        "'notify' for important information that doesn't need a response, "
        "'respond' for emails that need a reply",
    )
    
class StateInput(TypedDict):
    # This is the input to the state
    email_input: dict

llm = ChatOpenAI(
    model="claude-sonnet-4-6",
    base_url=os.getenv("OPENAI_API_BASE"),
    api_key=os.getenv("OPENAI_API_KEY")
)

llm_router = llm.with_structured_output(RouterSchema, method="function_calling")
llm_with_tools = llm.bind_tools([write_email, schedule_meeting, check_calendar_availability, Done] , tool_choice="any")

# Create a mapping of tool names to actual tool instances.
tools_by_name = {
    "write_email": write_email,
    "schedule_meeting": schedule_meeting,
    "check_calendar_availability": check_calendar_availability,
}


def format_email_markdown(subject, author, to, email_thread, email_id=None):
    """Format email details into a nicely formatted markdown string for display
    
    Args:
        subject: Email subject
        author: Email sender
        to: Email recipient
        email_thread: Email content
        email_id: Optional email ID (for Gmail API)
    """
    id_section = f"\n**ID**: {email_id}" if email_id else ""
    
    return f"""

**Subject**: {subject}
**From**: {author}
**To**: {to}{id_section}

{email_thread}

---
"""

def parse_email(email_input: dict) -> dict:
    """Parse an email input dictionary.

    Args:
        email_input (dict): Dictionary containing email fields:
            - author: Sender's name and email
            - to: Recipient's name and email
            - subject: Email subject line
            - email_thread: Full email content

    Returns:
        tuple[str, str, str, str]: Tuple containing:
            - author: Sender's name and email
            - to: Recipient's name and email
            - subject: Email subject line
            - email_thread: Full email content
    """
    return (
        email_input["author"],
        email_input["to"],
        email_input["subject"],
        email_input["email_thread"],
    )

# Nodes
def llm_call(state: State):
    """LLM decides whether to call a tool or not"""

    return {
        "messages": [
            llm_with_tools.invoke(
                [
                    {"role": "system", "content": agent_system_prompt.format(
                        tools_prompt=AGENT_TOOLS_PROMPT,
                        background=default_background,
                        response_preferences=default_response_preferences, 
                        cal_preferences=default_cal_preferences)
                    },
                    
                ]
                + state["messages"]
            )
        ]
    }

def tool_node(state: State):
    """Performs the tool call"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append({"role": "tool", "content" : observation, "tool_call_id": tool_call["id"]})
    return {"messages": result}

# Conditional edge function
def should_continue(state: State) -> Literal["Action", "__end__"]:
    """Route to Action, or end if Done tool called"""
    messages = state["messages"]
    if not messages:
        return END
    last_message = messages[-1]
    if not getattr(last_message, "tool_calls", None):
        return END
    if any(tool_call["name"] == "Done" for tool_call in last_message.tool_calls):
        return END
    return "Action"

# Build workflow
agent_builder = StateGraph(State)

agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("environment", tool_node)

agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges("llm_call", should_continue, {"Action": "environment", END: END})
agent_builder.add_edge("environment", "llm_call")

# Compile the agent
agent = agent_builder.compile()

def triage_router(state: State) -> Command[Literal["response_agent", "__end__"]]:

    author, to, subject, email_thread = parse_email(state["email_input"])
    system_prompt = triage_system_prompt.format(
        background=default_background,
        triage_instructions=default_triage_instructions
    )

    user_prompt = triage_user_prompt.format(
        author=author, to=to, subject=subject, email_thread=email_thread
    )

    # Create email markdown for Agent Inbox in case of notification  
    email_markdown = format_email_markdown(subject, author, to, email_thread)

    # Run the router LLM
    result = llm_router.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    # Decision
    classification = result.classification

    if classification == "respond":
        print("Classification: RESPOND")
        goto = "response_agent"
        # Add the email to the messages
        update = {
            "classification_decision": result.classification,
            "messages": [{"role": "user",
                            "content": f"Respond to the email: {email_markdown}"
                        }],
        }
    elif result.classification == "ignore":
        print("Classification: IGNORE")
        update =  {
            "classification_decision": result.classification,
        }
        goto = END
    elif result.classification == "notify":

        print("Classification: NOTIFY")
        update = {
            "classification_decision": result.classification,
        }
        goto = END
    else:
        raise ValueError(f"Invalid classification: {result.classification}")
    return Command(goto=goto, update=update)

overall_workflow = (
    StateGraph(State, input=StateInput)
    .add_node(triage_router)
    .add_node("response_agent", agent)
    .add_edge(START, "triage_router")
    .add_edge("response_agent", END)
)

email_assistant = overall_workflow.compile()