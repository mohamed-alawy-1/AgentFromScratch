import os
from typing import Literal, TypedDict
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.graph import MessagesState
from dotenv import load_dotenv
from src.prompts import (
    triage_system_prompt,
    triage_user_prompt,
    agent_system_prompt_hitl,
    default_background,
    default_triage_instructions,
    default_response_preferences,
    default_cal_preferences,
    HITL_TOOLS_PROMPT
)

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

@tool
def Question(content: str) -> str:
    """Ask the user any follow-up questions."""
    return f"Question asked: {content}"

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
    email_input: dict

def format_for_display(tool_call) -> str:
    """Format a tool call for display to the user in human-in-the-loop review."""
    name = getattr(tool_call, "name", None) or tool_call.get("name", "")
    args = getattr(tool_call, "args", None) or tool_call.get("args", {})

    formatted_args = []
    for k, v in args.items():
        if isinstance(v, list):
            v_str = ", ".join(str(item) for item in v)
        else:
            v_str = str(v)
        formatted_args.append(f"  * {k}: {v_str}")

    args_section = "\n".join(formatted_args)
    return f"\n\n**Proposed Tool Call**: `{name}`\n**Arguments**:\n{args_section}\n"

def parse_email(email_input: dict) -> dict:
    """Extract email details from the input dictionary."""
    return (
        email_input["author"],
        email_input["to"],
        email_input["subject"],
        email_input["email_thread"],
    )

def format_email_markdown(subject, author, to, email_thread, email_id=None):

    id_section = f"\n**ID**: {email_id}" if email_id else ""

    return f"""

**Subject**: {subject}
**From**: {author}
**To**: {to}{id_section}

{email_thread}

---
"""

llm = ChatOpenAI(
    model="claude-sonnet-4-6",
    base_url=os.getenv("OPENAI_API_BASE"),
    api_key=os.getenv("OPENAI_API_KEY")
)

llm_router = llm.with_structured_output(RouterSchema, method="function_calling")
llm_with_tools = llm.bind_tools([write_email, schedule_meeting, check_calendar_availability, Question, Done], tool_choice="any")

# Create a mapping of tool names to actual tool instances.
tools_by_name = {
    "write_email": write_email,
    "schedule_meeting": schedule_meeting,
    "check_calendar_availability": check_calendar_availability,
    "Question": Question,
}

# Nodes
def triage_router(state: State) -> Command[Literal["triage_interrupt_handler", "response_agent", "__end__"]]:

    author, to, subject, email_thread = parse_email(state["email_input"])
    user_prompt = triage_user_prompt.format(
        author=author, to=to, subject=subject, email_thread=email_thread
    )

    email_markdown = format_email_markdown(subject, author, to, email_thread)

    system_prompt = triage_system_prompt.format(
        background=default_background,
        triage_instructions=default_triage_instructions
    )

    result = llm_router.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    classification = result.classification

    if classification == "respond":
        print("Classification: RESPOND")
        goto = "response_agent"
        update = {
            "classification_decision": classification,
            "messages": [{"role": "user",
                            "content": f"Respond to the email: {email_markdown}"
                        }],
        }
    elif classification == "ignore":
        print("Classification: IGNORE")

        goto = END
        update = {
            "classification_decision": classification,
        }

    elif classification == "notify":
        print("Classification: NOTIFY")

        goto = "triage_interrupt_handler"
        update = {
            "classification_decision": classification,
        }

    else:
        raise ValueError(f"Invalid classification: {classification}")
    return Command(goto=goto, update=update)

def triage_interrupt_handler(state: State) -> Command[Literal["response_agent", "__end__"]]:
    """Handles interrupts from the triage step"""

    author, to, subject, email_thread = parse_email(state["email_input"])

    email_markdown = format_email_markdown(subject, author, to, email_thread)

    messages = [{"role": "user",
                "content": f"Email to notify user about: {email_markdown}"
                }]

    # Create interrupt for Agent Inbox
    request = {
        "action_request": {
            "action": f"Email Assistant: {state['classification_decision']}",
            "args": {}
        },
        "config": {
            "allow_ignore": True,
            "allow_respond": True,
            "allow_edit": False,
            "allow_accept": False,
        },
        # Email to show in Agent Inbox
        "description": email_markdown,
    }

    # Agent Inbox responds with a list
    response_raw = interrupt(request)
    response = response_raw[0] if isinstance(response_raw, list) else response_raw

    # If user provides feedback, go to response agent and use feedback to respond to email
    if response["type"] == "response":
        user_input = response["args"]
        messages.append({"role": "user",
                        "content": f"User wants to reply to the email. Use this feedback to respond: {user_input}"
                        })
        goto = "response_agent"

    # If user ignores email, go to END
    elif response["type"] == "ignore":
        goto = END

    else:
        raise ValueError(f"Invalid response: {response}")

    update = {
        "messages": messages,
    }

    return Command(goto=goto, update=update)

def llm_call(state: State):
    """LLM decides whether to call a tool or not"""

    return {
        "messages": [
            llm_with_tools.invoke(
                [
                    {"role": "system", "content": agent_system_prompt_hitl.format(
                        tools_prompt=HITL_TOOLS_PROMPT,
                        background=default_background,
                        response_preferences=default_response_preferences,
                        cal_preferences=default_cal_preferences
                    )}
                ]
                + state["messages"]
            )
        ]
    }

def interrupt_handler(state: State) -> Command[Literal["llm_call", "__end__"]]:
    """Creates an interrupt for human review of tool calls"""

    result = []

    # Go to the LLM call node next
    goto = "llm_call"

    # Iterate over the tool calls in the last message
    for tool_call in state["messages"][-1].tool_calls:

        # Allowed tools for HITL
        hitl_tools = ["write_email", "schedule_meeting", "Question"]

        # If tool is not in our HITL list, execute it directly without interruption
        if tool_call["name"] not in hitl_tools:

            # Execute search_memory and other tools without interruption
            tool = tools_by_name[tool_call["name"]]
            observation = tool.invoke(tool_call["args"])
            result.append({"role": "tool", "content": observation, "tool_call_id": tool_call["id"]})
            continue

        # Get original email from email_input in state
        email_input = state["email_input"]
        author, to, subject, email_thread = parse_email(email_input)
        original_email_markdown = format_email_markdown(subject, author, to, email_thread)

        # Format tool call for display and prepend the original email
        tool_display = format_for_display(tool_call)
        description = original_email_markdown + tool_display

        # Configure what actions are allowed in Agent Inbox
        if tool_call["name"] == "write_email":
            config = {
                "allow_ignore": True,
                "allow_respond": True,
                "allow_edit": True,
                "allow_accept": True,
            }
        elif tool_call["name"] == "schedule_meeting":
            config = {
                "allow_ignore": True,
                "allow_respond": True,
                "allow_edit": True,
                "allow_accept": True,
            }
        elif tool_call["name"] == "Question":
            config = {
                "allow_ignore": True,
                "allow_respond": True,
                "allow_edit": False,
                "allow_accept": False,
            }
        else:
            raise ValueError(f"Invalid tool call: {tool_call['name']}")

        # Create the interrupt request
        request = {
            "action_request": {
                "action": tool_call["name"],
                "args": tool_call["args"]
            },
            "config": config,
            "description": description,
        }

        # Send to Agent Inbox and wait for response
        response_raw = interrupt(request)
        response = response_raw[0] if isinstance(response_raw, list) else response_raw

        # Handle the responses
        if response["type"] == "accept":

            # Execute the tool with original args
            tool = tools_by_name[tool_call["name"]]
            observation = tool.invoke(tool_call["args"])
            result.append({"role": "tool", "content": observation, "tool_call_id": tool_call["id"]})

        elif response["type"] == "edit":

            # Tool selection
            tool = tools_by_name[tool_call["name"]]

            # Get edited args from Agent Inbox
            edited_args = response["args"]["args"]

            # Update the AI message's tool call with edited content (reference to the message in the state)
            ai_message = state["messages"][-1] 
            current_id = tool_call["id"]

            updated_tool_calls = [tc for tc in ai_message.tool_calls if tc["id"] != current_id] + [
                {"type": "tool_call", "name": tool_call["name"], "args": edited_args, "id": current_id}
            ]

            result.append(ai_message.model_copy(update={"tool_calls": updated_tool_calls}))

            # Update the write_email tool call with the edited content from Agent Inbox
            if tool_call["name"] == "write_email":

                # Execute the tool with edited args
                observation = tool.invoke(edited_args)

                # Add only the tool response message
                result.append({"role": "tool", "content": observation, "tool_call_id": current_id})

            elif tool_call["name"] == "schedule_meeting":

                observation = tool.invoke(edited_args)

                result.append({"role": "tool", "content": observation, "tool_call_id": current_id})

            else:
                raise ValueError(f"Invalid tool call: {tool_call['name']}")

        elif response["type"] == "ignore":
            if tool_call["name"] == "write_email":
                result.append({"role": "tool", "content": "User ignored this email draft. Ignore this email and end the workflow.", "tool_call_id": tool_call["id"]})
                goto = END
            elif tool_call["name"] == "schedule_meeting":
                result.append({"role": "tool", "content": "User ignored this calendar meeting draft. Ignore this email and end the workflow.", "tool_call_id": tool_call["id"]})
                goto = END
            elif tool_call["name"] == "Question":
                result.append({"role": "tool", "content": "User ignored this question. Ignore this email and end the workflow.", "tool_call_id": tool_call["id"]})
                goto = END
            else:
                raise ValueError(f"Invalid tool call: {tool_call['name']}")

        elif response["type"] == "response":
            user_feedback = response["args"]
            if tool_call["name"] == "write_email":
                result.append({"role": "tool", "content": f"User gave feedback, which can we incorporate into the email. Feedback: {user_feedback}", "tool_call_id": tool_call["id"]})
            elif tool_call["name"] == "schedule_meeting":
                result.append({"role": "tool", "content": f"User gave feedback, which can we incorporate into the meeting request. Feedback: {user_feedback}", "tool_call_id": tool_call["id"]})
            elif tool_call["name"] == "Question":
                result.append({"role": "tool", "content": f"User answered the question, which can we can use for any follow up actions. Feedback: {user_feedback}", "tool_call_id": tool_call["id"]})
            else:
                raise ValueError(f"Invalid tool call: {tool_call['name']}")

        # Catch all other responses
        else:
            raise ValueError(f"Invalid response: {response}")

    # Update the state
    update = {
        "messages": result,
    }

    return Command(goto=goto, update=update)

# Conditional edge function
def should_continue(state: State) -> Literal["interrupt_handler", "__end__"]:
    """Route to tool handler, or end if Done tool called"""
    messages = state["messages"]
    if not messages:
        return END
    last_message = messages[-1]
    if not getattr(last_message, "tool_calls", None):
        return END
    if any(tool_call["name"] == "Done" for tool_call in last_message.tool_calls):
        return END
    return "interrupt_handler"

# Build workflow
agent_builder = StateGraph(State)

agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("interrupt_handler", interrupt_handler)

agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        "interrupt_handler": "interrupt_handler",
        END: END,
    },
)

response_agent = agent_builder.compile()

overall_workflow = (
    StateGraph(State, input=StateInput)
    .add_node(triage_router)
    .add_node(triage_interrupt_handler)
    .add_node("response_agent", response_agent)
    .add_edge(START, "triage_router")
    .add_edge("response_agent", END)
)

email_assistant = overall_workflow.compile()
