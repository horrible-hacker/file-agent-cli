import ollama
import subprocess
from agent_cli.classifier_tool import classify_command
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from typing_extensions import TypedDict, Annotated
import operator


def run_shell(command: str) -> str:
    """Execute shell command"""
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        return f"Error: {result.stderr}"
    return result.stdout

AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Run shell commands on your computer",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to run"
                    }
                },
                "required": ["command"]
            }
        }
    }
]

SYSTEM_PROMPT = """You are a helpful computer assistant.
When the user asks you to do something, use the run_shell tool to actually execute it.
Always use tools - never just describe what you would do."""


class State(TypedDict):
    messages: Annotated[list, operator.add] 
    pending_cmd: str | None
    classification: dict | None


def agent_node(state: State) -> State:
    """Main agent - calls LLM to generate response or tool call"""
    
    messages = state["messages"]
    response = ollama.chat(
        model="qwen3:4b",
        messages=messages,
        tools=AGENT_TOOLS,
        stream=False
    )
    
    message = response["message"]
    
    if message.get("tool_calls"):
        command = message["tool_calls"][0]["function"]["arguments"]["command"]
        
        return {
            "messages": [message],
            "pending_cmd": command,
            "classification": None
        }
    else:
        print(f"\nAgent: {message['content']}\n")
        
        return {
            "messages": [message],
            "pending_cmd": None
        }

def classify_node(state: State) -> State:
    """Classify the pending command as safe or dangerous"""
    command = state["pending_cmd"]
    print(f" Classifying: {command}")
    classification = classify_command(command)
    print(f"  ✓ {classification['classification'].upper()}: {classification['reason']}")
    return {
        "classification": classification
    }

def approval_node(state: State) -> State:
    """Pause graph using interrupt to ask user for approval"""
    
    command = state["pending_cmd"]
    reason = state["classification"]["reason"]
    approval = interrupt({
        "type": "approval_needed",
        "command": command,
        "reason": reason
    })
    if approval.lower() == "yes":
        print(f"Executing...")
        result = run_shell(command)
        print(f" Done\n")
        return {
            "messages": [{
                "role": "tool",
                "content": result
            }],
            "pending_cmd": None
        }
    else:
        print(f" Operation rejected\n")
        return {
            "messages": [{
                "role": "tool",
                "content": "User rejected the operation"
            }],
            "pending_cmd": None
        }

def execute_safe_node(state: State) -> State:
    """Auto-execute safe commands"""
    command = state["pending_cmd"]
    reason = state["classification"]["reason"]
    print(f"SAFE: {reason}")
    print(f"Auto-executing...")
    result = run_shell(command)
    print(f"Done\n")
    if result:
        print(result)
    
    return {
        "messages": [{
            "role": "tool",
            "content": result
        }],
        "pending_cmd": None
    }



workflow = StateGraph(State)

workflow.add_node("agent", agent_node)
workflow.add_node("classify", classify_node)
workflow.add_node("approval", approval_node)
workflow.add_node("execute_safe", execute_safe_node)

def route_after_agent(state: State):
    """After agent generates response, route based on tool call"""
    if state.get("pending_cmd"):
        return "classify"
    else:
        return END

def route_after_classify(state: State):
    """After classification, decide if approval needed"""
    if state["classification"]["classification"] == "safe":
        return "execute_safe"
    else:
        return "approval"

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", route_after_agent)
workflow.add_conditional_edges("classify", route_after_classify)
workflow.add_edge("execute_safe", "agent")
workflow.add_edge("approval", "agent")

graph = workflow.compile()


initial_messages = [{"role": "system", "content": SYSTEM_PROMPT}]

print("=" * 70)
print("Agent Ready")
print("=" * 70)
print("Type 'exit' to quit\n")

state = {
    "messages": initial_messages,
    "pending_cmd": None,
    "classification": None
}

while True:
    user_input = input("You: ").strip()
    
    if user_input.lower() == "exit":
        print("Goodbye!")
        break
    
    if not user_input:
        continue
    
    state["messages"] = [{"role": "user", "content": user_input}]
    
    try:
        state = graph.invoke(state)
        
    except Exception as e:
        if hasattr(e, 'args') and e.args and isinstance(e.args[0], dict):
            interrupt_data = e.args[0]
            
            if interrupt_data.get("type") == "approval_needed":
                command = interrupt_data["command"]
                reason = interrupt_data["reason"]
                
                print(f"\nDANGEROUS OPERATION")
                print(f"Command: {command}")
                print(f"Risk: {reason}")
                print(f"\nApprove? (yes/no): ", end="")
                
                approval = input().strip().lower()
                
                state = graph.invoke(state, resume_input=approval)