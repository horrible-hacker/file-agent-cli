"""
Command Classifier Tool
Uses an LLM with web search to classify shell commands as safe or dangerous
"""

import ollama
import json
import re


CLASSIFIER_PROMPT = """You are a security expert analyzing shell commands.

Your job: Determine if a command is SAFE or DANGEROUS based on what it does.

Consider these risk factors:
- Does it delete/modify files? (rm, mv, cp, sed -i)
- Does it change permissions? (chmod, chown)
- Does it restart/shutdown system? (reboot, shutdown, systemctl stop/restart)
- Does it install/download from internet? (npm install, pip install, curl, wget)
- Does it run as root/admin? (sudo, su)
- Does it modify system config? (passwd, useradd, systemctl)
- Does it access sensitive data? (cat /etc/passwd, /root/...)
- Or is it just reading info? (ls, cat, grep, git log, npm list)

If you don't recognize the command, search the web to understand it first.

Command to analyze: {command}

Respond with ONLY valid JSON (no markdown, no explanation):
{{
  "classification": "safe" or "dangerous",
  "reason": "one sentence explanation of why",
  "searched_web": true or false
}}
"""


CLASSIFIER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web to understand what a command does",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for (e.g., 'systemctl restart command' or 'jq JSON processor')"
                    }
                },
                "required": ["query"]
            }
        }
    }
]


def web_search(query: str) -> str:
    mock_database = {
        "systemctl": "systemctl is a system service manager for Linux systems. It controls services (start, stop, restart). systemctl restart SERVICENAME restarts a service which interrupts traffic.",
        "jq": "jq is a lightweight JSON processor. It parses and filters JSON data. It is read-only, does not modify files.",
        "curl": "curl is a tool to transfer data using URLs. It downloads files from the internet. Can execute downloaded scripts.",
        "wget": "wget is a network downloader. Downloads files from the internet. Can execute downloaded scripts.",
        "sed": "sed is a stream editor for filtering and transforming text. Can modify files if used with -i flag (in-place edit).",
        "awk": "awk is a text processing language. Processes and displays text. Read-only unless piped to modify.",
        "docker": "docker is a container platform. Docker run executes containers which can do anything inside them.",
        "npm install": "npm install downloads and installs Node.js packages from npm registry. Executes package installation scripts.",
        "pip install": "pip install downloads and installs Python packages from PyPI. Executes package installation scripts.",
        "git push": "git push uploads local commits to a remote repository. Modifies the remote repository.",
        "git pull": "git pull downloads commits from remote and merges them. Modifies local repository.",
        "chmod": "chmod changes file permissions. Modifies access control on files and directories.",
        "chown": "chown changes file ownership. Modifies who owns files.",
        "sudo": "sudo runs a command with superuser/admin privileges. Dangerous if combined with other commands.",
    }
    query_lower = query.lower()
    for key, description in mock_database.items():
        if key in query_lower:
            return description
    
    return f"No specific information found for '{query}'. Please provide more context."

def classify_command(command: str) -> dict:
    """
    Classify a shell command as safe or dangerous.
    
    Returns:
        {
            "classification": "safe" or "dangerous",
            "reason": "explanation",
            "searched_web": True/False
        }
    """
    
    messages = [
        {"role": "user", "content": CLASSIFIER_PROMPT.format(command=command)}
    ]
    
    print(f"  [Classifier] Analyzing: {command}")
    response = ollama.chat(
        model="qwen3:4b",
        messages=messages,
        tools=CLASSIFIER_TOOLS,
        stream=False
    )
    
    message = response["message"]
    if message.get("tool_calls"):
        for tool_call in message["tool_calls"]:
            if tool_call["function"]["name"] == "web_search":
                query = tool_call["function"]["arguments"]["query"]
                print(f"  [Classifier] Searching web for: {query}")
                search_result = web_search(query)
                print(f"  [Classifier] Found: {search_result[:100]}...")
                messages.append({"role": "assistant", "content": message["content"]})
                messages.append({
                    "role": "tool",
                    "content": search_result
                })
                messages.append({
                    "role": "user",
                    "content": f"Based on this information, classify the command again. Respond with ONLY JSON:\n{CLASSIFIER_PROMPT.format(command=command)}"
                })
                response = ollama.chat(
                    model="qwen3:4b",
                    messages=messages,
                    tools=CLASSIFIER_TOOLS,
                    stream=False
                )
                message = response["message"]
    
    content = message.get("content", "").strip()
    
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()
    
    try:
        result = json.loads(content)
        result["searched_web"] = len([tc for tc in (message.get("tool_calls") or [])]) > 0
        return result
    except json.JSONDecodeError:
        print(f"  [Classifier] Failed to parse response: {content}")
        return {
            "classification": "dangerous",
            "reason": "Could not classify - treating as dangerous for safety",
            "searched_web": False
        }

