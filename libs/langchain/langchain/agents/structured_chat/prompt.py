# flake8: noqa
PREFIX = """Respond to the human as helpfully and accurately as possible. You have access to the following tools:"""
FORMAT_INSTRUCTIONS = """Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).

Valid "action" values: "Final Answer" or {tool_names}

Provide only ONE action per $JSON_BLOB, as shown:

```
{{{{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}}}}
```

Follow this format:

Question: input question to answer
Thought: consider previous and subsequent steps
Action:
```
$JSON_BLOB
```
Observation: action result
... (repeat Thought/Action/Observation N times)
Thought: I know what to respond
Action:
```
{{{{
  "action": "Final Answer",
  "action_input": "Final response to human"
}}}}
```"""
SUFFIX = """Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools if necessary. Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation:.
Thought:"""

# Non-langchain native prompts added below

FIX_ACTION = """Complete and return the json blob below by adding a detailed action_input key (tool input). 

Only modify the $INPUT value, otherwise do not change any other formatting.

```
{{{{
  "action": {tool_name},
  "action_input": $INPUT
}}}}
```

Use the following tool definition and raw input text to generate the valid action_input key (tool input).

-------------------------
Tool Definition:

{tool_def}
__________________________

-------------------------
Raw Input:

{input}
__________________________
"""