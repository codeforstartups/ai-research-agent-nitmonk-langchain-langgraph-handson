"""Model Context Protocol (MCP) layer.

MCP is the *external capability boundary* of the agent. Instead of importing
tool functions directly, the agent connects to MCP **servers** through an MCP
**client** and discovers their capabilities at runtime.

MCP has three primitives, distinguished by *who controls their use*:

* **tools** — model-controlled actions (``search_web``, ``calculate``),
* **resources** — application-controlled context (``research://source/{id}``),
* **prompts** — user-controlled reusable templates (``compare_competitors``).

Servers live in ``mcp/servers/`` (analysis, web_research, knowledge); the client
lives in ``mcp/client/``. This package uses the official MCP Python SDK v2.
"""
