# Connecting AI Agents to PSI RAG

This document provides complete, copy-pasteable instructions for connecting external AI agents (like Claude Desktop, Cursor IDE, VS Code MCP client, or agentic frameworks like Google ADK and CrewAI) to the PSI Logistics RAG application.

---

## 1. Model Context Protocol (MCP)

The RAG pipeline exposes two MCP transport mechanisms: **Stdio** (for local development/IDE tools) and **Server-Sent Events (SSE)** (for remote agent integration over HTTP).

### A. Local Integration: MCP via Stdio
To use this with desktop agents or IDEs, configure them to start the MCP server via standard input/output.

#### Cursor IDE Config
Add the following to your Cursor MCP settings (`Settings -> Features -> MCP -> + New MCP Client`):
* **Name**: `PSI-Logistics-RAG`
* **Type**: `command`
* **Command**: `python c:/Users/naman/Downloads/PSI_RAG/scripts/run_mcp_stdio.py`

#### Claude Desktop Config
Add this to your `claude_desktop_config.json` (typically under `%APPDATA%\Claude\claude_desktop_config.json` on Windows):
```json
{
  "mcpServers": {
    "psi-logistics-rag": {
      "command": "python",
      "args": [
        "c:/Users/naman/Downloads/PSI_RAG/scripts/run_mcp_stdio.py"
      ]
    }
  }
}
```

#### VS Code (MCP Client) Config
For extensions like Cline or Roo Code, configure the command in the MCP settings:
```json
{
  "mcpServers": {
    "psi-logistics-rag": {
      "command": "python",
      "args": ["c:/Users/naman/Downloads/PSI_RAG/scripts/run_mcp_stdio.py"]
    }
  }
}
```

---

### B. Remote Integration: MCP via SSE
For remote servers or web applications, connect via Server-Sent Events (SSE).

* **SSE Endpoint**: `GET /api/v1/mcp/sse`
* **Client Post Messages Endpoint**: `POST /api/v1/mcp/messages`

To integrate programmatically in Python using the MCP SDK:
```python
from mcp import ClientSession
from mcp.client.sse import sse_client

async with sse_client("https://namangt-psi-rag.hf.space/api/v1/mcp/sse") as (read_stream, write_stream):
    async with ClientSession(read_stream, write_stream) as session:
        # Initialize
        await session.initialize()
        
        # Call a tool
        result = await session.call_tool(
            "answer_carrier_question", 
            arguments={"question": "What is the carrier tolerance for Platinum level?"}
        )
        print(result.content[0].text)
```

---

### C. Exposed MCP Tools & Resources

#### Tools
1. **`answer_carrier_question`**
   * **Purpose**: Ask a question to the fully guardrailed, verified RAG QA pipeline.
   * **Parameters**: 
     - `question` (string, required)
     - `username` (string, optional, default: `mcp_client`)
   * **Returns**: Grounded carrier SLA advisor reply or gate refusal.

2. **`search_logistics_docs`**
   * **Purpose**: Query semantic search index for raw matching document snippets.
   * **Parameters**: 
     - `query` (string, required)
     - `top_k` (integer, optional, default: `5`)
   * **Returns**: Formatted text chunks with source names and similarity scores.

#### Resources
* **`resource://carrier_sla`**: Exposes the full markdown content of the Carrier SLA Agreement (DOC1).
* **`resource://customs_tariff`**: Exposes the full markdown content of the Customs Tariff Reference (DOC2).
* **`resource://shipment_delay`**: Exposes the full markdown content of the Shipment Delay Policy (DOC3).

---

## 2. Agent-to-Agent (A2A) Protocol

The RAG application supports the Google Agent-to-Agent (A2A) standard, enabling multi-agent orchestrators to discover this agent's capabilities, delegate tasks, and run them asynchronously.

### A. Discovery (Agent Card)
To fetch the agent's capabilities:
* **Endpoint**: `GET /.well-known/agent-card.json` (redirects to `GET /api/v1/a2a/agent-card`)
* **Response Payload**:
```json
{
  "name": "PSI-Logistics-RAG-Agent",
  "description": "An agent that provides verified, guardrailed carrier SLA advice, tariff definitions, and delay exception guidelines.",
  "version": "1.0.0",
  "url": "/a2a",
  "capabilities": {
    "streaming": false
  },
  "skills": [
    {
      "id": "logistics_sla_advisor",
      "name": "Logistics SLA Advisor",
      "description": "Exposes carrier agreement SLA terms, delay thresholds, compensations, and customs tariff reference queries.",
      "tags": ["logistics", "sla", "customs", "tariffs", "delays"]
    }
  ]
}
```

### B. Task Delegation Workflow

1. **Create a Task**:
   * **Method & Endpoint**: `POST /api/v1/a2a/tasks`
   * **Payload**:
     ```json
     {
       "skillId": "logistics_sla_advisor",
       "input": {
         "question": "What is the customs entry process for shipment delay classification?",
         "username": "external_agent"
       }
     }
     ```
   * **Response**: Returns a unique `taskId` and a status of `"created"`.

2. **Trigger Task Execution**:
   * **Method & Endpoint**: `PUT /api/v1/a2a/tasks/{taskId}/execute`
   * **Response**: Transitions the task state to `"running"` in the background.

3. **Check Task Status & Results**:
   * **Method & Endpoint**: `GET /api/v1/a2a/tasks/{taskId}`
   * **Response**:
     ```json
     {
       "taskId": "task-uuid-here",
       "status": "completed",
       "skillId": "logistics_sla_advisor",
       "input": {
         "question": "...",
         "username": "..."
       },
       "output": "The grounded response generated by the orchestrator...",
       "artifacts": [
         {
           "name": "grounded_response",
           "mimeType": "text/plain",
           "content": "..."
         }
       ],
       "createdAt": "2026-06-13T22:00:00Z"
     }
     ```

---

## 3. Caching, Valkey, and BetterDB Observability

For production reliability and low-latency responses, the RAG pipeline integrates semantic and response caching powered by **Valkey** (the high-performance, open-source successor to Redis).

### Valkey Compatibility
Valkey is a drop-in replacement for Redis and utilizes the standard RESP connection protocol. The RAG pipeline automatically connects to Valkey if configured via the `REDIS_URL` environment variable:
```env
REDIS_URL=redis://localhost:6379/0
```

### BetterDB Telemetry
To achieve full observability, the caching layers print structured telemetry logs. An observability platform like **BetterDB** parses these logs to populate real-time dashboards, perform slowlog analysis, and track cache-hit ratios.

#### Sample Telemetry Log Structure
```text
INFO: ValkeyRedis Command Telemetry | Command: GET | Key: resp:3c2f0... | Latency: 1.15ms | Hit: True
INFO: ValkeyRedis Command Telemetry | Command: SET | Key: resp:a5f1d... | Latency: 2.34ms | TTL: 3600s
```

#### BetterDB Monitoring Features
1. **Real-time Dashboards**: Monitor caching performance, total requests, active clients, and memory footprints.
2. **Slowlog Analysis**: Capture slow cache operations (e.g. key eviction or large namespace scans) using Valkey's built-in command tracing.
3. **Client Analytics**: Monitor connection spikes and ACL rules from active agent instances.
4. **Anomaly Detection**: Flags sudden cache invalidations or elevated cache-miss rates.
