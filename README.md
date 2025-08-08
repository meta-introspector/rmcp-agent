# RMCP Agent

A powerful Rust library that extends [langchain-rust](https://github.com/Abraxas-365/langchain-rust) with support for Model Context Protocol (MCP) tool integration and streaming tool execution capabilities.

## Features

- **🔗 Langchain-rust Extension**: Built as an extension to the langchain-rust ecosystem, providing seamless integration with existing langchain workflows
- **⚡ RMCP Tool Integration**: Native support for Model Context Protocol (MCP) tools via SSE transport, enabling AI agents to interact with mathematical calculations and custom services
- **🌊 Streaming Tool Execution**: Real-time streaming of tool execution results with detailed progress tracking and error handling
- **🔧 Flexible Agent Builder**: Easy-to-use builder pattern for constructing agents with custom instructions and tool configurations
- **🎯 OpenAI Compatible**: Optimized for OpenAI models with streaming tool calling support
- **📊 Built-in Mathematical Tools**: Includes demonstration tools for addition, subtraction, and factorial calculations
- **🚦 Anti-Repetition Control**: Advanced prompt engineering to prevent redundant tool calls and ensure efficient execution
- **📋 Comprehensive Logging**: Detailed logging and monitoring of tool execution with result summaries

## Quick Start

### Running the Examples

The repository includes comprehensive examples that demonstrate the capabilities of RMCP Agent:

#### 1. MCP Demo Server

Start the demo server that provides basic mathematical tools:

```bash
cargo run --example rmcp_demo_server
```

This will start a server on `http://127.0.0.1:8000` with SSE endpoint at `/sse` providing tools:

- **sum**: Add two integers
- **sub**: Subtract two integers  
- **factorial**: Calculate factorial of a positive integer (1-20)

#### 2. Streaming Tool Usage

Run the streaming client that demonstrates real-time tool execution:

```bash
# Create .env file with your API credentials
cp examples/.env.example examples/.env
# Edit examples/.env with your OPENAI_API_KEY and OPENAI_API_BASE

cargo run --example streaming_with_rmcp_tools
```

This demonstrates the library's capabilities with a real mathematical computation task: "Calculate 3 + 5 - 1, then find the factorial of the result."

### Expected Output

When running the streaming example, you'll see output like:

```text
🚀 Demonstrating RMCP tool usage...

## Execution Plan
1. Calculate the sum of 3 and 5 using the `sum` function.
2. Subtract 1 from the result using the `sub` function.
3. Calculate the factorial of the result using the `factorial` function.

## Tool Selection
- sum function: To add 3 and 5
- sub function: To subtract 1 from the sum
- factorial function: To compute the factorial of the final result

## Task Execution

🏗️  sum calling...
🔧 Tool executed: sum 
💡 Result: 8

🏗️  sub calling...
🔧 Tool executed: sub 
💡 Result: 7

🏗️  factorial calling...
🔧 Tool executed: factorial 
💡 Result: 5040

## Results Summary
Based on tool execution results: sum=8, sub=7, factorial=5040
Task completion status: Successfully completed all calculations
Final answer: The result of 3 + 5 - 1 is 7, and the factorial of 7 is 5040

✅ Execution completed
📋 Tool execution results summary:
   sum (call_abc123): 8
   sub (call_def456): 7
   factorial (call_ghi789): 5040
🎉 Demo completed!
```

### Basic Usage

Here's the complete example from `streaming_with_rmcp_tools.rs`:

```rust
use std::pin::Pin;
use std::sync::Arc;

use futures_util::{Stream, StreamExt};
use langchain_rust::chain::ChainError;
use langchain_rust::prompt_args;
use langchain_rust::schemas::StreamData;
use rmcp::model::{ClientCapabilities, ClientInfo, Implementation, InitializeRequestParam};
use rmcp::service::RunningService;
use rmcp::transport::SseClientTransport;
use rmcp::{RoleClient, ServiceExt};
use rmcp_agent::agent::builder::OpenAIMcpAgentBuilder;
use rmcp_agent::agent::executor::OpenAIMcpAgentExecutor;

#[tokio::main]
async fn main() {
    dotenv::from_path("examples/.env").ok();

    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
    let api_base = std::env::var("OPENAI_API_BASE").expect("OPENAI_API_BASE not set");

    // Initialize MCP client
    let client = init_mcp_client("http://127.0.0.1:8000/sse").await;

    let model = "GPT-4o";

    // Build the agent with custom instructions
    let prefix = "You are a professional AI assistant...";
    let mut agent_builder = OpenAIMcpAgentBuilder::new(api_key, api_base, model)
        .prefix(prefix);

    // Add MCP tools
    let tools = client.list_all_tools().await.unwrap();
    agent_builder = agent_builder.mcp_tools(client.clone(), tools);

    let agent = agent_builder.build().unwrap();

    // Create executor with streaming support
    let executor = OpenAIMcpAgentExecutor::new(Arc::new(agent), model)
        .with_max_iterations(10)
        .with_break_if_error(true);

    // Execute with streaming
    let input_variables = prompt_args! {
        "input" => "Please tell me the result of 3 + 5 - 1, then calculate the factorial of the result."
    };

    let stream = executor.stream(input_variables).await.unwrap();
    print_stream(stream).await;
}

async fn init_mcp_client(
    sse_server_addr: &str,
) -> Arc<RunningService<RoleClient, InitializeRequestParam>> {
    let transport = SseClientTransport::start(sse_server_addr)
        .await
        .expect("Failed to start SSE transport");

    let client_info = ClientInfo {
        protocol_version: Default::default(),
        capabilities: ClientCapabilities::default(),
        client_info: Implementation {
            name: "rmcp-agent demo client".to_string(),
            version: "0.1.0".to_string(),
        },
    };

    Arc::new(
        client_info
            .serve(transport)
            .await
            .expect("Failed to create MCP client"),
    )
}
```

## Creating Your Own MCP Tools

You can create custom MCP tools using the provided server framework. Here's an example from `rmcp_demo_server.rs`:

```rust
use rmcp::{tool, tool_handler, tool_router, ServerHandler};
use rmcp::handler::server::tool::{Parameters, ToolRouter};

#[derive(Debug)]
pub struct McpDemoService {
    tool_router: ToolRouter<Self>,
}

#[tool_router]
impl McpDemoService {
    pub fn new() -> Self {
        Self {
            tool_router: Self::tool_router(),
        }
    }

    #[tool(description = "Adds two integers and returns their sum. Use this for mathematical addition operations. Always pass integer values, not floats. Example: to calculate 3+5, call sum with a=3, b=5")]
    fn sum(&self, Parameters(SumRequest { a, b }): Parameters<SumRequest>) -> String {
        (a + b).to_string()
    }

    #[tool(description = "Calculates factorial of a positive integer (n!). CRITICAL: The parameter 'n' MUST be passed as an integer value (like 7), NOT as a float (like 7.0) or string. Valid range: 1-20. Use this after getting integer results from other calculations. Example: factorial with n=7 calculates 7! = 5040")]
    fn factorial(&self, Parameters(FactorialRequest { n }): Parameters<FactorialRequest>) -> String {
        let mut result = 1u64;
        for i in 1..=n {
            result *= i as u64;
        }
        result.to_string()
    }
}
```

## Architecture

### Core Components

- **`OpenAIMcpAgentBuilder`**: Builder for creating agents with MCP tool integration
- **`OpenAIMcpAgentExecutor`**: Executor that handles streaming tool calls and agent iterations  
- **`IntermediateStep`**: Trait for handling intermediate reasoning steps
- **Tool Integration**: Seamless integration with RMCP tools via Model Context Protocol

### Streaming Flow

```text
User Input → Agent Planning → Tool Calls → Streaming Execution → Results
     ↑                                                            ↓
     └─────────── Iterative Refinement ←────────────────────────┘
```

## Advanced Features

### Custom Prompt Engineering

The library supports sophisticated prompt engineering to control agent behavior:

```rust
let prefix = "
You are a professional AI assistant. For every task, you must strictly follow this workflow:

## MANDATORY WORKFLOW - DO NOT SKIP ANY STEPS:
**Step 1: Create Execution Plan (REQUIRED)**
- MUST start your response with '## Execution Plan'
- List detailed step-by-step execution plan

**Step 2: Tool Selection (REQUIRED)**
- Specify which tools you will use and why

**Step 3: Task Execution (REQUIRED)**
- Execute the plan using selected tools
- **IMPORTANT: After tool execution completes, proceed directly to Step 4**

**Step 4: Results Summary (REQUIRED)**
- Summarize execution results
- Verify task completion status
";

let agent_builder = OpenAIMcpAgentBuilder::new(api_key, api_base, model)
    .prefix(prefix);
```

### Tool Type Safety

The library emphasizes type safety for tool parameters:

```rust
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
struct FactorialRequest {
    #[schemars(description = "Positive integer to calculate factorial for (1-20). MUST be integer type, not float or string")]
    n: i32,
}
```

### Streaming Configuration with Error Handling

```rust
let executor = OpenAIMcpAgentExecutor::new(Arc::new(agent), model)
    .with_max_iterations(10)          // Maximum reasoning iterations
    .with_break_if_error(true);       // Stop on first error

// Execute with comprehensive error handling
let stream = executor.stream(input_variables).await?;
while let Some(chunk) = stream.next().await {
    match chunk {
        Ok(data) => {
            // Process successful streaming data
        }
        Err(e) => {
            eprintln!("Tool execution error: {}", e);
            break;
        }
    }
}
```

### Real-time Tool Monitoring

The library provides detailed real-time feedback:

- **🏗️ Tool Calling**: `sum calling...`  
- **🔧 Tool Results**: `Tool executed: sum - Result: 8`
- **📋 Summary**: Complete execution summary with all tool call results
- **🚨 Error Handling**: Detailed error messages for debugging

## Current Limitations & Roadmap

### 🚧 Current Limitations

- **MCP Transport**: Currently only supports **SSE (Server-Sent Events)** transport for MCP integration
  - Other transport methods (Streamable HTTP, stdio) are planned for future releases
- **Deep Thinking**: Advanced reasoning and deep thinking capabilities are not yet supported
  - Planning to integrate with models that support chain-of-thought and step-by-step reasoning

## Requirements

- Rust 2021 edition or later
- Tokio runtime for async support
- Valid OpenAI API key (or compatible API endpoint)
- MCP server endpoints for tool integration
- Environment variables:
  - `OPENAI_API_KEY`: Your OpenAI API key
  - `OPENAI_API_BASE`: Your OpenAI API base URL (for custom endpoints)

### Dependencies

Key dependencies include:

- `langchain-rust`: Core langchain functionality
- `rmcp`: Model Context Protocol implementation
- `tokio`: Async runtime
- `futures-util`: Stream processing
- `serde`: Serialization support

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built on top of [langchain-rust](https://github.com/Abraxas-365/langchain-rust)
- Powered by [Model Context Protocol (MCP)](https://github.com/modelcontextprotocol)
- Inspired by the need for better streaming tool integration in Rust AI applications
