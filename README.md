# RMCP Agent

A powerful Rust library that extends [langchain-rust](https://github.com/Abraxas-365/langchain-rust) with support for Model Context Protocol (MCP) tool integration and streaming tool execution capabilities.

## Features

- **🔗 Langchain-rust Extension**: Built as an extension to the langchain-rust ecosystem, providing seamless integration with existing langchain workflows
- **⚡ RMCP Tool Integration**: Native support for Model Context Protocol (MCP) tools via SSE transport, enabling AI agents to interact with external services and APIs
- **🌊 Streaming Tool Output**: Real-time streaming of tool execution results, providing immediate feedback during long-running operations
- **🔧 Flexible Agent Builder**: Easy-to-use builder pattern for constructing agents with custom configurations
- **🎯 OpenAI Compatible**: Optimized for OpenAI models with streaming tool calling support

## Quick Start

### Basic Usage

Here's a simple example of how to use RMCP Agent:

```rust
use rmcp_agent::agent::builder::OpenAIMcpAgentBuilder;
use rmcp_agent::agent::executor::OpenAIMcpAgentExecutor;
use langchain_rust::prompt_args;
use std::sync::Arc;

#[tokio::main]
async fn main() {
    // Initialize MCP clients
    let transport = SseClientTransport::start(sse_server_addr)
        .await
        .expect("Failed to start SSE transport");

    let client_info = ClientInfo {
        protocol_version: Default::default(),
        capabilities: ClientCapabilities::default(),
        client_info: Implementation {
            name: "tool_name demo client".to_string(),
            version: "0.0.1".to_string(),
        },
    };

    let client = Arc::new(
        client_info
            .serve(transport)
            .await
            .inspect_err(|e| {
                tracing::error!("client error: {e:?}");
            })
            .expect("Failed to create MCP client"),
    );

    // Build the agent
    let mut agent_builder = OpenAIMcpAgentBuilder::new(
        api_key,
        api_base,
        "gpt-4"
    ).prefix("You are a helpful AI assistant.");

    // Add MCP tools
    let tools = client.list_all_tools().await.unwrap();
    agent_builder = agent_builder.mcp_tools(client.clone(), tools);

    let agent = Arc::new(agent_builder.build().unwrap());

    // Create executor
    let executor = OpenAIMcpAgentExecutor::new(agent, "gpt-4")
        .with_max_iterations(10)
        .with_break_if_error(true);

    // Execute with streaming
    let input = prompt_args! {
        "input" => "Analyze system performance data, detect any anomalies using statistical methods, send a summary report via notification service, and generate a detailed technical report"
    };

    let stream = executor.stream(input).await.unwrap();
    
    // Process streaming results
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(data) => {
                // Handle streaming data
                println!("{data:?}");
            }
            Err(e) => {
                eprintln!("Error: {e}");
                break;
            }
        }
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

### Custom Tool Integration

```rust
// Add custom MCP tools
let tools = client.list_all_tools().await?;
agent_builder = agent_builder.mcp_tools(client.clone(), tools);
```

### Streaming Configuration

```rust
let executor = OpenAIMcpAgentExecutor::new(Arc::new(agent), model)
    .with_max_iterations(10)          // Maximum reasoning iterations
    .with_break_if_error(true);       // Stop on first error
```

### Real-time Tool Monitoring

The library provides real-time feedback on tool execution:

- **⚙️ Tool Calling**: Immediate notification when tools are invoked
- **🔧 Tool Results**: Streaming display of execution results  
- **📋 Summary**: Final summary of all tool executions

## Current Limitations & Roadmap

### 🚧 Current Limitations

- **MCP Transport**: Currently only supports **SSE (Server-Sent Events)** transport for MCP integration
  - Other transport methods (Streamable HTTP, stdio) are planned for future releases
- **Deep Thinking**: Advanced reasoning and deep thinking capabilities are not yet supported
  - Planning to integrate with models that support chain-of-thought and step-by-step reasoning

## Requirements

- Rust 2024 edition
- Tokio runtime for async support
- Valid OpenAI API key (or compatible API)
- MCP server endpoints for tool integration

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built on top of [langchain-rust](https://github.com/Abraxas-365/langchain-rust)
- Powered by [Model Context Protocol (MCP)](https://github.com/modelcontextprotocol)
- Inspired by the need for better streaming tool integration in Rust AI applications
