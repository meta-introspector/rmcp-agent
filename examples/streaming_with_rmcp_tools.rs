use std::pin::Pin;
use std::sync::Arc;

use futures_util::{Stream, StreamExt};
use langchain_rust::chain::{Chain, ChainError};
use langchain_rust::prompt_args;
use langchain_rust::schemas::StreamData;
use rmcp::model::{ClientCapabilities, ClientInfo, Implementation, InitializeRequestParam};
use rmcp::service::RunningService;
use rmcp::transport::SseClientTransport;
use rmcp::{RoleClient, ServiceExt};
use rmcp_agent::agent::builder::OpenAIMcpAgentBuilder;
use rmcp_agent::agent::executor::OpenAIMcpAgentExecutor;
use tokio::io::AsyncWriteExt;

#[tokio::main]
async fn main() {
    dotenv::from_path("examples/.env").ok();

    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
    let api_base = std::env::var("OPENAI_API_BASE").expect("OPENAI_API_BASE not set");

    let client = init_mcp_client("http://127.0.0.1:8000/sse").await;

    let model = "GPT-4o";

    let prefix = "
You are a professional AI assistant. For every task, you must strictly follow this workflow:

## MANDATORY WORKFLOW - DO NOT SKIP ANY STEPS:

**Step 1: Create Execution Plan (REQUIRED)**
- MUST start your response with '## Execution Plan'
- List detailed step-by-step execution plan
- Clearly number each step
- Explain what you will do and why

**Step 2: Tool Selection (REQUIRED)**
- Specify which tools you will use and why
- Provide reasonable justification for your tool choices

**Step 3: Task Execution (REQUIRED)**
- Execute the plan using selected tools
- Clearly demonstrate your work process
- **IMPORTANT: After tool execution completes, proceed directly to Step 4, do NOT restart planning**

**Step 4: Results Summary (REQUIRED)**
- Summarize execution results
- Verify task completion status
- Provide final answer

## CRITICAL EXECUTION RULES:
1. **MUST start with execution plan - NO EXCEPTIONS**
2. **MUST show your plan before doing anything else**
3. **NEVER call tools directly without showing plan first**
4. **EXECUTE EACH STEP ONLY ONCE - NO REPETITION**
5. **FOLLOW THE PLAN SEQUENTIALLY - Complete step 1, then step 2, then step 3, etc.**
6. **DO NOT REPEAT THE SAME TOOL CALL MULTIPLE TIMES**
7. **After tool execution, summarize results directly, do NOT repeat planning steps**
8. **Only ONE execution plan per task, NO repetition**
9. **Minimize the number of tool calls**

## STEP-BY-STEP EXECUTION CONTROL:
- **Sequential Execution**: Execute plan steps in exact order (1→2→3→...)
- **One Call Per Step**: Each planned step should result in exactly ONE tool call
- **Progress Tracking**: Keep track of which step you are currently executing
- **Result Validation**: After each tool call, verify the result before proceeding to next step
- **No Backtracking**: Do not repeat previous steps unless there was an error

## TOOL USAGE GUIDELINES:
- **Data Type Precision**: Always ensure parameter types match tool expectations exactly
- **Integer vs Number**: Use integer values (e.g., 7) for tools expecting integers, not floating point numbers (e.g., 7.0)
- **Type Conversion**: When using results from previous tool calls as inputs, ensure proper type conversion:
  - If a tool returns a number but the next tool expects an integer, convert it explicitly
  - For factorial function: ALWAYS use integer format (e.g., 7, not 7.0 or string)
  - For sum/sub functions: Both parameters should be integers when dealing with whole numbers
- **Parameter Validation**: Before calling a tool, verify that all parameters match the expected data types in the tool schema
- **Error Handling**: If a tool call fails due to type mismatch, analyze the error and retry with correct data types
- **Chain Tool Calls**: When chaining tool calls, ensure the output of one tool is properly formatted as input for the next tool
- **Use Previous Results**: When a step depends on previous results, use the actual output from the previous tool call but ensure correct type formatting
- **Specific Tool Requirements**:
  - factorial function: Parameter must be an integer (not float, not string)
  - sum function: Both a and b parameters must be integers
  - sub function: Both a and b parameters must be integers

## Response Format Example:
Based on tool execution results: [results]
Task completion status: [verification]
Final answer: [answer]

**KEY REMINDERS:**
- Create execution plan only once at the beginning
- Execute each step exactly once in sequential order
- Use results from previous steps as inputs for next steps
- After tool execution completes, proceed directly to results summary
- Do NOT restart planning after tool execution
- Pay strict attention to parameter data types when calling tools";

    let mut agent_builder = OpenAIMcpAgentBuilder::new(api_key, api_base, model).prefix(prefix);

    let tools = client.list_all_tools().await.unwrap();
    agent_builder = agent_builder.mcp_tools(client.clone(), tools);

    let agent = agent_builder.build().unwrap();

    let executor = OpenAIMcpAgentExecutor::new(Arc::new(agent), model)
        .with_max_iterations(10)
        .with_break_if_error(true);

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
            name: "tool_name demo client".to_string(),
            version: "0.0.1".to_string(),
        },
    };

    Arc::new(
        client_info
            .serve(transport)
            .await
            .inspect_err(|e| {
                tracing::error!("client error: {e:?}");
            })
            .expect("Failed to create MCP client"),
    )
}

async fn print_stream(
    mut stream: Pin<Box<dyn Stream<Item = Result<StreamData, ChainError>> + Send>>,
) {
    let mut tool_results = vec![];
    let mut tool_call_states = std::collections::HashMap::<String, (String, String)>::new(); // ID -> (name, args)
    let mut printed_tool_calls = std::collections::HashSet::<String>::new(); // Track printed tool calls

    let mut stdout = tokio::io::stdout();
    stdout
        .write_all("🚀 Demonstrating RMCP tool usage...\n".as_bytes())
        .await
        .unwrap();

    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(stream_data) => {
                if let Some(choices) = stream_data.value.get("choices").and_then(|c| c.as_array()) {
                    if let Some(choice) = choices.first() {
                        if let Some(delta) = choice.get("delta") {
                            if let Some(tool_calls) =
                                delta.get("tool_calls").and_then(|tc| tc.as_array())
                            {
                                for tool_call in tool_calls {
                                    if let Some(tool_call_id) =
                                        tool_call.get("id").and_then(|id| id.as_str())
                                    {
                                        let clean_id = tool_call_id.trim_matches('"');
                                        let is_new_tool_call =
                                            !tool_call_states.contains_key(clean_id);

                                        if let Some(function) = tool_call.get("function") {
                                            // Get current state or create new
                                            let (current_name, current_args) = tool_call_states
                                                .get(clean_id)
                                                .cloned()
                                                .unwrap_or_default();

                                            // Update name if provided (usually only in first chunk)
                                            let name = if let Some(func_name) =
                                                function.get("name").and_then(|n| n.as_str())
                                            {
                                                func_name.to_string()
                                            } else {
                                                current_name
                                            };

                                            // Update args if provided (accumulate across chunks)
                                            let args = if let Some(func_args) =
                                                function.get("arguments").and_then(|a| a.as_str())
                                            {
                                                current_args + func_args
                                            } else {
                                                current_args
                                            };

                                            // Show "calling" message for new tool calls
                                            if is_new_tool_call && !name.is_empty() {
                                                stdout
                                                    .write_all(
                                                        format!("\n\n🏗️  {name} calling...\n")
                                                            .as_bytes(),
                                                    )
                                                    .await
                                                    .unwrap();
                                            }

                                            // Always update state with the latest information
                                            tool_call_states.insert(
                                                clean_id.to_string(),
                                                (name.clone(), args.clone()),
                                            );
                                        }
                                    }
                                }
                            }

                            if let Some(content) = delta.get("content").and_then(|c| c.as_str()) {
                                stdout.write_all(content.as_bytes()).await.unwrap();
                                stdout.flush().await.unwrap();
                            }

                            if let Some(error) = delta.get("error_message").and_then(|c| c.as_str())
                            {
                                let tool_call_id = delta
                                    .get("tool_call_id")
                                    .and_then(|id| id.as_str())
                                    .unwrap();

                                let tool_name = delta
                                    .get("tool_name")
                                    .and_then(|name| name.as_str())
                                    .unwrap();

                                stdout
                                    .write_all(
                                        format!(
                                            "\n🚨 Error: {error}\n Tool call id: {tool_call_id}, {tool_name}",
                                        )
                                        .as_bytes(),
                                    )
                                    .await
                                    .unwrap();
                                stdout.flush().await.unwrap();
                                break;
                            }

                            if let Some(parsed) = delta.get("parsed") {
                                let tool_call_id =
                                    delta.get("tool_call_id").and_then(|id| id.as_str());
                                let tool_name =
                                    delta.get("tool_name").and_then(|name| name.as_str());

                                match (tool_call_id, tool_name) {
                                    (Some(id), Some(name)) => {
                                        // Parse JSON result for better display
                                        let display_result = if let Ok(json_val) =
                                            serde_json::from_value::<serde_json::Value>(
                                                parsed.clone(),
                                            ) {
                                            if let Some(content) =
                                                json_val.get("content").and_then(|c| c.as_str())
                                            {
                                                content.to_string()
                                            } else if let Some(status) =
                                                json_val.get("status").and_then(|s| s.as_str())
                                            {
                                                if let Some(result) = json_val.get("result") {
                                                    format!("{result} ({status})")
                                                } else {
                                                    status.to_string()
                                                }
                                            } else {
                                                parsed.to_string()
                                            }
                                        } else {
                                            parsed.to_string()
                                        };

                                        stdout
                                            .write_all(
                                                format!(
                                                    "\n🔧 Tool executed: {name} \n💡 Result: {display_result}\n",
                                                )
                                                .as_bytes(),
                                            )
                                            .await
                                            .unwrap();

                                        tool_results.push((
                                            id.to_string(),
                                            name.to_string(),
                                            parsed.clone(),
                                        ));
                                    }
                                    (_id, name) => {
                                        stdout
                                            .write_all(
                                                "\n� Tool executed (incomplete info)\n".as_bytes(),
                                            )
                                            .await
                                            .unwrap();
                                        if let Some(name) = name {
                                            stdout
                                                .write_all(format!("   Tool: {name}\n").as_bytes())
                                                .await
                                                .unwrap();
                                        }
                                        stdout
                                            .write_all(format!("   Result: {parsed}\n").as_bytes())
                                            .await
                                            .unwrap();
                                    }
                                }
                            }
                        }

                        if let Some(finish_reason) =
                            choice.get("finish_reason").and_then(|f| f.as_str())
                        {
                            // When we get a finish_reason, print all accumulated tool calls
                            if finish_reason == "tool_calls" {
                                for (tool_id, (name, args)) in &tool_call_states {
                                    if !name.is_empty() && !printed_tool_calls.contains(tool_id) {
                                        stdout
                                            .write_all(format!("🔧 Tool call: {name}\n").as_bytes())
                                            .await
                                            .unwrap();

                                        stdout
                                            .write_all(
                                                format!("   🆔 Tool Call ID: {tool_id}\n")
                                                    .as_bytes(),
                                            )
                                            .await
                                            .unwrap();

                                        stdout
                                            .write_all(
                                                format!("   📋 Arguments: {args}\n").as_bytes(),
                                            )
                                            .await
                                            .unwrap();

                                        printed_tool_calls.insert(tool_id.clone());
                                    }
                                }
                            }

                            match finish_reason {
                                "stop" => {
                                    stdout
                                        .write_all("\n✅ Execution completed\n".as_bytes())
                                        .await
                                        .unwrap();
                                    break;
                                }
                                "length" => {
                                    stdout
                                        .write_all("\n⚠️ Maximum length reached\n".as_bytes())
                                        .await
                                        .unwrap();
                                    break;
                                }
                                "tool_calls" => {
                                    // Continue processing, don't break
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
            Err(e) => {
                stdout
                    .write_all(format!("❌ Streaming processing error: {e}\n").as_bytes())
                    .await
                    .unwrap();
                break;
            }
        }
    }

    stdout
        .write_all("\n📋 Tool execution results summary:\n".as_bytes())
        .await
        .unwrap();

    for (id, name, result) in &tool_results {
        stdout
            .write_all(format!("   {name} ({id}): {result}\n").as_bytes())
            .await
            .unwrap();
    }

    stdout
        .write_all("\n🎉 Demo completed!\n".as_bytes())
        .await
        .unwrap();
}
