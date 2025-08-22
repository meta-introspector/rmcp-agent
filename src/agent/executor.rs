use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use chrono::Utc;
use futures_util::Stream;
use langchain_rust::agent::AgentError;
use langchain_rust::chain::{Chain, ChainError};
use langchain_rust::language_models::GenerateResult;
use langchain_rust::memory::SimpleMemory;
use langchain_rust::prompt::PromptArgs;
use langchain_rust::schemas::{AgentAction, AgentEvent, BaseMemory, LogTools, Message, StreamData};
use langchain_rust::tools::Tool;
use serde_json::{Value, json};
use tokio::sync::Mutex;
use tokio_stream::wrappers::UnboundedReceiverStream;
use uuid::Uuid;

use crate::agent::extension::{AgentEventChunk, AgentExt, DeltaEvent};

pub struct OpenAIMcpAgentExecutor<A>
where
    A: AgentExt,
{
    agent: Arc<A>,
    max_iterations: Option<i32>,
    break_if_error: bool,

    pub model: String,
    pub memory: Option<Arc<Mutex<dyn BaseMemory>>>,
}

impl<A: AgentExt> OpenAIMcpAgentExecutor<A> {
    pub fn new(agent: Arc<A>, model: impl ToString) -> Self {
        Self {
            agent,
            max_iterations: Some(10),
            break_if_error: false,
            memory: None,
            model: model.to_string(),
        }
    }

    pub fn with_max_iterations(mut self, max_iterations: i32) -> Self {
        self.max_iterations = Some(max_iterations);
        self
    }

    pub fn with_memory(mut self, memory: Arc<Mutex<dyn BaseMemory>>) -> Self {
        self.memory = Some(memory);
        self
    }

    pub fn with_break_if_error(mut self, break_if_error: bool) -> Self {
        self.break_if_error = break_if_error;
        self
    }

    fn get_name_to_tools(&self) -> HashMap<String, Arc<dyn Tool>> {
        let mut name_to_tool = HashMap::new();
        for tool in self.agent.get_tools().iter() {
            tracing::debug!("Loading Tool: {}", tool.name());
            name_to_tool.insert(tool.name().trim().replace(" ", "_"), tool.clone());
        }
        name_to_tool
    }
}

#[async_trait]
impl<A> Chain for OpenAIMcpAgentExecutor<A>
where
    A: AgentExt + 'static,
{
    async fn call(&self, input_variables: PromptArgs) -> Result<GenerateResult, ChainError> {
        let mut input_variables = input_variables.clone();
        let name_to_tools = self.get_name_to_tools();
        let mut steps: Vec<(AgentAction, String)> = Vec::new();
        tracing::debug!("steps: {steps:?}");
        if let Some(memory) = &self.memory {
            let memory = memory.lock().await;
            input_variables.insert("chat_history".to_string(), json!(memory.messages()));
        } else {
            input_variables.insert(
                "chat_history".to_string(),
                json!(SimpleMemory::new().messages()),
            );
        }

        loop {
            let agent_event = self
                .agent
                .plan(&steps, input_variables.clone())
                .await
                .map_err(|e| ChainError::AgentError(format!("Error in agent planning: {e}")))?;

            match agent_event {
                AgentEvent::Action(actions) => {
                    for action in actions {
                        tracing::debug!("Action: {:?}", action.tool_input);
                        let tool = name_to_tools
                            .get(&action.tool.trim().replace(" ", "_"))
                            .ok_or_else(|| {
                                AgentError::ToolError(format!("Tool {} not found", action.tool))
                            })
                            .map_err(|e| ChainError::AgentError(e.to_string()))?;

                        let observation = match tool.call(&action.tool_input).await {
                            Ok(result) => result,
                            Err(err) => {
                                let error_msg = err.to_string();
                                tracing::info!("The tool return the following error: {error_msg}");
                                if self.break_if_error {
                                    return Err(ChainError::AgentError(
                                        AgentError::ToolError(error_msg).to_string(),
                                    ));
                                } else {
                                    format!("The tool return the following error: {error_msg}")
                                }
                            }
                        };

                        steps.push((action, observation));
                    }
                }
                AgentEvent::Finish(finish) => {
                    if let Some(memory) = &self.memory {
                        let mut memory = memory.lock().await;

                        memory.add_user_message(match &input_variables["input"] {
                            // Avoid adding extra quotes to the user input in the history.
                            Value::String(s) => s,
                            x => x, // This is the JSON encoded value.
                        });

                        let mut tools_ai_message_seen: HashMap<String, ()> = HashMap::default();
                        for (action, observation) in steps {
                            let LogTools { tool_id, tools } = serde_json::from_str(&action.log)?;
                            let tools_value = serde_json::from_str(&tools)?;
                            if tools_ai_message_seen.insert(tools, ()).is_none() {
                                memory.add_message(
                                    Message::new_ai_message("").with_tool_calls(tools_value),
                                );
                            }
                            memory.add_message(Message::new_tool_message(observation, tool_id));
                        }

                        memory.add_ai_message(&finish.output);
                    }

                    return Ok(GenerateResult {
                        generation: finish.output,
                        ..Default::default()
                    });
                }
            }

            if let Some(max_iterations) = self.max_iterations {
                if steps.len() >= max_iterations as usize {
                    return Ok(GenerateResult {
                        generation: "Max iterations reached".to_string(),
                        ..Default::default()
                    });
                }
            }
        }
    }

    async fn stream(
        &self,
        input_variables: PromptArgs,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamData, ChainError>> + Send>>, ChainError>
    {
        let mut input_variables = input_variables.clone();
        let name_to_tools = self.get_name_to_tools();
        let mut steps: Vec<(AgentAction, String)> = Vec::new();

        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

        if let Some(memory) = &self.memory {
            let memory = memory.lock().await;
            input_variables.insert("chat_history".to_string(), json!(memory.messages()));
        } else {
            input_variables.insert(
                "chat_history".to_string(),
                json!(SimpleMemory::new().messages()),
            );
        }

        let conversation_id = input_variables
            .get("conversation_id")
            .unwrap_or(&Value::String(Uuid::now_v7().to_string()))
            .to_string();

        let chat_completion_id = format!("chatcmpl-{}", Uuid::now_v7());
        let model = self.model.clone();
        let created = Utc::now().timestamp();

        // Send initial chunk
        let _ = tx.send(Ok(StreamData::new(
            json!({
                "id": chat_completion_id,
                "conversation_id": conversation_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": null
                    },
                    "logprobs": null,
                    "finish_reason": null
                }]
            }),
            None,
            "",
        )));

        let agent = self.agent.clone();
        let memory = self.memory.clone();
        let max_iterations = self.max_iterations;
        let break_if_error = self.break_if_error;

        tokio::spawn(async move {
            use futures_util::StreamExt;

            let mut accumulated_content = String::new();
            let mut current_iteration_steps: Vec<(AgentAction, String)> = Vec::new();

            loop {
                accumulated_content.clear();
                current_iteration_steps.clear();

                let mut plan_stream = match agent.plan_stream(&steps, input_variables.clone()).await
                {
                    Ok(stream) => stream,
                    Err(e) => {
                        let _ = tx.send(Ok(StreamData::new(
                            json!({
                                "id": chat_completion_id,
                                "conversation_id": conversation_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {
                                        "content": format!("Error: {e}")
                                    },
                                    "logprobs": null,
                                    "finish_reason": "stop"
                                }]
                            }),
                            None,
                            "",
                        )));
                        return;
                    }
                };

                while let Some(chunk_result) = plan_stream.next().await {
                    match chunk_result {
                        Ok(chunk) => match chunk {
                            AgentEventChunk::Delta(event) => match event {
                                DeltaEvent::Content(content) => {
                                    if !content.is_empty() {
                                        accumulated_content.push_str(&content);

                                        let _ = tx.send(Ok(StreamData::new(
                                            json!({
                                                "id": chat_completion_id,
                                                "conversation_id": conversation_id,
                                                "object": "chat.completion.chunk",
                                                "created": created,
                                                "model": model,
                                                "choices": [{
                                                    "index": 0,
                                                    "delta": {
                                                        "content": content
                                                    },
                                                    "logprobs": null,
                                                    "finish_reason": null
                                                }]
                                            }),
                                            None,
                                            content,
                                        )));
                                    }
                                }
                                DeltaEvent::Action(action) => {
                                    // Generate a tool call ID for this partial action
                                    let log: Value = serde_json::from_str(action.log.as_str())
                                        .unwrap_or_default();

                                    let tool_call_id = match log.get("tool_id") {
                                        Some(id) => id
                                            .as_str()
                                            .unwrap_or_else(|| {
                                                tracing::warn!("tool_id is not a string: {}", id);
                                                ""
                                            })
                                            .to_string(),
                                        None => {
                                            let tmp_id = Uuid::now_v7().to_string();
                                            tracing::error!(
                                                "missing `tool_id` in action.log, tmp id: {tmp_id}"
                                            );
                                            println!(
                                                "missing `tool_id` in action.log, tmp id: {tmp_id}"
                                            );
                                            tmp_id
                                        }
                                    };

                                    // Create the tool call JSON structure for the partial action
                                    let tool_call_json = json!({
                                        "id": tool_call_id,
                                        "conversation_id": conversation_id,
                                        "type": "function",
                                        "function": {
                                            "name": action.tool,
                                            "arguments": action.tool_input,
                                        }
                                    });

                                    // Send the partial tool call information
                                    let _ = tx.send(Ok(StreamData::new(
                                        json!({
                                            "id": chat_completion_id,
                                            "conversation_id": conversation_id,
                                            "object": "chat.completion.chunk",
                                            "created": created,
                                            "model": model,
                                            "choices": [{
                                                "index": 0,
                                                "delta": {
                                                    "tool_calls": [tool_call_json]
                                                },
                                                "logprobs": null,
                                                "finish_reason": null
                                            }]
                                        }),
                                        None,
                                        "",
                                    )));
                                }
                            },
                            AgentEventChunk::Final(event) => {
                                tracing::debug!("got event: {event:?}");
                                match event {
                                    AgentEvent::Action(actions) => {
                                        for action in actions {
                                            let tool = match name_to_tools
                                                .get(&action.tool.trim().replace(" ", "_"))
                                            {
                                                Some(tool) => tool,
                                                None => {
                                                    let error_msg =
                                                        format!("Tool {} not found", action.tool);
                                                    let _ = tx.send(Ok(StreamData::new(
                                                        json!({
                                                            "id": chat_completion_id,
                                                            "conversation_id": conversation_id,
                                                            "object": "chat.completion.chunk",
                                                            "created": created,
                                                            "model": model,
                                                            "choices": [{
                                                                "index": 0,
                                                                "delta": {
                                                                    "content": error_msg
                                                                },
                                                                "logprobs": null,
                                                                "finish_reason": "stop"
                                                            }]
                                                        }),
                                                        None,
                                                        error_msg,
                                                    )));
                                                    return;
                                                }
                                            };

                                            let log: Value = serde_json::from_str(&action.log)
                                                .unwrap_or_default();

                                            let tool_call_id = log
                                                .get("tool_id")
                                                .and_then(|v| v.as_str())
                                                .map(|s| s.to_string())
                                                .unwrap_or_else(|| Uuid::now_v7().to_string());

                                            let tool_call_json = json!({
                                                "id": tool_call_id,
                                                "conversation_id": conversation_id,
                                                "type": "function",
                                                "function": {
                                                    "name": action.tool,
                                                    "arguments": action.tool_input,
                                                }
                                            });

                                            let _ = tx.send(Ok(StreamData::new(
                                                json!({
                                                    "id": chat_completion_id,
                                                    "conversation_id": conversation_id,
                                                    "object": "chat.completion.chunk",
                                                    "created": created,
                                                    "model": model,
                                                    "choices": [{
                                                        "index": 0,
                                                        "delta": {
                                                            "tool_calls": [tool_call_json]
                                                        },
                                                        "logprobs": null,
                                                        "finish_reason": ""
                                                    }]
                                                }),
                                                None,
                                                "",
                                            )));

                                            let observation =
                                                match tool.call(&action.tool_input).await {
                                                    Ok(result) => result,
                                                    Err(err) => {
                                                        let error_msg =
                                                            format!("Tool error: {err}");

                                                        if break_if_error {
                                                            let _ = tx.send(Ok(StreamData::new(
                                                            json!({
                                                                "id": chat_completion_id,
                                                                "conversation_id": conversation_id,
                                                                "object": "chat.completion.chunk",
                                                                "created": created,
                                                                "model": model,
                                                                "choices": [{
                                                                    "index": 0,
                                                                    "delta": {
                                                                        "content": error_msg
                                                                    },
                                                                    "logprobs": null,
                                                                    "finish_reason": "stop"
                                                                }]
                                                            }),
                                                            None,
                                                            error_msg,
                                                        )));
                                                            return;
                                                        } else {
                                                            error_msg
                                                        }
                                                    }
                                                };

                                            let parsed = match serde_json::from_str::<Value>(
                                                &observation,
                                            ) {
                                                Ok(json_result) => json_result,
                                                Err(e) => {
                                                    tracing::warn!(
                                                        "got error in parsing resp: {e:?}, raw: {observation}"
                                                    );
                                                    Value::String(observation.clone())
                                                }
                                            };

                                            let delta = json!({
                                                "content": null,
                                                "parsed": parsed,
                                                "tool_name": tool.name(),
                                                "tool_call_id": tool_call_id
                                            });

                                            let _ = tx.send(Ok(StreamData::new(
                                                json!({
                                                    "id": chat_completion_id,
                                                    "conversation_id": conversation_id,
                                                    "object": "chat.completion.chunk",
                                                    "created": created,
                                                    "model": model,
                                                    "choices": [{
                                                        "index": 0,
                                                        "delta": delta,
                                                        "logprobs": null,
                                                        "finish_reason": null
                                                    }]
                                                }),
                                                None,
                                                parsed.to_string(),
                                            )));

                                            tracing::debug!("observation: {observation}");

                                            current_iteration_steps
                                                .push((action.clone(), observation.clone()));
                                            steps.push((action, observation));
                                        }

                                        if !accumulated_content.is_empty() {
                                            if let Some(memory) = &memory {
                                                let mut memory = memory.lock().await;
                                                memory.add_ai_message(&accumulated_content);
                                            }
                                        }

                                        if let Some(memory) = &memory {
                                            let mut memory = memory.lock().await;
                                            let mut tools_ai_message_seen: HashMap<String, ()> =
                                                HashMap::default();

                                            for (action, observation) in &current_iteration_steps {
                                                match serde_json::from_str::<LogTools>(&action.log)
                                                {
                                                    Ok(LogTools { tool_id, tools }) => {
                                                        if let Ok(tools_value) =
                                                            serde_json::from_str::<Value>(&tools)
                                                        {
                                                            if tools_ai_message_seen
                                                                .insert(tools, ())
                                                                .is_none()
                                                            {
                                                                memory.add_message(
                                                                    Message::new_ai_message("")
                                                                        .with_tool_calls(
                                                                            tools_value,
                                                                        ),
                                                                );
                                                            }
                                                            memory.add_message(
                                                                Message::new_tool_message(
                                                                    observation.clone(),
                                                                    tool_id,
                                                                ),
                                                            );
                                                        } else {
                                                            tracing::warn!(
                                                                "Failed to parse tools JSON: {}",
                                                                tools
                                                            );
                                                        }
                                                    }
                                                    Err(e) => {
                                                        tracing::warn!(
                                                            "Failed to parse action log: {}",
                                                            e
                                                        );
                                                    }
                                                }
                                            }
                                        }

                                        break;
                                    }
                                    AgentEvent::Finish(finish) => {
                                        if let Some(memory) = &memory {
                                            let mut memory = memory.lock().await;

                                            if steps.is_empty()
                                                && current_iteration_steps.is_empty()
                                            {
                                                memory.add_user_message(
                                                    match &input_variables["input"] {
                                                        Value::String(s) => s,
                                                        x => x,
                                                    },
                                                );
                                            }

                                            if !accumulated_content.is_empty() {
                                                memory.add_ai_message(&accumulated_content);
                                            }

                                            let mut tools_ai_message_seen: HashMap<String, ()> =
                                                HashMap::default();
                                            for (action, observation) in &steps {
                                                match serde_json::from_str::<LogTools>(&action.log)
                                                {
                                                    Ok(LogTools { tool_id, tools }) => {
                                                        if let Ok(tools_value) =
                                                            serde_json::from_str::<Value>(&tools)
                                                        {
                                                            if tools_ai_message_seen
                                                                .insert(tools, ())
                                                                .is_none()
                                                            {
                                                                memory.add_message(
                                                                    Message::new_ai_message("")
                                                                        .with_tool_calls(
                                                                            tools_value,
                                                                        ),
                                                                );
                                                            }
                                                            memory.add_message(
                                                                Message::new_tool_message(
                                                                    observation.clone(),
                                                                    tool_id,
                                                                ),
                                                            );
                                                        } else {
                                                            tracing::warn!(
                                                                "Failed to parse tools JSON: {tools}"
                                                            );
                                                        }
                                                    }
                                                    Err(e) => {
                                                        tracing::warn!(
                                                            "Failed to parse action log: {e}"
                                                        );
                                                    }
                                                }
                                            }
                                            memory.add_ai_message(&finish.output);
                                        }

                                        let _ = tx.send(Ok(StreamData::new(
                                            json!({
                                                "id": chat_completion_id,
                                                "conversation_id": conversation_id,
                                                "object": "chat.completion.chunk",
                                                "created": created,
                                                "model": model,
                                                "choices": [{
                                                    "index": 0,
                                                    "delta": {},
                                                    "logprobs": null,
                                                    "finish_reason": "stop"
                                                }]
                                            }),
                                            None,
                                            "stop",
                                        )));
                                        return;
                                    }
                                }
                            }
                        },
                        Err(e) => {
                            let _ = tx.send(Ok(StreamData::new(
                                json!({
                                    "id": chat_completion_id,
                                    "conversation_id": conversation_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": model,
                                    "choices": [{
                                        "index": 0,
                                        "delta": {
                                            "content": format!("Stream error: {e}")
                                        },
                                        "logprobs": null,
                                        "finish_reason": "stop"
                                    }]
                                }),
                                None,
                                "",
                            )));
                            return;
                        }
                    }
                }

                if let Some(memory) = &memory {
                    let memory = memory.lock().await;
                    let messages = memory.messages();
                    input_variables.insert("chat_history".to_string(), json!(messages));
                }

                // Check max iterations before continuing
                if let Some(max_iterations) = max_iterations {
                    if steps.len() >= max_iterations as usize {
                        let _ = tx.send(Ok(StreamData::new(
                            json!({
                                "id": chat_completion_id,
                                "conversation_id": conversation_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {
                                        "content": "Maximum iterations reached."
                                    },
                                    "logprobs": null,
                                    "finish_reason": "length"
                                }]
                            }),
                            None,
                            "Maximum iterations reached.",
                        )));
                        return;
                    }
                }
            }
        });

        Ok(Box::pin(UnboundedReceiverStream::new(rx)))
    }
}
