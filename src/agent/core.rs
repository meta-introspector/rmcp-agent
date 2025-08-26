use std::sync::Arc;

use async_trait::async_trait;
use langchain_rust::agent::{Agent, AgentError};
use langchain_rust::chain::Chain;
use langchain_rust::prompt::{HumanMessagePromptTemplate, MessageFormatterStruct, PromptArgs};
use langchain_rust::schemas::{
    AgentAction, AgentEvent, AgentFinish, FunctionCallResponse, LogTools, Message,
};
use langchain_rust::tools::Tool;
use langchain_rust::{
    fmt_message, fmt_placeholder, fmt_template, message_formatter, template_jinja2,
};
use serde_json::json;

use crate::agent::extension::{AgentEventChunk, AgentExt, AgentStream, DeltaEvent};
use crate::agent::intermediate::IntermediateStep;

pub struct OpenAIMcpAgent {
    pub chain: Box<dyn Chain>,
    pub tools: Vec<Arc<dyn Tool>>,
}

impl OpenAIMcpAgent {
    pub(crate) fn create_prompt(prefix: &str) -> MessageFormatterStruct {
        let message = Message::new_system_message(prefix);
        let template = HumanMessagePromptTemplate::new(template_jinja2!("{{input}}", "input"));

        message_formatter![
            fmt_message!(message),
            fmt_template!(template),
            fmt_placeholder!("chat_history"),
            fmt_placeholder!("agent_scratchpad")
        ]
    }

    pub fn construct_scratchpad(
        &self,
        intermediate_steps: &[impl IntermediateStep],
    ) -> Result<Vec<Message>, AgentError> {
        let mut thoughts: Vec<Message> = vec![];

        const MAX_STEPS: usize = 5;
        const SUMMARY_THRESHOLD: usize = 10;

        if intermediate_steps.len() > SUMMARY_THRESHOLD {
            let summary_msg = self.create_summary_message(
                &intermediate_steps[..intermediate_steps.len() - MAX_STEPS],
            )?;
            thoughts.push(summary_msg);

            for step in &intermediate_steps[intermediate_steps.len() - MAX_STEPS..] {
                step.append_to_conversation(&mut thoughts)?;
            }
        } else {
            for step in intermediate_steps {
                step.append_to_conversation(&mut thoughts)?;
            }
        }

        Ok(thoughts)
    }

    fn create_summary_message(
        &self,
        old_steps: &[impl IntermediateStep],
    ) -> Result<Message, AgentError> {
        let summary = format!(
            "Previous {} steps summary: [Summarized execution history with {} actions completed]",
            old_steps.len(),
            old_steps.len()
        );

        Ok(Message::new_system_message(&summary))
    }

    fn process_chunk_delta(
        chunk: &serde_json::Value,
        model_output: &mut String,
        tool_call_acc: &mut ToolCallAccumulator,
        has_tool_calls: &mut bool,
    ) -> Vec<AgentEventChunk> {
        let mut events = Vec::new();

        let Some(choices) = chunk.get("choices").and_then(|c| c.as_array()) else {
            return events;
        };

        let Some(choice) = choices.first() else {
            return events;
        };

        let Some(delta) = choice.get("delta") else {
            return events;
        };

        // Handle tool calls
        if let Some(content) = delta.get("content").and_then(|c| c.as_str()) {
            if !content.is_empty() {
                model_output.push_str(content);
                events.push(AgentEventChunk::Delta(DeltaEvent::Content(
                    content.to_string(),
                )));
            }
        } else if let Some(_tool_calls) = delta.get("tool_calls").and_then(|tc| tc.as_array()) {
            *has_tool_calls = true;
            let tool_call_chunk = tool_call_acc.accumulate(delta);
            events.push(AgentEventChunk::Delta(DeltaEvent::Action(tool_call_chunk)));
        }

        // Handle finish reason
        if let Some(finish_reason) = choice.get("finish_reason").and_then(|f| f.as_str()) {
            let final_event = Self::handle_finish_reason(
                finish_reason,
                tool_call_acc,
                *has_tool_calls,
                model_output,
            );
            events.push(AgentEventChunk::Final(final_event));
        }

        events
    }

    fn handle_finish_reason(
        finish_reason: &str,
        tool_call_acc: &mut ToolCallAccumulator,
        has_tool_calls: bool,
        model_output: &str,
    ) -> AgentEvent {
        match finish_reason {
            "tool_calls" => {
                let action = tool_call_acc.take_action();
                AgentEvent::Action(vec![action])
            }
            "stop" => {
                if has_tool_calls {
                    let action = tool_call_acc.take_action();
                    AgentEvent::Action(vec![action])
                } else {
                    AgentEvent::Finish(AgentFinish {
                        output: model_output.to_string(),
                    })
                }
            }
            _ => AgentEvent::Finish(AgentFinish {
                output: model_output.to_string(),
            }),
        }
    }
}

#[async_trait]
impl Agent for OpenAIMcpAgent {
    async fn plan(
        &self,
        intermediate_steps: &[(AgentAction, String)],
        inputs: PromptArgs,
    ) -> Result<AgentEvent, AgentError> {
        let mut inputs = inputs.clone();
        let scratchpad = self.construct_scratchpad(intermediate_steps)?;
        inputs.insert("agent_scratchpad".to_string(), json!(scratchpad));
        let output = self.chain.call(inputs).await?.generation;

        match serde_json::from_str::<Vec<FunctionCallResponse>>(&output) {
            Ok(tools) => {
                let mut actions = Vec::with_capacity(tools.len());
                for tool in tools {
                    //Log tools will be send as log
                    let log = LogTools {
                        tool_id: tool.id,
                        tools: output.clone(),
                    };
                    actions.push(AgentAction {
                        tool: tool.function.name,
                        tool_input: tool.function.arguments,
                        log: serde_json::to_string(&log)?,
                    });
                }
                return Ok(AgentEvent::Action(actions));
            }
            Err(_) => return Ok(AgentEvent::Finish(AgentFinish { output })),
        }
    }

    fn get_tools(&self) -> Vec<Arc<dyn Tool>> {
        self.tools.clone()
    }
}

#[async_trait]
impl AgentExt for OpenAIMcpAgent {
    async fn plan_with_steps(
        &self,
        steps: &[impl IntermediateStep],
        inputs: PromptArgs,
    ) -> Result<AgentEvent, AgentError> {
        let mut inputs = inputs.clone();
        let scratchpad = self.construct_scratchpad(steps)?;
        inputs.insert("agent_scratchpad".to_string(), json!(scratchpad));
        let output = self.chain.call(inputs).await?.generation;

        match serde_json::from_str::<Vec<FunctionCallResponse>>(&output) {
            Ok(tools) => {
                let mut actions = Vec::with_capacity(tools.len());
                for tool in tools {
                    // Log tools will be sent as log
                    let log = LogTools {
                        tool_id: tool.id,
                        tools: output.clone(),
                    };
                    actions.push(AgentAction {
                        tool: tool.function.name,
                        tool_input: tool.function.arguments,
                        log: serde_json::to_string(&log)?,
                    });
                }
                return Ok(AgentEvent::Action(actions));
            }
            Err(_) => return Ok(AgentEvent::Finish(AgentFinish { output })),
        }
    }

    async fn plan_stream(
        &self,
        steps: &[impl IntermediateStep],
        inputs: PromptArgs,
    ) -> Result<AgentStream, AgentError> {
        use async_stream::stream;
        use futures_util::StreamExt;

        let mut inputs = inputs.clone();
        let scratchpad = self.construct_scratchpad(steps)?;
        inputs.insert("agent_scratchpad".to_string(), json!(scratchpad));

        let mut chain_stream = self.chain.stream(inputs).await?;
        let mut model_output = String::new();
        let mut tool_call_acc = ToolCallAccumulator::new();
        let mut has_tool_calls = false;

        let s = stream! {
            while let Some(chunk_result) = chain_stream.next().await {
                let chunk = match chunk_result {
                    Ok(chunk) => chunk,
                    Err(e) => {
                        yield Err(e);
                        return;
                    }
                };

                // Process chunk and get events
                let events = Self::process_chunk_delta(&chunk.value, &mut model_output, &mut tool_call_acc, &mut has_tool_calls);

                for event in events {
                    // Check if this is a final event that should end the stream
                    let is_final = matches!(event, AgentEventChunk::Final(_));
                    yield Ok(event);

                    if is_final {
                        return;
                    }
                }
            }

            match has_tool_calls {
                true => {
                    let action = tool_call_acc.take_action();
                    yield Ok(AgentEventChunk::Final(AgentEvent::Action(vec![action])))
                },
                false => yield Ok(AgentEventChunk::Final(AgentEvent::Finish(AgentFinish { output: model_output })))
            }
        };

        Ok(Box::pin(s) as AgentStream)
    }
}

struct ToolCallAccumulator {
    name: Option<String>,
    args: String,
    id: Option<String>,
}

impl ToolCallAccumulator {
    fn new() -> Self {
        Self {
            name: None,
            args: String::new(),
            id: None,
        }
    }

    fn accumulate(&mut self, delta: &serde_json::Value) -> AgentAction {
        let mut args_chunk = String::default();

        if let Some(tool_call) = delta
            .get("tool_calls")
            .and_then(|v| v.as_array())
            .and_then(|v| v.first())
        {
            if let Some(function) = tool_call.get("function") {
                if let Some(name) = function.get("name").and_then(|n| n.as_str()) {
                    self.name = Some(name.to_string());
                }
                if let Some(args) = function.get("arguments").and_then(|a| a.as_str()) {
                    self.args.push_str(args);
                    args_chunk = args.to_string();
                }
            }
            if let Some(id) = tool_call.get("id").and_then(|i| i.as_str()) {
                if !id.is_empty() {
                    self.id = Some(id.to_string());
                }
            }
        };

        self.to_action_chunk(args_chunk)
    }

    fn take_action(&mut self) -> AgentAction {
        let args = std::mem::take(&mut self.args);
        self.to_action_chunk(args)
    }

    fn to_action_chunk(&self, args_chunk: String) -> AgentAction {
        let processed_args = if args_chunk.trim().is_empty() {
            "{}".to_string()
        } else {
            args_chunk
        };

        let function_call_response = serde_json::json!({
            "id": self.id.clone(),
            "type": "function",
            "function": {
                "name": self.name.clone(),
                "arguments": processed_args
            }
        });

        // Construct tool call array (consistent with non-streaming method)
        let tools_array = serde_json::json!([function_call_response]);
        let tools_output = serde_json::to_string(&tools_array).unwrap_or_else(|_| {
            "[{{\"error\": \"Failed to serialize function call\"}}]".to_string()
        });

        let log_tools = LogTools {
            tool_id: self.id.clone().unwrap_or_default(),
            tools: tools_output,
        };

        let log_str = serde_json::to_string(&log_tools).unwrap_or_else(|_| {
            // If serialization fails, return a simple format
            format!(
                "{{\"tool_id\": \"{}\", \"tools\": \"[]\"}}",
                self.id.clone().unwrap_or_default()
            )
        });

        AgentAction {
            tool: self.name.clone().unwrap_or_default(),
            tool_input: processed_args,
            log: log_str,
        }
    }
}
