use langchain_rust::{
    agent::AgentError,
    schemas::{AgentAction, FunctionCallResponse, LogTools, Message},
};
use serde_json::json;

pub trait IntermediateStep: Send + Sync {
    fn append_to_conversation(&self, thoughts: &mut Vec<Message>) -> Result<(), AgentError>;
}

impl IntermediateStep for (AgentAction, String) {
    fn append_to_conversation(&self, thoughts: &mut Vec<Message>) -> Result<(), AgentError> {
        let (action, observation) = (&self.0, &self.1);

        let LogTools { tool_id, tools } = match serde_json::from_str(&action.log) {
            Ok(log_tools) => log_tools,
            Err(e) => return Err(AgentError::SerdeJsonError(e)),
        };

        let tools: Vec<FunctionCallResponse> = match serde_json::from_str(&tools) {
            Ok(tools) => tools,
            Err(e) => return Err(AgentError::SerdeJsonError(e)),
        };

        if thoughts.is_empty() {
            thoughts.push(Message::new_ai_message("").with_tool_calls(json!(tools)));
        }
        thoughts.push(Message::new_tool_message(observation, tool_id));

        Ok(())
    }
}
