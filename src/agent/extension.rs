use std::pin::Pin;

use async_trait::async_trait;
use futures_util::Stream;
use langchain_rust::agent::{Agent, AgentError};
use langchain_rust::chain::ChainError;
use langchain_rust::prompt::PromptArgs;
use langchain_rust::schemas::{AgentAction, AgentEvent};

use crate::agent::intermediate::IntermediateStep;

#[async_trait]
pub trait AgentExt: Agent {
    async fn plan_with_steps(
        &self,
        intermediate_steps: &[impl IntermediateStep],
        inputs: PromptArgs,
    ) -> Result<AgentEvent, AgentError>;

    async fn plan_stream(
        &self,
        steps: &[impl IntermediateStep],
        inputs: PromptArgs,
    ) -> Result<AgentStream, AgentError>;
}

pub type AgentStream = Pin<Box<dyn Stream<Item = Result<AgentEventChunk, ChainError>> + Send>>;

pub enum AgentEventChunk {
    Delta(DeltaEvent),
    Final(AgentEvent),
}

pub enum DeltaEvent {
    Action(AgentAction),
    Content(String),
}
