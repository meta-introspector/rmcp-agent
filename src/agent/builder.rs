use std::sync::Arc;

use langchain_rust::agent::AgentError;
use langchain_rust::chain::LLMChainBuilder;
use langchain_rust::chain::options::ChainCallOptions;
use langchain_rust::language_models::llm::LLM;
use langchain_rust::language_models::options::CallOptions;
use langchain_rust::llm::{OpenAI, OpenAIConfig};
use langchain_rust::schemas::FunctionDefinition;
use langchain_rust::tools::Tool;
use rmcp::RoleClient;
use rmcp::model::InitializeRequestParam;
use rmcp::service::RunningService;

use crate::agent::core::OpenAIMcpAgent;
use crate::tool::rmcp::RmcpTool;

const PREFIX: &str = r#"
Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

Overall, Assistant is a powerful system that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.
"#;

pub struct OpenAIMcpAgentBuilder {
    tools: Option<Vec<Arc<dyn Tool>>>,
    prefix: Option<String>,
    options: Option<ChainCallOptions>,

    llm: OpenAI<OpenAIConfig>,
}

impl OpenAIMcpAgentBuilder {
    pub fn new(api_key: impl ToString, api_base: impl ToString, model: impl ToString) -> Self {
        let config = OpenAIConfig::default()
            .with_api_base(api_base.to_string())
            .with_api_key(api_key.to_string());

        let llm = OpenAI::default()
            .with_config(config)
            .with_model(model.to_string());

        OpenAIMcpAgentBuilder {
            tools: None,
            prefix: None,
            options: None,
            llm,
        }
    }

    pub fn mcp_tools(
        mut self,
        mcp_client: Arc<RunningService<RoleClient, InitializeRequestParam>>,
        tools: Vec<rmcp::model::Tool>,
    ) -> Self {
        let mut langchain_tools: Vec<Arc<dyn Tool>> = Vec::with_capacity(tools.len());
        for tool in tools {
            let t = RmcpTool::new(tool, mcp_client.clone());
            langchain_tools.push(Arc::new(t));
        }

        match self.tools {
            Some(ref mut tools) => tools.extend(langchain_tools),
            None => self.tools = Some(langchain_tools),
        }

        self
    }

    pub fn prefix<S: Into<String>>(mut self, prefix: S) -> Self {
        self.prefix = Some(prefix.into());
        self
    }

    pub fn options(mut self, options: ChainCallOptions) -> Self {
        self.options = Some(options);
        self
    }

    pub fn build(self) -> Result<OpenAIMcpAgent, AgentError> {
        let tools = self.tools.unwrap_or_default();
        let prefix = self.prefix.unwrap_or_else(|| PREFIX.to_string());
        let mut llm = self.llm;

        let prompt = OpenAIMcpAgent::create_prompt(&prefix);
        let default_options = ChainCallOptions::default().with_max_tokens(1000);
        let functions = tools
            .iter()
            .map(FunctionDefinition::from_langchain_tool)
            .collect::<Vec<FunctionDefinition>>();

        llm.add_options(CallOptions::new().with_functions(functions));

        let chain = Box::new(
            LLMChainBuilder::new()
                .prompt(prompt)
                .llm(llm)
                .options(self.options.unwrap_or(default_options))
                .build()?,
        );

        Ok(OpenAIMcpAgent { chain, tools })
    }
}
