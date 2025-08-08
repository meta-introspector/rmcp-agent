use std::sync::Arc;

use async_trait::async_trait;
use langchain_rust::tools::Tool;
use rmcp::RoleClient;
use rmcp::model::{CallToolRequestParam, InitializeRequestParam, object};
use rmcp::service::RunningService;
use serde_json::{Map, Value};

pub struct RmcpTool {
    tool: rmcp::model::Tool,
    client: Arc<RunningService<RoleClient, InitializeRequestParam>>,
}

impl RmcpTool {
    pub(crate) fn new(
        tool: rmcp::model::Tool,
        client: Arc<RunningService<RoleClient, InitializeRequestParam>>,
    ) -> Self {
        Self { tool, client }
    }
}

#[async_trait]
impl Tool for RmcpTool {
    fn name(&self) -> String {
        self.tool.name.to_string()
    }

    fn description(&self) -> String {
        self.tool
            .description
            .clone()
            .unwrap_or_default()
            .to_string()
    }

    fn parameters(&self) -> Value {
        self.tool.schema_as_json_value()
    }

    async fn run(&self, input: Value) -> Result<String, Box<dyn std::error::Error>> {
        let response = self
            .client
            .call_tool(CallToolRequestParam {
                name: self.tool.name.clone(),
                arguments: Some(object(input)),
            })
            .await?;

        let mut resp = String::default();
        let raw_content = response.content.unwrap_or_default();
        for content in raw_content {
            let t = content.as_text();
            if let Some(text) = t {
                resp.push_str(&text.text);
            }
        }
        Ok(resp)
    }

    async fn parse_input(&self, input: &str) -> Value {
        match serde_json::from_str::<Map<String, Value>>(input) {
            Ok(parsed_input) => Value::Object(parsed_input),
            Err(_) => serde_json::json!({
                "value": input,
            }),
        }
    }
}
