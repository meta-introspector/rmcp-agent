use clap::Parser;
use rmcp::handler::server::tool::{Parameters, ToolRouter};
use rmcp::model::{ServerCapabilities, ServerInfo};
use rmcp::transport::SseServer;
use rmcp::transport::sse_server::SseServerConfig;
use rmcp::{ServerHandler, schemars, tool, tool_handler, tool_router};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

#[derive(Parser, Debug)]
#[command(name = "rmcp-demo-server")]
#[command(about = "A demo MCP server with basic tools")]
struct Args {
    /// Port to bind the server to
    #[arg(short, long, default_value = "8000")]
    port: u16,

    /// Host address to bind to
    #[arg(short = 'H', long, default_value = "127.0.0.1")]
    host: String,
}

#[tokio::main]
async fn main() {
    let args = Args::parse();
    let bind_address = format!("{}:{}", args.host, args.port);

    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "debug".to_string().into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let config = SseServerConfig {
        bind: bind_address.parse().unwrap(),
        sse_path: "/sse".to_string(),
        post_path: "/message".to_string(),
        ct: tokio_util::sync::CancellationToken::new(),
        sse_keep_alive: None,
    };

    let (sse_server, router) = SseServer::new(config);

    let listener = tokio::net::TcpListener::bind(sse_server.config.bind)
        .await
        .unwrap();

    let ct = sse_server.config.ct.child_token();

    tracing::info!("🚀 Starting MCP demo server on {}", bind_address);
    tracing::info!("📡 SSE endpoint: http://{}/sse", bind_address);
    tracing::info!("📬 Message endpoint: http://{}/message", bind_address);

    let server = axum::serve(listener, router).with_graceful_shutdown(async move {
        ct.cancelled().await;
    });

    tokio::spawn(async move {
        if let Err(e) = server.await {
            tracing::error!(error = %e, "sse server shutdown with error");
        }
    });

    let ct = sse_server.with_service(McpDemoService::new);

    tokio::signal::ctrl_c().await.unwrap();
    ct.cancel();
}

#[derive(Debug)]
pub struct McpDemoService {
    tool_router: ToolRouter<Self>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
struct SumRequest {
    #[schemars(description = "First integer to add (left operand)")]
    a: i32,
    #[schemars(description = "Second integer to add (right operand)")]
    b: i32,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
struct SubRequest {
    #[schemars(description = "The minuend (number to subtract from) - must be integer")]
    a: i32,
    #[schemars(description = "The subtrahend (number to subtract) - must be integer")]
    b: i32,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
struct FactorialRequest {
    #[schemars(
        description = "Positive integer to calculate factorial for (1-20). MUST be integer type, not float or string"
    )]
    n: i32,
}

#[tool_router]
impl McpDemoService {
    pub fn new() -> Self {
        Self {
            tool_router: Self::tool_router(),
        }
    }

    #[tool(
        description = "Adds two integers and returns their sum. Use this for mathematical addition operations. Always pass integer values, not floats. Example: to calculate 3+5, call sum with a=3, b=5"
    )]
    fn sum(&self, Parameters(SumRequest { a, b }): Parameters<SumRequest>) -> String {
        (a + b).to_string()
    }

    #[tool(
        description = "Subtracts second integer from first integer (a-b) and returns the difference. Use this for mathematical subtraction operations. Always pass integer values, not floats. Example: to calculate 8-1, call sub with a=8, b=1"
    )]
    fn sub(&self, Parameters(SubRequest { a, b }): Parameters<SubRequest>) -> String {
        (a - b).to_string()
    }

    #[tool(
        description = "Calculates factorial of a positive integer (n!). CRITICAL: The parameter 'n' MUST be passed as an integer value (like 7), NOT as a float (like 7.0) or string. Valid range: 1-20. Use this after getting integer results from other calculations. Example: factorial with n=7 calculates 7! = 5040"
    )]
    fn factorial(
        &self,
        Parameters(FactorialRequest { n }): Parameters<FactorialRequest>,
    ) -> String {
        tracing::info!("Calculating factorial of: {}", n);
        let mut result = 1u64;
        for i in 1..=n {
            result *= i as u64;
        }
        result.to_string()
    }
}

#[tool_handler]
impl ServerHandler for McpDemoService {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            instructions: Some("A simple calculator".into()),
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            ..Default::default()
        }
    }
}
