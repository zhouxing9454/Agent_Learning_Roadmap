//go:build !server
// +build !server

// Package main 实现了一个 MCP 客户端，将 MCP 工具集成到 eino 框架中。
//
// 本程序演示了如何：
//   - 使用 StreamableHTTP 传输连接到 MCP 服务器
//   - 发现并使用 MCP 服务器提供的工具
//   - 将 MCP 工具适配为 eino 的 BaseTool 接口
//   - 创建一个可以使用 MCP 工具的 ReAct Agent
//
// 模型上下文协议（MCP）是一个开放标准，用于实现 LLM 与外部系统、
// 数据源和工具之间的标准化通信。它采用客户端-服务器架构：
//   - MCP 服务器暴露工具、资源和提示
//   - MCP 客户端（如本程序）发现并使用这些能力
//
// 运行方式：go run main.go
// 确保 MCP 服务器在配置的地址上运行（默认：http://localhost:8080/mcp）
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/cloudwego/eino-ext/components/model/openai"
	"github.com/cloudwego/eino/components/tool"
	"github.com/cloudwego/eino/compose"
	"github.com/cloudwego/eino/flow/agent/react"
	"github.com/cloudwego/eino/schema"
	"github.com/mark3labs/mcp-go/client"
	"github.com/mark3labs/mcp-go/client/transport"
	"github.com/mark3labs/mcp-go/mcp"
)

// float32Ptr 返回给定 float32 值的指针。
// 这个辅助函数常用于配置需要指针类型作为可选参数的模型（例如 Temperature）。
func float32Ptr(f float32) *float32 {
	return &f
}

// MCPToolAdapter 将 MCP 工具适配为 eino 的 BaseTool 接口。
// 这使得 MCP 工具可以与 eino Agent 无缝使用。
type MCPToolAdapter struct {
	mcpClient *client.Client // 用于调用工具的 MCP 客户端
	tool      mcp.Tool       // MCP 工具定义
}

// NewMCPToolAdapter 为给定的 MCP 工具创建一个新的适配器。
// 适配器包装 MCP 工具，使其与 eino 的工具系统兼容。
func NewMCPToolAdapter(mcpClient *client.Client, tool mcp.Tool) *MCPToolAdapter {
	return &MCPToolAdapter{
		mcpClient: mcpClient,
		tool:      tool,
	}
}

// Info 实现 tool.BaseTool 接口。
// 它将 MCP 工具的输入模式转换为 eino 的 ToolInfo 格式。
func (m *MCPToolAdapter) Info(ctx context.Context) (*schema.ToolInfo, error) {
	params := make(map[string]*schema.ParameterInfo)

	// 解析 MCP 工具的输入模式属性
	// InputSchema.Properties 的类型是 map[string]any，每个值是一个 JSON Schema 对象
	// 注意：在 Go 中对 nil map 进行 range 是安全的，所以不需要检查 nil
	for paramName, paramValue := range m.tool.InputSchema.Properties {
		// 将 paramValue 转换为 map[string]any，这是 JSON Schema 对象的格式
		paramMap, ok := paramValue.(map[string]any)
		if !ok {
			// 如果转换失败，可能是其他格式（如使用 WithInputSchema[StructType]），跳过
			continue
		}

		paramInfo := &schema.ParameterInfo{
			Desc: getStringFromMap(paramMap, "description"),
		}

		// 将 MCP 参数类型映射到 eino 类型
		// JSON Schema 中的 type 字段可能是字符串或数组（对于联合类型）
		paramType := getParameterType(paramMap)
		switch paramType {
		case "string":
			paramInfo.Type = schema.String
		case "number", "integer":
			// number 和 integer 都映射到 eino 的 Number 类型
			paramInfo.Type = schema.Number
		case "boolean":
			// 注意：eino 没有 Bool 类型，所以使用 String
			paramInfo.Type = schema.String
		case "array", "object":
			// 复杂类型也映射为 String，因为 eino 主要支持简单类型
			paramInfo.Type = schema.String
		default:
			// 未知类型或空类型，默认为 String
			paramInfo.Type = schema.String
		}

		// 检查此参数是否为必需参数
		// Required 是一个字符串切片，包含所有必需参数的名称
		paramInfo.Required = isRequired(m.tool.InputSchema.Required, paramName)

		params[paramName] = paramInfo
	}

	return &schema.ToolInfo{
		Name:        m.tool.Name,
		Desc:        m.tool.Description,
		ParamsOneOf: schema.NewParamsOneOfByParams(params),
	}, nil
}

// getParameterType 从 JSON Schema 对象中提取参数类型。
// 支持 type 字段为字符串或数组（联合类型）的情况。
func getParameterType(paramMap map[string]any) string {
	typeVal, exists := paramMap["type"]
	if !exists {
		return ""
	}

	// 如果 type 是字符串，直接返回
	if typeStr, ok := typeVal.(string); ok {
		return typeStr
	}

	// 如果 type 是数组（联合类型），返回第一个类型
	if typeArr, ok := typeVal.([]any); ok && len(typeArr) > 0 {
		if firstType, ok := typeArr[0].(string); ok {
			return firstType
		}
	}

	return ""
}

// isRequired 检查参数名是否在必需参数列表中。
func isRequired(requiredList []string, paramName string) bool {
	for _, req := range requiredList {
		if req == paramName {
			return true
		}
	}
	return false
}

// getStringFromMap 安全地从 map[string]any 中提取字符串值。
// 这是 getString 的类型安全版本，专门用于 map[string]any。
func getStringFromMap(m map[string]any, key string) string {
	val, ok := m[key]
	if !ok {
		return ""
	}
	str, ok := val.(string)
	if !ok {
		return ""
	}
	return str
}

// extractTextFromContent 从 MCP 工具结果的内容数组中提取文本内容。
// 它会遍历所有内容项，找到第一个文本类型的内容并返回。
// 如果找到多个文本内容，会将它们合并（用换行符分隔）。
func extractTextFromContent(contents []mcp.Content) string {
	var textParts []string

	for _, content := range contents {
		// 尝试将内容转换为文本类型
		if textContent, ok := mcp.AsTextContent(content); ok {
			if textContent.Text != "" {
				textParts = append(textParts, textContent.Text)
			}
		}
		// 注意：这里只处理文本内容，其他类型（图片、音频等）会被忽略
		// 如果需要支持其他类型，可以在这里添加相应的处理逻辑
	}

	// 如果有多个文本内容，用换行符连接
	if len(textParts) > 0 {
		return strings.Join(textParts, "\n")
	}

	return ""
}

// InvokableRun 实现 tool.BaseTool 接口。
// 它使用提供的参数执行 MCP 工具并返回结果。
func (m *MCPToolAdapter) InvokableRun(ctx context.Context, argumentsInJSON string, opts ...tool.Option) (string, error) {
	// 解析 JSON 参数
	var args map[string]interface{}
	if err := json.Unmarshal([]byte(argumentsInJSON), &args); err != nil {
		return "", fmt.Errorf("无效的参数: %w", err)
	}

	fmt.Printf("\n--- 🛠️ MCP 工具调用：%s，参数：%s ---\n", m.tool.Name, argumentsInJSON)

	// 通过客户端调用 MCP 工具
	result, err := m.mcpClient.CallTool(ctx, mcp.CallToolRequest{
		Params: mcp.CallToolParams{
			Name:      m.tool.Name,
			Arguments: args,
		},
	})
	if err != nil {
		return "", fmt.Errorf("MCP 工具调用失败: %w", err)
	}

	// 从结果中提取文本内容
	// MCP 工具的结果可能包含多种类型的内容（文本、图片、音频、资源链接等）
	// CallToolResult.Content 是一个 Content 数组，每个元素可能是：
	//   - TextContent: 文本内容
	//   - ImageContent: 图片内容（base64 编码）
	//   - AudioContent: 音频内容
	//   - EmbeddedResource: 嵌入的资源
	// 我们需要找到文本类型的内容并提取出来
	textResult := extractTextFromContent(result.Content)
	if textResult != "" {
		fmt.Printf("--- ✅ MCP 工具结果：%s ---\n", textResult)
		return textResult, nil
	}

	return "", fmt.Errorf("MCP 工具返回了空结果或没有文本内容")
}

func main() {
	ctx := context.Background()

	// ============================================================================
	// 步骤 1: 从环境变量加载配置
	// ============================================================================
	openaiAPIKey := os.Getenv("OPENAI_API_KEY")
	if openaiAPIKey == "" {
		fmt.Println("错误: 未设置 OPENAI_API_KEY 环境变量")
		os.Exit(1)
	}

	openaiBaseURL := os.Getenv("OPENAI_BASE_URL") // 可选：用于自定义 API 端点

	// MCP 服务器地址（如果未设置，默认为 localhost:8080/mcp）
	mcpServerURL := os.Getenv("MCP_SERVER_URL")
	if mcpServerURL == "" {
		mcpServerURL = "http://localhost:8080/mcp"
	}

	// ============================================================================
	// 步骤 2: 初始化 LLM 模型
	// ============================================================================
	llmConfig := &openai.ChatModelConfig{
		Model:       "Qwen/Qwen2.5-72B-Instruct",
		APIKey:      openaiAPIKey,
		Temperature: float32Ptr(0.7), // 较高的温度值以获得更有创造性的响应
	}

	if openaiBaseURL != "" {
		llmConfig.BaseURL = openaiBaseURL
	}

	llm, err := openai.NewChatModel(ctx, llmConfig)
	if err != nil {
		fmt.Printf("初始化语言模型失败: %v\n", err)
		os.Exit(1)
	}
	fmt.Println("✅ LLM 模型已初始化")

	// ============================================================================
	// 步骤 3: 连接到 MCP 服务器
	// ============================================================================
	fmt.Printf("🔌 正在连接到 MCP 服务器: %s\n", mcpServerURL)

	// 创建用于 MCP 通信的 StreamableHTTP 传输
	httpTransport, err := transport.NewStreamableHTTP(mcpServerURL)
	if err != nil {
		fmt.Printf("创建 MCP 传输失败: %v\n", err)
		os.Exit(1)
	}

	// 使用传输创建 MCP 客户端
	mcpClient := client.NewClient(httpTransport)

	// 启动客户端连接
	if err := mcpClient.Start(ctx); err != nil {
		fmt.Printf("启动 MCP 客户端失败: %v\n", err)
		fmt.Println("\n提示: 请确保 MCP 服务器正在运行")
		fmt.Println("运行命令: cd mcp-server && go run -tags server mcp_server.go")
		os.Exit(1)
	}
	defer mcpClient.Close() // 确保退出时清理资源

	// ============================================================================
	// 步骤 4: 初始化 MCP 协议
	// ============================================================================
	// 创建 MCP 初始化请求对象
	// InitializeRequest 是客户端向服务器发送的第一个请求，用于协商协议版本和能力
	initReq := mcp.InitializeRequest{
		// Params 字段包含初始化请求的所有参数
		Params: mcp.InitializeParams{
			// ProtocolVersion: 指定客户端支持的 MCP 协议版本
			// LATEST_PROTOCOL_VERSION 是库中定义的最新协议版本常量（如 "2024-11-05"）
			// 服务器会根据这个版本决定使用哪个协议版本进行通信
			ProtocolVersion: mcp.LATEST_PROTOCOL_VERSION,

			// ClientInfo: 客户端信息，用于标识客户端身份
			// 服务器可以基于此信息进行日志记录、统计或提供不同的服务
			ClientInfo: mcp.Implementation{
				Name:    "Eino MCP Client", // 客户端名称，用于标识这个客户端
				Version: "1.0.0",           // 客户端版本号，用于版本兼容性检查
			},

			// Capabilities: 客户端能力声明
			// 告诉服务器客户端支持哪些高级功能
			// 注意：工具、资源、提示等基础功能是客户端默认支持的，不需要在这里声明
			// 这里声明的是可选的高级能力：
			//   Sampling: &struct{}{},     // 支持从 LLM 采样（服务器可以向客户端请求 LLM 生成）
			//   Elicitation: &struct{}{},  // 支持服务器发起的请求（服务器可以主动请求客户端执行操作）
			//   Roots: &struct{ListChanged: true}, // 支持根资源列表变更通知
			//   Experimental: map[string]any{...}, // 实验性功能
			// 空结构体表示只使用基础能力，不启用任何高级功能
			Capabilities: mcp.ClientCapabilities{},
		},
	}

	initResult, err := mcpClient.Initialize(ctx, initReq)
	if err != nil {
		fmt.Printf("初始化 MCP 协议失败: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("✅ 已连接到 MCP 服务器: %s v%s\n", initResult.ServerInfo.Name, initResult.ServerInfo.Version)

	// ============================================================================
	// 步骤 5: 发现可用的 MCP 工具
	// ============================================================================
	toolsResp, err := mcpClient.ListTools(ctx, mcp.ListToolsRequest{})
	if err != nil {
		fmt.Printf("获取 MCP 工具列表失败: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("📋 可用 MCP 工具 (%d 个):\n", len(toolsResp.Tools))
	for _, t := range toolsResp.Tools {
		fmt.Printf("  - %s: %s\n", t.Name, t.Description)
	}

	// ============================================================================
	// 步骤 6: 将 MCP 工具适配为 eino 的 BaseTool 接口
	// ============================================================================
	var einoTools []tool.BaseTool
	for _, mcpTool := range toolsResp.Tools {
		adapter := NewMCPToolAdapter(mcpClient, mcpTool)
		einoTools = append(einoTools, adapter)
	}

	// ============================================================================
	// 步骤 7: 使用适配后的工具创建 ReAct Agent
	// ============================================================================
	agentConfig := &react.AgentConfig{
		ToolCallingModel: llm, // 决定何时调用工具的 LLM
		ToolsConfig: compose.ToolsNodeConfig{
			Tools: einoTools, // Agent 可用的工具
		},
		MaxStep: 10, // 停止前的最大推理步数
	}

	agent, err := react.NewAgent(ctx, agentConfig)
	if err != nil {
		fmt.Printf("创建 Agent 失败: %v\n", err)
		os.Exit(1)
	}

	// ============================================================================
	// 步骤 8: 运行演示查询
	// ============================================================================
	fmt.Println("\n" + strings.Repeat("=", 70))
	fmt.Println("## MCP Agent 演示：使用 MCP 工具 ##")
	fmt.Println(strings.Repeat("=", 70))

	queries := []string{
		"请向张三打招呼",           // 测试 greet 工具
		"计算 15 + 27 等于多少？",  // 测试 calculate 工具（加法）
		"计算 100 - 45 等于多少？", // 测试 calculate 工具（减法）
		"计算 8 * 9 等于多少？",    // 测试 calculate 工具（乘法）
		"计算 144 / 12 等于多少？", // 测试 calculate 工具（除法）
		"现在几点了？",            // 测试 get_current_time 工具
	}

	for i, query := range queries {
		fmt.Printf("\n--- [轮次 %d] 用户输入: %s ---\n", i+1, query)

		messages := []*schema.Message{
			schema.UserMessage(query),
		}

		// 使用 Agent 生成响应
		// Agent 会根据查询自动决定使用哪些工具
		response, err := agent.Generate(ctx, messages)
		if err != nil {
			fmt.Printf("🛑 Agent 执行期间发生错误：%v\n", err)
			continue
		}

		fmt.Println("\n--- ✅ Agent 响应 ---")
		fmt.Println(response.Content)
		fmt.Println(strings.Repeat("-", 60))

		// 添加短暂延迟，避免请求过快
		time.Sleep(1 * time.Second)
	}

	// ============================================================================
	// 总结
	// ============================================================================
	fmt.Println("\n" + strings.Repeat("=", 70))
	fmt.Println("## 演示完成 ##")
	fmt.Println(strings.Repeat("=", 70))
	fmt.Println("\n关键要点：")
	fmt.Println("1. MCP 提供了标准化的协议，让 Agent 能够使用外部工具")
	fmt.Println("2. MCP 服务器提供工具、资源和提示，客户端（Agent）使用这些能力")
	fmt.Println("3. 通过适配器模式，可以将 MCP 工具集成到 eino 框架中")
	fmt.Println("4. 这种架构使得工具和 Agent 可以独立开发和部署")
}
