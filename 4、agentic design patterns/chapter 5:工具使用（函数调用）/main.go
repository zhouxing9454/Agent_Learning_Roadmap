/*
	工具使用（Tool Calling/Function Calling）是 Agent 系统的"能力扩展器"，它让 AI 能够调用外部函数和 API，从而突破纯文本生成的限制，实现与真实世界的交互，获取实时数据、执行操作、访问外部服务。

	工具类型	   适用场景		      核心优势		           核心劣势					     你的 Go Agent 该选谁？
	API 工具	   外部服务调用	 实时数据获取		         依赖外部服务稳定性			     天气查询、股票价格、翻译服务
	计算工具	   数学运算		 精确计算		             功能有限			     计算器、统计分析
	搜索工具	   信息检索		 获取最新信息		         可能返回不相关信息			     网络搜索、文档检索
	系统工具	   系统操作		 直接操作系统		         安全风险高			     文件操作、命令执行（需谨慎）

	此代码根据 MIT 许可证授权。
	请参阅仓库中的 LICENSE 文件以获取完整许可文本。
*/

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/cloudwego/eino-ext/components/model/openai"
	"github.com/cloudwego/eino/components/tool"
	"github.com/cloudwego/eino/compose"
	"github.com/cloudwego/eino/flow/agent/react"
	"github.com/cloudwego/eino/schema"
)

// float32Ptr: 辅助函数，将 float32 值转换为 *float32 指针
func float32Ptr(f float32) *float32 {
	return &f
}

// --- 定义工具 ---

// CalculatorTool: 自定义计算器工具
type CalculatorTool struct{}

func (c *CalculatorTool) Info(ctx context.Context) (*schema.ToolInfo, error) {
	return &schema.ToolInfo{
		Name: "calculator",
		Desc: "执行基本算术运算（加、减、乘、除）",
		ParamsOneOf: schema.NewParamsOneOfByParams(map[string]*schema.ParameterInfo{
			"operation": {
				Type:     schema.String,
				Desc:     "要执行的操作：add（加）、subtract（减）、multiply（乘）、divide（除）",
				Required: true,
			},
			"a": {
				Type:     schema.Number,
				Desc:     "第一个数字",
				Required: true,
			},
			"b": {
				Type:     schema.Number,
				Desc:     "第二个数字",
				Required: true,
			},
		}),
	}, nil
}

func (c *CalculatorTool) InvokableRun(ctx context.Context, argumentsInJSON string, opts ...tool.Option) (string, error) {
	var args struct {
		Operation string  `json:"operation"`
		A         float64 `json:"a"`
		B         float64 `json:"b"`
	}

	if err := json.Unmarshal([]byte(argumentsInJSON), &args); err != nil {
		return "", fmt.Errorf("无效的参数: %w", err)
	}

	fmt.Printf("\n--- 🛠️ 工具调用：calculator，操作：'%s'，参数：a=%.2f, b=%.2f ---\n", args.Operation, args.A, args.B)

	var result float64
	switch strings.ToLower(args.Operation) {
	case "add":
		result = args.A + args.B
	case "subtract":
		result = args.A - args.B
	case "multiply":
		result = args.A * args.B
	case "divide":
		if args.B == 0 {
			return "", fmt.Errorf("除以零错误")
		}
		result = args.A / args.B
	default:
		return "", fmt.Errorf("未知操作: %s", args.Operation)
	}

	resultStr := fmt.Sprintf("%.2f", result)
	fmt.Printf("--- 工具结果：%s ---\n", resultStr)
	return resultStr, nil
}

func main() {
	ctx := context.Background()

	// --- 配置 ---
	// 从环境变量读取 API 密钥
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		fmt.Println("错误: 请设置 OPENAI_API_KEY 环境变量")
		os.Exit(1)
	}

	// 从环境变量读取自定义 BaseURL（可选）
	baseURL := os.Getenv("OPENAI_BASE_URL")

	// 创建 OpenAI ChatModel 配置
	config := &openai.ChatModelConfig{
		Model:       "deepseek-ai/DeepSeek-V3.1", // 需要支持工具调用的模型
		APIKey:      apiKey,
		Temperature: float32Ptr(0), // 使用较低的温度以获得更确定性的输出
	}

	// 如果设置了自定义 BaseURL，则使用它（支持代理或兼容 API）
	if baseURL != "" {
		config.BaseURL = baseURL
	}

	// 初始化 LLM
	llm, err := openai.NewChatModel(ctx, config)
	if err != nil {
		fmt.Printf("初始化语言模型时出错: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("✅ 语言模型已初始化: %s\n", config.Model)

	// --- 创建工具 ---
	calculator := &CalculatorTool{}

	tools := []tool.BaseTool{calculator}

	// --- 创建 ReAct Agent ---
	agentConfig := &react.AgentConfig{
		ToolCallingModel: llm,
		ToolsConfig: compose.ToolsNodeConfig{
			Tools: tools,
		},
		MaxStep: 10,
	}

	agent, err := react.NewAgent(ctx, agentConfig)
	if err != nil {
		fmt.Printf("创建 Agent 失败: %v\n", err)
		os.Exit(1)
	}

	// --- 运行 Agent 查询 ---
	queries := []string{
		"5+6等于多少？",
		"5-6等于多少？",
		"5*6等于多少？",
		"5/6等于多少？",
	}

	for _, query := range queries {
		fmt.Printf("\n--- 🏃 使用查询运行 Agent：'%s' ---\n", query)

		messages := []*schema.Message{
			schema.UserMessage(query),
		}

		response, err := agent.Generate(ctx, messages)
		if err != nil {
			fmt.Printf("🛑 Agent 执行期间发生错误：%v\n", err)
			continue
		}

		fmt.Println("\n--- ✅ 最终 Agent 响应 ---")
		fmt.Println(response.Content)
		fmt.Println(strings.Repeat("-", 60))
	}
}
