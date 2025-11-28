package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/cloudwego/eino-ext/components/model/openai"
	"github.com/cloudwego/eino/components/prompt"
	"github.com/cloudwego/eino/compose"
	"github.com/cloudwego/eino/schema"
	"github.com/joho/godotenv"
)

// AgentState 定义了在 Graph 节点之间流转的全局状态
// 对应 Python 示例中的 state 字典
type AgentState struct {
	UserQuery             string
	LocationResult        string // 存储工具调用返回的 JSON 或字符串结果
	PrimaryLocationFailed bool   // 状态标志：主要工具是否失败
	FinalResponse         string // 最终生成的回复
}

// 辅助函数：将 float32 转为指针
func float32Ptr(f float32) *float32 {
	return &f
}

func main() {
	// 1. --- 环境设置 ---
	_ = godotenv.Load()
	ctx := context.Background()

	// 2. --- 初始化共享的 LLM 模型 ---
	fmt.Println("📡 初始化 OpenAI LLM (gpt-4o)...")
	chatModel, err := openai.NewChatModel(ctx, &openai.ChatModelConfig{
		Model:       "gpt-4o",
		Temperature: float32Ptr(0.1), // 降低温度以获得更确定的工具参数提取
		APIKey:      os.Getenv("OPENAI_API_KEY"),
		BaseURL:     os.Getenv("OPENAI_BASE_URL"),
	})
	if err != nil {
		log.Fatalf("无法初始化模型: %v", err)
	}

	// ========================================================================
	// 🛠️ 模拟工具函数 (Mock Tools)
	// 在实际应用中，这些可能是外部 API 调用或 Eino Tool 组件
	// ========================================================================

	// Tool 1: get_precise_location_info
	// 模拟逻辑：只有包含 "Street" 或 "路" 的地址才能找到精确位置
	getPreciseLocationInfo := func(address string) (string, bool) {
		if strings.Contains(address, "Street") || strings.Contains(address, "路") {
			return fmt.Sprintf(`{"lat": 37.7749, "lng": -122.4194, "address": "%s", "precision": "high"}`, address), true
		}
		return "", false
	}

	// Tool 2: get_general_area_info
	// 模拟逻辑：返回城市的通用信息
	getGeneralAreaInfo := func(city string) string {
		return fmt.Sprintf(`{"city": "%s", "region_code": "US-CA", "description": "General metropolitan area of %s"}`, city, city)
	}

	// ========================================================================
	// 🏗️ Agent 1: Primary Handler (主要处理器)
	// 职责：尝试获取精确位置。如果无法提取精确地址或工具调用失败，标记失败。
	// ========================================================================
	primarySysPrompt := `你的工作是获取精确的位置信息。
请从用户查询中提取具体的街道地址。
如果查询中包含具体的街道地址，请输出该地址。
如果查询模糊、只包含城市名或无法提取地址，请输出 "FAIL"。`

	primaryTemplate := prompt.FromMessages(
		schema.FString,
		schema.SystemMessage(primarySysPrompt),
		schema.UserMessage("用户查询：{{.UserQuery}}"),
	)

	primaryChain, _ := compose.NewChain[map[string]any, *schema.Message]().
		AppendChatTemplate(primaryTemplate).
		AppendChatModel(chatModel).
		Compile(ctx)

	// 将 Chain 包装为 Node，包含工具调用逻辑
	primaryNode := compose.InvokableLambda(func(ctx context.Context, state AgentState) (AgentState, error) {
		fmt.Println("📍 [Agent 1] Primary Handler: 尝试获取精确位置...")

		resp, err := primaryChain.Invoke(ctx, map[string]any{"UserQuery": state.UserQuery})
		if err != nil {
			return state, err
		}

		address := strings.TrimSpace(resp.Content)
		if address == "FAIL" {
			fmt.Println("   -> ⚠️  无法提取精确地址，标记失败。")
			state.PrimaryLocationFailed = true
		} else {
			// 模拟工具调用
			info, success := getPreciseLocationInfo(address)
			if success {
				state.LocationResult = info
				state.PrimaryLocationFailed = false
				fmt.Printf("   -> ✅ 成功获取精确位置: %s\n", info)
			} else {
				fmt.Println("   -> ⚠️  工具调用失败 (地址无效)，标记失败。")
				state.PrimaryLocationFailed = true
			}
		}
		return state, nil
	})

	// ========================================================================
	// 🏗️ Agent 2: Fallback Handler (回退处理器)
	// 职责：检查 state["primary_location_failed"]。如果为 True，使用通用信息工具。
	// ========================================================================
	fallbackSysPrompt := `你是一个回退处理器。
从用户的原始查询中提取城市名称。仅输出城市名称。`

	fallbackTemplate := prompt.FromMessages(
		schema.FString,
		schema.SystemMessage(fallbackSysPrompt),
		schema.UserMessage("用户查询：{{.UserQuery}}"),
	)

	fallbackChain, _ := compose.NewChain[map[string]any, *schema.Message]().
		AppendChatTemplate(fallbackTemplate).
		AppendChatModel(chatModel).
		Compile(ctx)

	fallbackNode := compose.InvokableLambda(func(ctx context.Context, state AgentState) (AgentState, error) {
		// 逻辑：如果主要位置查找未失败（即成功），则什么也不做
		if !state.PrimaryLocationFailed {
			fmt.Println("🛡️ [Agent 2] Fallback Handler: Primary 成功，跳过回退。")
			return state, nil
		}

		fmt.Println("🛡️ [Agent 2] Fallback Handler: 检测到 Primary 失败，执行回退逻辑...")
		resp, err := fallbackChain.Invoke(ctx, map[string]any{"UserQuery": state.UserQuery})
		if err != nil {
			return state, err
		}

		city := strings.TrimSpace(resp.Content)
		// 模拟工具调用
		info := getGeneralAreaInfo(city)
		state.LocationResult = info
		fmt.Printf("   -> ℹ️  获取通用区域信息: %s\n", info)

		return state, nil
	})

	// ========================================================================
	// 🏗️ Agent 3: Response Agent (响应生成器)
	// 职责：查看 state["location_result"] 并向用户呈现信息。
	// ========================================================================
	responseSysPrompt := `查看提供的位置结果信息。
向用户清晰简洁地呈现此信息。
如果位置结果不存在或为空，请道歉您无法检索位置。`

	responseTemplate := prompt.FromMessages(
		schema.FString,
		schema.SystemMessage(responseSysPrompt),
		schema.UserMessage(`
用户查询：{{.UserQuery}}
位置结果：{{.LocationResult}}

请生成回复：`),
	)

	responseChain, _ := compose.NewChain[map[string]any, *schema.Message]().
		AppendChatTemplate(responseTemplate).
		AppendChatModel(chatModel).
		Compile(ctx)

	responseNode := compose.InvokableLambda(func(ctx context.Context, state AgentState) (AgentState, error) {
		fmt.Println("💬 [Agent 3] Response Agent: 生成最终回复...")

		input := map[string]any{
			"UserQuery":      state.UserQuery,
			"LocationResult": state.LocationResult,
		}

		resp, err := responseChain.Invoke(ctx, input)
		if err != nil {
			return state, err
		}

		state.FinalResponse = resp.Content
		return state, nil
	})

	// ========================================================================
	// 🕸️ 构建 Sequential Graph (顺序执行)
	// 对应 Python 中的 SequentialAgent(sub_agents=[primary, fallback, response])
	// ========================================================================
	graph := compose.NewGraph[AgentState, AgentState]()

	_ = graph.AddLambdaNode("PrimaryHandler", primaryNode)
	_ = graph.AddLambdaNode("FallbackHandler", fallbackNode)
	_ = graph.AddLambdaNode("ResponseAgent", responseNode)

	// 定义线性执行路径: START -> Primary -> Fallback -> Response -> END
	_ = graph.AddEdge(compose.START, "PrimaryHandler")
	_ = graph.AddEdge("PrimaryHandler", "FallbackHandler")
	_ = graph.AddEdge("FallbackHandler", "ResponseAgent")
	_ = graph.AddEdge("ResponseAgent", compose.END)

	runnable, err := graph.Compile(ctx)
	if err != nil {
		log.Fatalf("编译 Graph 失败: %v", err)
	}

	// ========================================================================
	// 🚀 运行测试场景
	// ========================================================================

	// 场景 A: 模糊查询 (预期触发 Primary 失败 -> Fallback 成功)
	fmt.Println("\n>>> 场景 A: 模糊查询 (触发 Fallback)")
	stateA := AgentState{UserQuery: "我想找一家在 San Francisco 的咖啡馆"}
	resA, _ := runnable.Invoke(ctx, stateA)
	fmt.Printf("🤖 最终输出:\n%s\n", resA.FinalResponse)
	fmt.Println(strings.Repeat("-", 50))

	// 场景 B: 精确查询 (预期 Primary 成功 -> Fallback 跳过)
	fmt.Println("\n>>> 场景 B: 精确查询 (Primary 成功)")
	stateB := AgentState{UserQuery: "定位到 123 Market Street, San Francisco"}
	resB, _ := runnable.Invoke(ctx, stateB)
	fmt.Printf("🤖 最终输出:\n%s\n", resB.FinalResponse)
}
