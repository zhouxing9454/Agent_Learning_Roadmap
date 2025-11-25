/*
	路由（Routing）是 Agent 系统的“动态决策枢纽”，它负责精准识别输入意图与系统状态，实时将任务分发给最匹配的工具或链路，从而将死板的线性流程转化为灵活的自适应智能。

	路由类型	   形象比喻		      核心优势		           核心劣势					 你的 Go Agent 该选谁？
	规则路由	   电话按键菜单		 极快、免费		         听不懂人话，不灵活			     兜底用（比如用户输入 /start）
	LLM 路由      真人前台	       最聪明、懂暗语		    慢、费钱					  开发初期用（最容易实现）
	嵌入路由	   图书管理员	  	性价比之王、懂语义	     需要向量数据库支持				 生产环境推荐（平衡了速度和智能）
	ML 路由	      专用分拣机	   快、量大时成本最低	    训练麻烦、难以冷启动		    巨头公司用（通常不做这个）
*/

package main

import (
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/cloudwego/eino-ext/components/model/openai"
	"github.com/cloudwego/eino/components/prompt"
	"github.com/cloudwego/eino/compose"
	"github.com/cloudwego/eino/schema"
)

// float32Ptr: 辅助函数，将 float32 值转换为 *float32 指针
func float32Ptr(f float32) *float32 {
	return &f
}

// --- 定义模拟子 Agent 处理程序（相当于 ADK 的 sub_agents）---

// bookingHandler: 模拟预订 Agent 处理请求
func bookingHandler(ctx context.Context, request string) (string, error) {
	fmt.Println("\n--- 委托给预订处理程序 ---")
	return fmt.Sprintf("预订处理程序处理了请求：'%s'。结果：模拟预订操作。", request), nil
}

// infoHandler: 模拟信息 Agent 处理请求
func infoHandler(ctx context.Context, request string) (string, error) {
	fmt.Println("\n--- 委托给信息处理程序 ---")
	return fmt.Sprintf("信息处理程序处理了请求：'%s'。结果：模拟信息检索。", request), nil
}

// unclearHandler: 处理无法委托的请求
func unclearHandler(ctx context.Context, request string) (string, error) {
	fmt.Println("\n--- 处理不清楚的请求 ---")
	return fmt.Sprintf("协调器无法委托请求：'%s'。请澄清。", request), nil
}

func main() {
	ctx := context.Background()

	// --- 配置 ---
	// 确保设置了您的 API 密钥环境变量（例如，OPENAI_API_KEY）
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		fmt.Println("错误: 请设置 OPENAI_API_KEY 环境变量")
		os.Exit(1)
	}

	// 从环境变量读取自定义 BaseURL（可选）
	baseURL := os.Getenv("OPENAI_BASE_URL")

	// 创建 OpenAI ChatModel 配置
	config := &openai.ChatModelConfig{
		Model:       "qwen/Qwen2.5-Coder-32B-Instruct", // 或您想要使用的模型
		APIKey:      apiKey,
		Temperature: float32Ptr(0), // 设置为 0 以获得确定性输出
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

	fmt.Printf("语言模型已初始化: %s\n", config.Model)

	// --- 定义协调器路由链（相当于 ADK 协调器的指令）---
	// 此链决定应委托给哪个处理程序。
	coordinatorRouterPrompt := prompt.FromMessages(
		schema.FString,
		schema.SystemMessage(`分析用户的请求并确定哪个专家处理程序应处理它。
     - 如果请求与预订航班或酒店相关，输出 'booker'。
     - 对于所有其他一般信息问题，输出 'info'。
     - 如果请求不清楚或不适合任一类别，输出 'unclear'。
     只输出一个词：'booker'、'info' 或 'unclear'。`),
		schema.UserMessage("{request}"),
	)

	// Lambda 函数：从 Message 中提取 Content 作为决策字符串
	extractDecision := compose.InvokableLambda(func(ctx context.Context, msg *schema.Message) (string, error) {
		decision := strings.TrimSpace(msg.Content)
		return decision, nil
	})

	// 构建路由链：Template -> ChatModel -> Lambda (提取决策)
	routerChain, err := compose.NewChain[map[string]any, string]().
		AppendChatTemplate(coordinatorRouterPrompt). // map -> []*Message
		AppendChatModel(llm).                        // []*Message -> *Message
		AppendLambda(extractDecision).               // *Message -> string
		Compile(ctx)
	if err != nil {
		fmt.Printf("编译路由链失败: %v\n", err)
		os.Exit(1)
	}

	// --- 定义委托逻辑（相当于 ADK 的基于 sub_agents 的自动流）---
	// 使用 Graph 和 Branch 根据路由链的输出进行路由

	// 定义输入结构：包含原始请求和路由决策
	type RouterInput struct {
		Request  string
		Decision string
	}

	// 定义输出结构：包含处理结果
	type RouterOutput struct {
		Output string
	}

	// 创建 Graph
	graph := compose.NewGraph[RouterInput, RouterOutput]()

	// Lambda 节点：booking 处理程序
	bookingLambda := compose.InvokableLambda(func(ctx context.Context, input RouterInput) (RouterOutput, error) {
		result, err := bookingHandler(ctx, input.Request)
		if err != nil {
			return RouterOutput{}, err
		}
		return RouterOutput{Output: result}, nil
	})

	// Lambda 节点：info 处理程序
	infoLambda := compose.InvokableLambda(func(ctx context.Context, input RouterInput) (RouterOutput, error) {
		result, err := infoHandler(ctx, input.Request)
		if err != nil {
			return RouterOutput{}, err
		}
		return RouterOutput{Output: result}, nil
	})

	// Lambda 节点：unclear 处理程序
	unclearLambda := compose.InvokableLambda(func(ctx context.Context, input RouterInput) (RouterOutput, error) {
		result, err := unclearHandler(ctx, input.Request)
		if err != nil {
			return RouterOutput{}, err
		}
		return RouterOutput{Output: result}, nil
	})

	// 添加节点
	if err := graph.AddLambdaNode("booking", bookingLambda); err != nil {
		fmt.Printf("添加 booking 节点失败: %v\n", err)
		os.Exit(1)
	}
	if err := graph.AddLambdaNode("info", infoLambda); err != nil {
		fmt.Printf("添加 info 节点失败: %v\n", err)
		os.Exit(1)
	}
	if err := graph.AddLambdaNode("unclear", unclearLambda); err != nil {
		fmt.Printf("添加 unclear 节点失败: %v\n", err)
		os.Exit(1)
	}

	// 创建分支：根据决策路由到不同的处理程序
	branch := compose.NewGraphBranch(
		func(ctx context.Context, input RouterInput) (string, error) {
			decision := strings.TrimSpace(strings.ToLower(input.Decision))
			switch decision {
			case "booker":
				return "booking", nil
			case "info":
				return "info", nil
			default:
				return "unclear", nil
			}
		},
		map[string]bool{
			"booking": true,
			"info":    true,
			"unclear": true,
		},
	)

	// 从 START 添加分支
	if err := graph.AddBranch(compose.START, branch); err != nil {
		fmt.Printf("添加分支失败: %v\n", err)
		os.Exit(1)
	}

	// 所有处理程序节点都连接到 END
	if err := graph.AddEdge("booking", compose.END); err != nil {
		fmt.Printf("添加 booking->END 边失败: %v\n", err)
		os.Exit(1)
	}
	if err := graph.AddEdge("info", compose.END); err != nil {
		fmt.Printf("添加 info->END 边失败: %v\n", err)
		os.Exit(1)
	}
	if err := graph.AddEdge("unclear", compose.END); err != nil {
		fmt.Printf("添加 unclear->END 边失败: %v\n", err)
		os.Exit(1)
	}

	// 编译 Graph
	delegationGraph, err := graph.Compile(ctx)
	if err != nil {
		fmt.Printf("编译委托图失败: %v\n", err)
		os.Exit(1)
	}

	// --- 组合路由链和委托图 ---
	// 创建一个协调器函数，首先执行路由链获取决策，然后将决策和原始请求传递给委托图
	coordinatorAgentFunc := func(ctx context.Context, request string) (string, error) {
		// 步骤 1: 执行路由链获取决策
		decision, err := routerChain.Invoke(ctx, map[string]any{
			"request": request,
		})
		if err != nil {
			return "", fmt.Errorf("路由链执行失败: %w", err)
		}

		// 步骤 2: 将决策和原始请求传递给委托图
		result, err := delegationGraph.Invoke(ctx, RouterInput{
			Request:  request,
			Decision: decision,
		})
		if err != nil {
			return "", fmt.Errorf("委托图执行失败: %w", err)
		}

		return result.Output, nil
	}

	// --- 示例用法 ---
	fmt.Println("\n--- 运行预订请求 ---")
	requestA := "给我预订去伦敦的航班。"
	resultA, err := coordinatorAgentFunc(ctx, requestA)
	if err != nil {
		fmt.Printf("执行失败: %v\n", err)
	} else {
		fmt.Printf("最终结果 A: %s\n", resultA)
	}

	fmt.Println("\n--- 运行信息请求 ---")
	requestB := "意大利的首都是什么？"
	resultB, err := coordinatorAgentFunc(ctx, requestB)
	if err != nil {
		fmt.Printf("执行失败: %v\n", err)
	} else {
		fmt.Printf("最终结果 B: %s\n", resultB)
	}

	fmt.Println("\n--- 运行不清楚的请求 ---")
	requestC := "告诉我关于量子物理学的事。"
	resultC, err := coordinatorAgentFunc(ctx, requestC)
	if err != nil {
		fmt.Printf("执行失败: %v\n", err)
	} else {
		fmt.Printf("最终结果 C: %s\n", resultC)
	}
}
