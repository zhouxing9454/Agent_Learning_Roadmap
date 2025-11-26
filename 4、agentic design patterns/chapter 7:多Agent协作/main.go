/*
多 Agent 协作（Multi-Agent Collaboration）是 Agent 系统的"团队协作模式"，
它让多个具有不同专长的 Agent 协同工作，通过分工合作完成复杂任务。

此代码根据 MIT 许可证授权。
请参阅仓库中的 LICENSE 文件以获取完整许可文本。
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

func main() {
	ctx := context.Background()

	// --- 设置环境 ---
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		fmt.Println("错误: 未找到 OPENAI_API_KEY。请在您的环境变量中设置它")
		os.Exit(1)
	}

	baseURL := os.Getenv("OPENAI_BASE_URL")

	// --- 创建语言模型 ---
	config := &openai.ChatModelConfig{
		Model:       "qwen/Qwen2.5-Coder-32B-Instruct", // 使用支持工具调用的模型
		APIKey:      apiKey,
		Temperature: float32Ptr(0.7),
	}

	// 如果设置了自定义 BaseURL，则使用它（支持代理或兼容 API）
	if baseURL != "" {
		config.BaseURL = baseURL
	}

	llm, err := openai.NewChatModel(ctx, config)
	if err != nil {
		fmt.Printf("初始化语言模型时出错: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("✅ 语言模型已初始化: %s\n\n", config.Model)

	// ========== 创建第一个 Agent：研究分析师 ==========
	// 角色：高级研究分析师
	// 目标：查找并总结 AI 的最新趋势
	// 背景：经验丰富的研究分析师，擅长识别关键趋势和综合信息
	researchSystemPrompt := `你是一位经验丰富的研究分析师，擅长识别关键趋势和综合信息。
你的任务是查找并总结 AI 的最新趋势，重点关注实际应用和潜在影响。
请提供详细、准确且有价值的研究结果。`

	researchTemplate := prompt.FromMessages(
		schema.FString,
		schema.SystemMessage(researchSystemPrompt),
		schema.UserMessage("{query}"),
	)

	// 创建研究 Agent Chain：Template -> ChatModel
	// 这个 Chain 代表一个独立的 Agent，具有自己的角色和职责
	researcherChain, err := compose.NewChain[map[string]any, *schema.Message]().
		AppendChatTemplate(researchTemplate).
		AppendChatModel(llm).
		Compile(ctx)
	if err != nil {
		fmt.Printf("创建研究 Agent 失败: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("✅ 研究分析师 Agent 已创建")

	// ========== 创建第二个 Agent：技术内容作家 ==========
	// 角色：技术内容作家
	// 目标：基于研究发现撰写清晰且引人入胜的博客文章
	// 背景：熟练的作家，可以将复杂的技术主题转化为易于理解的内容
	writingSystemPrompt := `你是一位熟练的作家，可以将复杂的技术主题转化为易于理解的内容。
你的任务是基于研究发现撰写清晰且引人入胜的博客文章。
文章应该引人入胜且易于普通读者理解。`

	writingTemplate := prompt.FromMessages(
		schema.FString,
		schema.SystemMessage(writingSystemPrompt),
		schema.UserMessage("基于以下研究发现，撰写一篇 500 字的博客文章：\n\n{research_results}\n\n请确保文章引人入胜且易于普通读者理解。"),
	)

	// 创建写作 Agent Chain：Template -> ChatModel
	// 这个 Chain 代表另一个独立的 Agent，具有自己的角色和职责
	writerChain, err := compose.NewChain[map[string]any, *schema.Message]().
		AppendChatTemplate(writingTemplate).
		AppendChatModel(llm).
		Compile(ctx)
	if err != nil {
		fmt.Printf("创建写作 Agent 失败: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("✅ 技术内容作家 Agent 已创建")

	// ========== 创建多 Agent 协作 Graph ==========
	// 使用 Graph 来协调多个 Agent 的顺序执行
	// 输入：包含查询的 map
	// 输出：最终的消息（博客文章）
	graph := compose.NewGraph[map[string]any, *schema.Message]()

	// Lambda 节点 1：执行研究 Agent
	// 将研究 Agent Chain 包装为 Lambda，嵌入到 Graph 中
	researcherLambda := compose.InvokableLambda(func(ctx context.Context, input map[string]any) (*schema.Message, error) {
		fmt.Println("🔍 研究分析师 Agent 正在工作...")
		result, err := researcherChain.Invoke(ctx, input)
		if err != nil {
			return nil, fmt.Errorf("研究 Agent 执行失败: %w", err)
		}
		fmt.Println("✅ 研究分析师 Agent 完成工作")
		return result, nil
	})
	if err := graph.AddLambdaNode("researcher_agent", researcherLambda); err != nil {
		fmt.Printf("添加研究 Agent 节点失败: %v\n", err)
		os.Exit(1)
	}

	// Lambda 节点 2：准备写作输入
	// 将研究结果转换为写作 Agent 需要的输入格式
	prepareWritingInput := compose.InvokableLambda(func(ctx context.Context, msg *schema.Message) (map[string]any, error) {
		return map[string]any{
			"research_results": msg.Content,
		}, nil
	})
	if err := graph.AddLambdaNode("prepare_writing", prepareWritingInput); err != nil {
		fmt.Printf("添加准备写作节点失败: %v\n", err)
		os.Exit(1)
	}

	// Lambda 节点 3：执行写作 Agent
	// 将写作 Agent Chain 包装为 Lambda，嵌入到 Graph 中
	writerLambda := compose.InvokableLambda(func(ctx context.Context, input map[string]any) (*schema.Message, error) {
		fmt.Println("✍️  技术内容作家 Agent 正在工作...")
		result, err := writerChain.Invoke(ctx, input)
		if err != nil {
			return nil, fmt.Errorf("写作 Agent 执行失败: %w", err)
		}
		fmt.Println("✅ 技术内容作家 Agent 完成工作")
		return result, nil
	})
	if err := graph.AddLambdaNode("writer_agent", writerLambda); err != nil {
		fmt.Printf("添加写作 Agent 节点失败: %v\n", err)
		os.Exit(1)
	}

	// ========== 定义边的连接（顺序执行）==========
	// 执行流程：
	// START -> researcher_agent -> prepare_writing -> writer_agent -> END
	// 这实现了两个 Agent 的顺序协作：研究 Agent 先工作，然后写作 Agent 基于研究结果工作
	if err := graph.AddEdge(compose.START, "researcher_agent"); err != nil {
		fmt.Printf("添加 START->researcher_agent 边失败: %v\n", err)
		os.Exit(1)
	}
	if err := graph.AddEdge("researcher_agent", "prepare_writing"); err != nil {
		fmt.Printf("添加 researcher_agent->prepare_writing 边失败: %v\n", err)
		os.Exit(1)
	}
	if err := graph.AddEdge("prepare_writing", "writer_agent"); err != nil {
		fmt.Printf("添加 prepare_writing->writer_agent 边失败: %v\n", err)
		os.Exit(1)
	}
	if err := graph.AddEdge("writer_agent", compose.END); err != nil {
		fmt.Printf("添加 writer_agent->END 边失败: %v\n", err)
		os.Exit(1)
	}

	// 编译 Graph
	compiledGraph, err := graph.Compile(ctx)
	if err != nil {
		fmt.Printf("编译 Graph 失败: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("✅ 多 Agent 协作团队已创建")
	fmt.Println(strings.Repeat("=", 70))
	fmt.Println("## 使用 OpenAI API 运行博客创建团队... ##")
	fmt.Println(strings.Repeat("=", 70))

	// --- 执行团队 ---
	// 定义研究任务
	researchQuery := "研究 2024-2025 年人工智能中出现的前 3 个趋势。重点关注实际应用和潜在影响。"

	input := map[string]any{
		"query": researchQuery,
	}

	fmt.Printf("\n📋 研究任务: %s\n\n", researchQuery)

	// 执行团队（顺序执行多个 Agent）
	result, err := compiledGraph.Invoke(ctx, input)
	if err != nil {
		fmt.Printf("\n发生意外错误：%v\n", err)
		os.Exit(1)
	}

	// 显示最终结果
	fmt.Println(strings.Repeat("-", 70))
	fmt.Println("## 团队最终输出 ##")
	fmt.Println(strings.Repeat("-", 70))
	fmt.Println(result.Content)
	fmt.Println(strings.Repeat("=", 70))
}
