/*
	并行化（Parallelization）是 Agent 系统的"效率加速器"，它通过同时执行多个独立任务，将串行等待转化为并行处理，从而大幅缩短整体执行时间，提升系统吞吐量和响应速度。

	并行化类型	   适用场景		      核心优势		           核心劣势					     你的 Go Agent 该选谁？
	任务并行	   多个独立任务		 显著缩短总时间		     需要任务间无依赖			     多任务处理（如同时生成摘要、问题、术语）
	数据并行	   相同任务不同数据	 提高吞吐量		         需要数据分片和聚合			     批量处理（如同时处理多个文档）
	流水线并行   任务有依赖关系	 提高资源利用率	     需要任务分段和缓冲			     复杂工作流（如预处理→处理→后处理）
	混合并行	   复杂场景		   最大化性能		         实现复杂度高				     生产环境优化（结合多种并行策略）

	此代码根据 MIT 许可证授权。
	请参阅仓库中的 LICENSE 文件以获取完整许可文本。
*/

package main

import (
	"context"
	"fmt"
	"os"

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
		Temperature: float32Ptr(0.7), // 设置为 0.7 以获得更自然的输出
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

	// --- 定义独立链 ---
	// 这三个链代表可以并行执行的不同任务。

	// 1. 摘要链：简洁地总结主题
	summarizePrompt := prompt.FromMessages(
		schema.FString,
		schema.SystemMessage("简洁地总结以下主题："),
		schema.UserMessage("{topic}"),
	)

	// Lambda 函数：从 Message 中提取 Content
	extractContent := compose.InvokableLambda(func(ctx context.Context, msg *schema.Message) (string, error) {
		return msg.Content, nil
	})

	// 构建摘要链：Template -> ChatModel -> Lambda
	summarizeChain, err := compose.NewChain[map[string]any, string]().
		AppendChatTemplate(summarizePrompt).
		AppendChatModel(llm).
		AppendLambda(extractContent).
		Compile(ctx)
	if err != nil {
		fmt.Printf("编译摘要链失败: %v\n", err)
		os.Exit(1)
	}

	// 2. 问题链：生成关于主题的三个有趣问题
	questionsPrompt := prompt.FromMessages(
		schema.FString,
		schema.SystemMessage("生成关于以下主题的三个有趣问题："),
		schema.UserMessage("{topic}"),
	)

	questionsChain, err := compose.NewChain[map[string]any, string]().
		AppendChatTemplate(questionsPrompt).
		AppendChatModel(llm).
		AppendLambda(extractContent).
		Compile(ctx)
	if err != nil {
		fmt.Printf("编译问题链失败: %v\n", err)
		os.Exit(1)
	}

	// 3. 术语链：识别关键术语
	termsPrompt := prompt.FromMessages(
		schema.FString,
		schema.SystemMessage("从以下主题中识别 5-10 个关键术语，用逗号分隔："),
		schema.UserMessage("{topic}"),
	)

	termsChain, err := compose.NewChain[map[string]any, string]().
		AppendChatTemplate(termsPrompt).
		AppendChatModel(llm).
		AppendLambda(extractContent).
		Compile(ctx)
	if err != nil {
		fmt.Printf("编译术语链失败: %v\n", err)
		os.Exit(1)
	}

	// --- 构建并行图 ---
	// 使用 Graph 实现并行执行，每个链作为一个节点
	// 定义输入类型：包含主题
	type ParallelInput struct {
		Topic string
	}

	// 创建并行图，输出类型为 map[string]any 以便访问各个节点的输出
	parallelGraph := compose.NewGraph[ParallelInput, map[string]any]()

	// Lambda 节点：执行摘要链
	summarizeLambda := compose.InvokableLambda(func(ctx context.Context, input ParallelInput) (string, error) {
		result, err := summarizeChain.Invoke(ctx, map[string]any{
			"topic": input.Topic,
		})
		if err != nil {
			return "", fmt.Errorf("摘要链执行失败: %w", err)
		}
		return result, nil
	})

	// Lambda 节点：执行问题链
	questionsLambda := compose.InvokableLambda(func(ctx context.Context, input ParallelInput) (string, error) {
		result, err := questionsChain.Invoke(ctx, map[string]any{
			"topic": input.Topic,
		})
		if err != nil {
			return "", fmt.Errorf("问题链执行失败: %w", err)
		}
		return result, nil
	})

	// Lambda 节点：执行术语链
	termsLambda := compose.InvokableLambda(func(ctx context.Context, input ParallelInput) (string, error) {
		result, err := termsChain.Invoke(ctx, map[string]any{
			"topic": input.Topic,
		})
		if err != nil {
			return "", fmt.Errorf("术语链执行失败: %w", err)
		}
		return result, nil
	})

	// Lambda 节点：传递原始主题
	topicLambda := compose.InvokableLambda(func(ctx context.Context, input ParallelInput) (string, error) {
		return input.Topic, nil
	})

	// 添加节点，使用 WithOutputKey 指定输出键名
	if err := parallelGraph.AddLambdaNode("summarize", summarizeLambda, compose.WithOutputKey("summary")); err != nil {
		fmt.Printf("添加 summarize 节点失败: %v\n", err)
		os.Exit(1)
	}
	if err := parallelGraph.AddLambdaNode("questions", questionsLambda, compose.WithOutputKey("questions")); err != nil {
		fmt.Printf("添加 questions 节点失败: %v\n", err)
		os.Exit(1)
	}
	if err := parallelGraph.AddLambdaNode("terms", termsLambda, compose.WithOutputKey("key_terms")); err != nil {
		fmt.Printf("添加 terms 节点失败: %v\n", err)
		os.Exit(1)
	}
	if err := parallelGraph.AddLambdaNode("topic", topicLambda, compose.WithOutputKey("topic")); err != nil {
		fmt.Printf("添加 topic 节点失败: %v\n", err)
		os.Exit(1)
	}

	// 所有并行节点从 START 开始（实现并行执行）
	if err := parallelGraph.AddEdge(compose.START, "summarize"); err != nil {
		fmt.Printf("添加 START->summarize 边失败: %v\n", err)
		os.Exit(1)
	}
	if err := parallelGraph.AddEdge(compose.START, "questions"); err != nil {
		fmt.Printf("添加 START->questions 边失败: %v\n", err)
		os.Exit(1)
	}
	if err := parallelGraph.AddEdge(compose.START, "terms"); err != nil {
		fmt.Printf("添加 START->terms 边失败: %v\n", err)
		os.Exit(1)
	}
	if err := parallelGraph.AddEdge(compose.START, "topic"); err != nil {
		fmt.Printf("添加 START->topic 边失败: %v\n", err)
		os.Exit(1)
	}

	// 所有并行节点直接连接到 END（图会自动合并所有 WithOutputKey 的输出）
	if err := parallelGraph.AddEdge("summarize", compose.END); err != nil {
		fmt.Printf("添加 summarize->END 边失败: %v\n", err)
		os.Exit(1)
	}
	if err := parallelGraph.AddEdge("questions", compose.END); err != nil {
		fmt.Printf("添加 questions->END 边失败: %v\n", err)
		os.Exit(1)
	}
	if err := parallelGraph.AddEdge("terms", compose.END); err != nil {
		fmt.Printf("添加 terms->END 边失败: %v\n", err)
		os.Exit(1)
	}
	if err := parallelGraph.AddEdge("topic", compose.END); err != nil {
		fmt.Printf("添加 topic->END 边失败: %v\n", err)
		os.Exit(1)
	}

	// 编译并行图，使用 AllPredecessor 触发模式确保所有节点完成后再返回结果
	compiledParallelGraph, err := parallelGraph.Compile(ctx, compose.WithNodeTriggerMode(compose.AllPredecessor))
	if err != nil {
		fmt.Printf("编译并行图失败: %v\n", err)
		os.Exit(1)
	}

	// --- 构建综合链 ---
	// 定义将组合并行结果的最终综合提示词
	synthesisPrompt := prompt.FromMessages(
		schema.FString,
		schema.SystemMessage(`基于以下信息：
    摘要：{summary}
    相关问题：{questions}
    关键术语：{key_terms}
    综合一个全面的答案。`),
		schema.UserMessage("原始主题：{topic}"),
	)

	// Lambda 函数：将 map 结果转换为综合提示词的输入
	prepareSynthesis := compose.InvokableLambda(func(ctx context.Context, results map[string]any) (map[string]any, error) {
		// 从结果中提取各个字段
		summary, _ := results["summary"].(string)
		questions, _ := results["questions"].(string)
		keyTerms, _ := results["key_terms"].(string)
		topic, _ := results["topic"].(string)

		return map[string]any{
			"summary":   summary,
			"questions": questions,
			"key_terms": keyTerms,
			"topic":     topic,
		}, nil
	})

	// 构建综合链：Lambda -> Template -> ChatModel -> Lambda
	synthesisChain, err := compose.NewChain[map[string]any, string]().
		AppendLambda(prepareSynthesis).
		AppendChatTemplate(synthesisPrompt).
		AppendChatModel(llm).
		AppendLambda(extractContent).
		Compile(ctx)
	if err != nil {
		fmt.Printf("编译综合链失败: %v\n", err)
		os.Exit(1)
	}

	// --- 组合完整链 ---
	// 创建完整的并行处理函数
	fullParallelChainFunc := func(ctx context.Context, topic string) (string, error) {
		// 步骤 1: 执行并行图
		parallelResult, err := compiledParallelGraph.Invoke(ctx, ParallelInput{Topic: topic})
		if err != nil {
			return "", fmt.Errorf("并行图执行失败: %w", err)
		}

		// 步骤 2: 执行综合链
		finalResult, err := synthesisChain.Invoke(ctx, parallelResult)
		if err != nil {
			return "", fmt.Errorf("综合链执行失败: %w", err)
		}

		return finalResult, nil
	}

	// --- 运行链 ---
	testTopic := "太空探索的历史"
	fmt.Printf("\n--- 运行主题的并行处理示例：'%s' ---\n", testTopic)

	response, err := fullParallelChainFunc(ctx, testTopic)
	if err != nil {
		fmt.Printf("\n链执行期间发生错误：%v\n", err)
		os.Exit(1)
	}

	fmt.Println("\n--- 最终响应 ---")
	fmt.Println(response)
}
