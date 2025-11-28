/*
	反思（Reflection）是 Agent 系统的"自我改进机制"，它通过多轮迭代的生成-审查-改进循环，让 AI 能够自我评估和优化输出，从而显著提升最终结果的质量和准确性。

	反思类型	   适用场景		      核心优势		           核心劣势					你的 Go Agent 该选谁？
	单轮反思	   简单任务		 快速改进		             改进有限			     简单代码生成、文本润色
	多轮反思	   复杂任务		 深度优化		             耗时较长			     复杂算法实现、架构设计
	对比反思	   多方案选择	 全面评估		             计算成本高			     方案选型、设计决策
	协作反思	   团队场景	   集思广益		             协调复杂			        代码审查、团队讨论

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

// ReflectionState: 反思循环的状态
type ReflectionState struct {
	CurrentCode    string
	MessageHistory []*schema.Message
	Iteration      int
}

func runReflectionLoop(ctx context.Context, llm *openai.ChatModel) error {
	// --- 核心任务 ---
	taskPrompt := `
你的任务是创建一个名为 calculate_factorial 的 Python 函数。

此函数应执行以下操作：
1. 接受单个整数 n 作为输入。
2. 计算其阶乘 (n!)。
3. 包含清楚解释函数功能的文档字符串。
4. 处理边缘情况：0 的阶乘是 1。
5. 处理无效输入：如果输入是负数，则引发 ValueError。
`

	// --- 构建生成链 ---
	// Lambda 函数：从 Message 中提取 Content
	extractContent := compose.InvokableLambda(func(ctx context.Context, msg *schema.Message) (string, error) {
		return msg.Content, nil
	})

	// 生成链：直接使用消息历史调用 LLM
	generateChain, err := compose.NewChain[[]*schema.Message, string]().
		AppendChatModel(llm).
		AppendLambda(extractContent).
		Compile(ctx)
	if err != nil {
		return fmt.Errorf("编译生成链失败: %w", err)
	}

	// --- 构建反思链 ---
	reflectorPrompt := prompt.FromMessages(
		schema.FString,
		schema.SystemMessage(`你是一名高级软件工程师和 Python 专家。
你的角色是执行细致的代码审查。
根据原始任务要求批判性地评估提供的 Python 代码。
查找错误、风格问题、缺失的边缘情况和改进领域。
如果代码完美并满足所有要求，用单一短语 'CODE_IS_PERFECT' 响应。
否则，提供批评的项目符号列表。`),
		schema.UserMessage("原始任务：\n{task_prompt}\n\n要审查的代码：\n{current_code}"),
	)

	// Lambda 函数：准备反思输入
	prepareReflection := compose.InvokableLambda(func(ctx context.Context, state ReflectionState) (map[string]any, error) {
		return map[string]any{
			"task_prompt":  taskPrompt,
			"current_code": state.CurrentCode,
		}, nil
	})

	// 反思链：Lambda -> Template -> ChatModel -> Lambda
	reflectionChain, err := compose.NewChain[ReflectionState, string]().
		AppendLambda(prepareReflection).
		AppendChatTemplate(reflectorPrompt).
		AppendChatModel(llm).
		AppendLambda(extractContent).
		Compile(ctx)
	if err != nil {
		return fmt.Errorf("编译反思链失败: %w", err)
	}

	// --- 反思循环 ---
	maxIterations := 3
	state := ReflectionState{
		CurrentCode:    "",
		MessageHistory: []*schema.Message{schema.UserMessage(taskPrompt)},
		Iteration:      0,
	}

	for i := 0; i < maxIterations; i++ {
		state.Iteration = i + 1
		fmt.Printf("\n%s 反思循环：迭代 %d %s\n", strings.Repeat("=", 25), state.Iteration, strings.Repeat("=", 25))

		// --- 1. 生成/完善阶段 ---
		if i == 0 {
			fmt.Println("\n>>> 阶段 1：生成初始代码...")
			// 第一次迭代：直接使用消息历史生成代码
			response, err := generateChain.Invoke(ctx, state.MessageHistory)
			if err != nil {
				return fmt.Errorf("生成代码失败: %w", err)
			}
			state.CurrentCode = response
		} else {
			fmt.Println("\n>>> 阶段 1：基于先前批评完善代码...")
			// 后续迭代：添加完善指令
			improveMessage := schema.UserMessage("请使用提供的批评完善代码。")
			improveHistory := append(state.MessageHistory, improveMessage)
			response, err := generateChain.Invoke(ctx, improveHistory)
			if err != nil {
				return fmt.Errorf("完善代码失败: %w", err)
			}
			state.CurrentCode = response
		}

		fmt.Printf("\n--- 生成的代码 (v%d) ---\n%s\n", state.Iteration, state.CurrentCode)

		// 将生成的代码添加到历史记录
		state.MessageHistory = append(state.MessageHistory, &schema.Message{
			Role:    schema.Assistant,
			Content: state.CurrentCode,
		})

		// --- 2. 反思阶段 ---
		fmt.Println("\n>>> 阶段 2：对生成的代码进行反思...")
		critique, err := reflectionChain.Invoke(ctx, state)
		if err != nil {
			return fmt.Errorf("反思失败: %w", err)
		}

		// --- 3. 停止条件 ---
		if strings.Contains(critique, "CODE_IS_PERFECT") {
			fmt.Println("\n--- 批评 ---\n未发现进一步批评。代码令人满意。")
			break
		}

		fmt.Printf("\n--- 批评 ---\n%s\n", critique)

		// 将批评添加到历史记录以用于下一个完善循环
		critiqueMessage := schema.UserMessage(fmt.Sprintf("对先前代码的批评：\n%s", critique))
		state.MessageHistory = append(state.MessageHistory, critiqueMessage)
	}

	fmt.Printf("\n%s 最终结果 %s\n", strings.Repeat("=", 30), strings.Repeat("=", 30))
	fmt.Println("\n反思过程后的最终精炼代码：\n")
	fmt.Println(state.CurrentCode)

	return nil
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
		Temperature: float32Ptr(0.1), // 使用较低的温度以获得更确定性的输出
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

	// 运行反思循环
	if err := runReflectionLoop(ctx, llm); err != nil {
		fmt.Printf("反思循环执行失败: %v\n", err)
		os.Exit(1)
	}
}
