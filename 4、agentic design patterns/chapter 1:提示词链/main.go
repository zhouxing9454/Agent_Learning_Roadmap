/*
	提示词链路（Prompt Chaining）：	核心是将复杂的任务目标解耦为顺序执行的模块化步骤。这种"分而治之"的策略能有效降低模型的认知负荷，通过中间过程的可控性，显著降低幻觉率并提升最终结果的准确性。

	上下文工程 (Context Engineering)： 在Token生成之前，系统性地构建各种信息，为AI提供一个完整的“决策环境”。
		核心理念： 上下文的质量 > 模型本身的架构。只要环境给的信息够丰富、够精准，模型的表现就能大幅提升。
		组成要素：
			Prompt Engineering： 基础指令。
			RAG (检索增强)： 主动查阅外部文档。
			Tools (工具输出)： 获取实时数据（如日历、API）。
			State/History (状态与历史)： 记住用户是谁，之前聊了什么。
			Memory (记忆)： 长期存储的关键信息。
			Structured Outputs (结构化输出)： 保证输出机器可读。

	我的理解：
		提示词链路就是拆分用户任务，像一条定义好的链路，一步一步执行，提高了model的准确率
		上下文工程通过prompt，tools，rag，memory，结构化输出，state/history，让model获得更多的外部知识以及上下文信息
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
	// ctx: 创建根上下文(非nil的空Context)，用于控制整个程序的执行流程
	ctx := context.Background()

	// apiKey: 从环境变量OPENAI_API_KEY读取 OpenAI API 密钥
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" { // 检查 API Key 是否为空
		fmt.Println("错误: 请设置 OPENAI_API_KEY 环境变量")
		os.Exit(1)
	}

	// baseURL: 从环境变量OPENAI_BASE_URL读取自定义 API 端点地址（可选）
	baseURL := os.Getenv("OPENAI_BASE_URL")

	// config: 创建 OpenAI 兼容模型的配置对象
	config := &openai.ChatModelConfig{
		Model:       "Qwen/Qwen2.5-Coder-32B-Instruct", // Model: 指定要使用的模型名称
		APIKey:      apiKey,                            // APIKey: 设置 API 密钥，用于身份验证
		Temperature: float32Ptr(0),                     // Temperature: 控制模型输出的随机性(0: 每次相同输入得到相同输出,0.0-2.0: 值越大，输出越随机和创造性)
	}

	// 如果设置了自定义 BaseURL，则使用它（支持代理或兼容 API）
	if baseURL != "" {
		config.BaseURL = baseURL
	}

	// llm: 创建 LLM 模型实例
	llm, err := openai.NewChatModel(ctx, config)
	if err != nil {
		fmt.Printf("初始化模型失败: %v\n", err)
		os.Exit(1)
	}

	// --- 提示词 1：提取信息 ---
	promptExtract := prompt.FromMessages(
		schema.FString,
		schema.UserMessage("从以下文本中提取技术规格：\n\n{text_input}"),
	)

	// --- 提示词 2：转换为 JSON ---
	promptTransform := prompt.FromMessages(
		schema.FString,
		schema.UserMessage("将以下规格转换为 JSON 对象，使用 'cpu'、'memory' 和 'storage' 作为键：\n\n{specifications}"),
	)

	// ========== Lambda 函数1: Message -> string ==========
	// 作用: 从 Message 对象中提取 Content 字段，转换为字符串
	// 对应 Python: StrOutputParser()
	extractContent := compose.InvokableLambda(func(ctx context.Context, msg *schema.Message) (string, error) {
		return msg.Content, nil
	})

	// ========== 构建提取链 ==========
	// 链结构: Template -> ChatModel -> Lambda
	// 对应 Python: extraction_chain = prompt_extract | llm | StrOutputParser()
	extractionChain, err := compose.NewChain[map[string]any, string]().
		AppendChatTemplate(promptExtract). // map -> []*Message
		AppendChatModel(llm).              // []*Message -> *Message
		AppendLambda(extractContent).      // *Message -> string
		Compile(ctx)
	if err != nil {
		fmt.Printf("编译提取链失败: %v\n", err)
		os.Exit(1)
	}

	// ========== Lambda 函数2: string -> map ==========
	// 作用: 将字符串包装成 map，供第二个链的模板使用
	// 对应 Python: {"specifications": extraction_chain}
	wrapSpecifications := compose.InvokableLambda(func(ctx context.Context, specifications string) (map[string]any, error) {
		return map[string]any{
			"specifications": specifications,
		}, nil
	})

	// ========== Lambda 函数3: Message -> string ==========
	// 作用: 从 Message 对象中提取 Content 字段
	// 对应 Python: StrOutputParser()
	extractFinalResult := compose.InvokableLambda(func(ctx context.Context, msg *schema.Message) (string, error) {
		return msg.Content, nil
	})

	// ========== 构建转换链 ==========
	// 链结构: Lambda -> Template -> ChatModel -> Lambda
	// 对应 Python: {"specifications": extraction_chain} | prompt_transform | llm | StrOutputParser()
	transformChain, err := compose.NewChain[string, string]().
		AppendLambda(wrapSpecifications).    // string -> map
		AppendChatTemplate(promptTransform). // map -> []*Message
		AppendChatModel(llm).                // []*Message -> *Message
		AppendLambda(extractFinalResult).    // *Message -> string
		Compile(ctx)
	if err != nil {
		fmt.Printf("编译转换链失败: %v\n", err)
		os.Exit(1)
	}

	// ========== 执行链 ==========
	inputText := "新款笔记本电脑型号配备 3.5 GHz 八核处理器、16GB 内存和 1TB NVMe 固态硬盘。"

	// 执行提取链
	extractedSpecs, err := extractionChain.Invoke(ctx, map[string]any{
		"text_input": inputText, // 键名必须与模板中的 {text_input} 占位符一致
	})
	if err != nil {
		fmt.Printf("提取链执行失败: %v\n", err)
		os.Exit(1)
	}

	// 执行转换链
	finalResult, err := transformChain.Invoke(ctx, extractedSpecs)
	if err != nil {
		fmt.Printf("转换链执行失败: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("\n--- 最终 JSON 输出 ---")
	fmt.Println(finalResult)
}
