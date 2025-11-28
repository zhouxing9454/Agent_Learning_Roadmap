package main

import (
	"context"
	"fmt"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"github.com/cloudwego/eino-ext/components/model/openai"
	"github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/components/prompt"
	"github.com/cloudwego/eino/compose"
	"github.com/cloudwego/eino/schema"
	"github.com/joho/godotenv"
)

// AgentState 定义了在 Graph 节点之间流转的全局状态
// 对应 Python 示例中的状态传递
type AgentState struct {
	UseCase       string
	Goals         []string
	CurrentCode   string
	Feedback      string
	Iteration     int
	MaxIterations int
	IsGoalMet     bool
}

const modelCallTimeout = 60 * time.Second

var fileNameCleanRe = regexp.MustCompile(`[^a-z0-9_]`)

// 辅助函数：将 float32 转为指针
func float32Ptr(f float32) *float32 {
	return &f
}

func main() {
	// 1. --- 环境设置 ---
	_ = godotenv.Load()
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatal("❌ 请设置 OPENAI_API_KEY 环境变量。")
	}

	ctx := context.Background()

	// 2. --- 初始化共享的 LLM 模型 ---
	fmt.Println("📡 初始化 OpenAI LLM (gpt-4o)...")
	chatModel, err := openai.NewChatModel(ctx, &openai.ChatModelConfig{
		Model:       "gpt-4o",
		Temperature: float32Ptr(0.3),
		APIKey:      apiKey,
		BaseURL:     os.Getenv("OPENAI_BASE_URL"),
	})
	if err != nil {
		log.Fatalf("无法初始化模型: %v", err)
	}

	// ========================================================================
	// 🏗️ Agent 1: Coder (程序员)
	// 职责：根据用例、目标和反馈生成代码
	// ========================================================================
	coderSystemPrompt := `你是一个 AI 编码专家。
你的工作是根据用户的用例编写 Python 代码。
如果提供了反馈，你需要根据反馈完善之前的代码。
只返回代码，不要包含 Markdown 标记或额外的解释。`

	coderTemplate := prompt.FromMessages(
		schema.FString,
		schema.SystemMessage(coderSystemPrompt),
		schema.UserMessage(`
用例：{{.UseCase}}

目标：
{{range .Goals}}- {{.}}
{{end}}

{{if .PreviousCode}}
之前生成的代码：
{{.PreviousCode}}
{{end}}

{{if .Feedback}}
对之前版本的反馈：
{{.Feedback}}
{{end}}

请仅返回修订后的 Python 代码。`),
	)

	// 创建 Coder Chain
	coderChain, err := compose.NewChain[map[string]any, *schema.Message]().
		AppendChatTemplate(coderTemplate).
		AppendChatModel(chatModel).
		Compile(ctx)
	if err != nil {
		log.Fatal(err)
	}

	// ========================================================================
	// 🏗️ Agent 2: Reviewer (审查员)
	// 职责：根据目标审查代码并给出反馈
	// ========================================================================
	reviewerSystemPrompt := `你是一个严格的代码审查员。
你的任务是根据设定的目标列表检查代码。
指出代码中的缺陷、边缘情况处理不当或不符合目标的地方。`

	reviewerTemplate := prompt.FromMessages(
		schema.FString,
		schema.SystemMessage(reviewerSystemPrompt),
		schema.UserMessage(`
基于以下目标：
{{range .Goals}}- {{.}}
{{end}}

请对此代码进行批评：
{{.Code}}

如果代码完美符合目标，请明确指出。`),
	)

	// 创建 Reviewer Chain
	reviewerChain, err := compose.NewChain[map[string]any, *schema.Message]().
		AppendChatTemplate(reviewerTemplate).
		AppendChatModel(chatModel).
		Compile(ctx)
	if err != nil {
		log.Fatal(err)
	}

	// ========================================================================
	// 🏗️ Agent 3: Judge (裁判)
	// 职责：判断是否达成目标，输出 True/False
	// ========================================================================
	judgeSystemPrompt := `你是一个决策者。你需要阅读代码审查的反馈，并判断所有目标是否都已达成。
仅输出 "True" 或 "False"。`

	judgeTemplate := prompt.FromMessages(
		schema.FString,
		schema.SystemMessage(judgeSystemPrompt),
		schema.UserMessage(`
目标列表：
{{range .Goals}}- {{.}}
{{end}}

审查反馈：
"""{{.Feedback}}"""

基于反馈，目标是否已完全达成？`),
	)

	// 创建 Judge Chain
	judgeChain, err := compose.NewChain[map[string]any, *schema.Message]().
		AppendChatTemplate(judgeTemplate).
		AppendChatModel(chatModel).
		Compile(ctx)
	if err != nil {
		log.Fatal(err)
	}

	// ========================================================================
	// 🕸️ 构建协作 Graph (编排 Agents)
	// ========================================================================
	graph := compose.NewGraph[AgentState, AgentState]()

	// --- 节点 1: Coder Node ---
	coderNode := compose.InvokableLambda(func(ctx context.Context, state AgentState) (AgentState, error) {
		state.Iteration++
		fmt.Printf("\n=== 🔁 迭代 %d / %d ===\n", state.Iteration, state.MaxIterations)
		fmt.Println("👨‍💻 Coder Agent 正在编写代码...")

		input := map[string]any{
			"UseCase":      state.UseCase,
			"Goals":        state.Goals,
			"PreviousCode": state.CurrentCode,
			"Feedback":     state.Feedback,
		}

		ctx2, cancel := context.WithTimeout(ctx, modelCallTimeout)
		defer cancel()
		resp, err := coderChain.Invoke(ctx2, input)
		if err != nil {
			return state, err
		}

		state.CurrentCode = cleanCodeBlock(resp.Content)
		printCodePreview(state.CurrentCode)
		return state, nil
	})

	// --- 节点 2: Reviewer Node ---
	reviewerNode := compose.InvokableLambda(func(ctx context.Context, state AgentState) (AgentState, error) {
		fmt.Println("🔍 Reviewer Agent 正在审查代码...")

		input := map[string]any{
			"Goals": state.Goals,
			"Code":  state.CurrentCode,
		}

		ctx2, cancel := context.WithTimeout(ctx, modelCallTimeout)
		defer cancel()
		resp, err := reviewerChain.Invoke(ctx2, input)
		if err != nil {
			return state, err
		}

		state.Feedback = resp.Content
		fmt.Printf("\n📥 审查反馈: %s\n", truncateString(state.Feedback, 100))
		return state, nil
	})

	// --- 节点 3: Judge Node ---
	judgeNode := compose.InvokableLambda(func(ctx context.Context, state AgentState) (AgentState, error) {
		fmt.Println("⚖️  Judge Agent 正在裁决...")

		input := map[string]any{
			"Goals":    state.Goals,
			"Feedback": state.Feedback,
		}

		ctx2, cancel := context.WithTimeout(ctx, modelCallTimeout)
		defer cancel()
		resp, err := judgeChain.Invoke(ctx2, input)
		if err != nil {
			return state, err
		}

		state.IsGoalMet = parseBoolFromLLM(resp.Content)

		if state.IsGoalMet {
			fmt.Println("✅ Judge 裁决：目标已达成 (True)")
		} else {
			fmt.Println("❌ Judge 裁决：目标未达成 (False)")
		}
		return state, nil
	})

	// 添加节点到 Graph
	_ = graph.AddLambdaNode("Coder", coderNode)
	_ = graph.AddLambdaNode("Reviewer", reviewerNode)
	_ = graph.AddLambdaNode("Judge", judgeNode)

	// 定义边 (Edges)
	_ = graph.AddEdge(compose.START, "Coder")
	_ = graph.AddEdge("Coder", "Reviewer")
	_ = graph.AddEdge("Reviewer", "Judge")
	_ = graph.AddEdge("Judge", compose.END)

	judgeBranch := compose.NewGraphBranch(func(ctx context.Context, state AgentState) (string, error) {
		if state.IsGoalMet {
			fmt.Println("🎉 流程结束：目标达成。")
			return compose.END, nil
		}
		if state.Iteration >= state.MaxIterations {
			fmt.Println("⚠️ 流程结束：达到最大迭代次数。")
			return compose.END, nil
		}
		fmt.Println("🔄 流程继续：返回 Coder 修改代码。")
		return "Coder", nil // 循环回到 Coder
	}, map[string]bool{
		"Coder":    true,
		"Reviewer": true,
		"Judge":    true,
	})
	_ = graph.AddBranch("Judge", judgeBranch)

	// 编译 Graph
	runnable, err := graph.Compile(ctx)
	if err != nil {
		log.Fatalf("编译 Graph 失败: %v", err)
	}

	// ========================================================================
	// 🚀 运行协作团队
	// ========================================================================

	// 示例任务
	useCase := "编写代码查找给定正整数的 BinaryGap"
	goalsInput := "代码简单易懂，功能正确，处理全面的边缘情况，仅接受正整数输入，打印结果并附带几个示例"
	parts := strings.Split(goalsInput, "，")
	if len(parts) == 1 {
		// 处理英文逗号
		parts = strings.Split(goalsInput, ",")
	}
	goals := make([]string, 0, len(parts))
	for _, g := range parts {
		g = strings.TrimSpace(g)
		if g != "" {
			goals = append(goals, g)
		}
	}

	initialState := AgentState{
		UseCase:       useCase,
		Goals:         goals,
		MaxIterations: 5,
		Iteration:     0,
	}

	fmt.Printf("\n🎯 任务：%s\n", useCase)
	fmt.Println(strings.Repeat("=", 50))

	finalState, err := runnable.Invoke(ctx, initialState)
	if err != nil {
		log.Fatalf("运行失败: %v", err)
	}

	// 保存结果
	if finalState.CurrentCode != "" {
		finalCode := addCommentHeader(finalState.CurrentCode, finalState.UseCase)
		saveCodeToFile(ctx, chatModel, finalCode, finalState.UseCase)
	}
}

// --- 🛠️ 实用工具函数 ---

func cleanCodeBlock(code string) string {
	lines := strings.Split(strings.TrimSpace(code), "\n")
	if len(lines) > 0 && strings.HasPrefix(strings.TrimSpace(lines[0]), "```") {
		lines = lines[1:]
	}
	if len(lines) > 0 && strings.TrimSpace(lines[len(lines)-1]) == "```" {
		lines = lines[:len(lines)-1]
	}
	return strings.TrimSpace(strings.Join(lines, "\n"))
}

func printCodePreview(code string) {
	lines := strings.Split(code, "\n")
	fmt.Println("🧾 代码预览：")
	if len(lines) > 5 {
		fmt.Println(strings.Join(lines[:5], "\n"))
		fmt.Println("... (剩余代码已隐藏)")
	} else {
		fmt.Println(code)
	}
}

func truncateString(s string, max int) string {
	// 统一去掉换行，按 rune 截断，避免 UTF-8 乱码
	clean := strings.ReplaceAll(s, "\n", " ")
	runes := []rune(clean)
	if len(runes) > max {
		return string(runes[:max]) + "..."
	}
	return clean
}

func addCommentHeader(code string, useCase string) string {
	return fmt.Sprintf("# 用例: %s\n\n%s", useCase, code)
}

// 注意：这里我们复用了 main 中的 chatModel 来生成文件名，所以传入 model.ChatModel
func saveCodeToFile(ctx context.Context, m model.ChatModel, code string, useCase string) {
	fmt.Println("\n💾 保存最终文件...")

	// 使用一个临时的 Chain 来生成文件名
	namePrompt := prompt.FromMessages(schema.FString, schema.UserMessage("为以下Python代码用例生成一个简短的文件名(只返回文件名,无后缀,全小写,下划线): {{.UseCase}}"))

	// 简单的直接调用（带超时与错误回退）
	ctx2, cancel := context.WithTimeout(ctx, 15*time.Second)
	defer cancel()
	msgs, err := namePrompt.Format(ctx2, map[string]any{"UseCase": useCase})
	var genContent string
	if err == nil {
		resp, genErr := m.Generate(ctx2, msgs)
		if genErr == nil {
			genContent = resp.Content
		}
	}

	baseName := strings.TrimSpace(genContent)
	baseName = fileNameCleanRe.ReplaceAllString(baseName, "")
	if baseName == "" {
		baseName = "script"
	}
	if len(baseName) > 15 {
		baseName = baseName[:15]
	}

	fileName := fmt.Sprintf("%s_%d.py", baseName, rand.Intn(1000)+1000)

	cwd, _ := os.Getwd()
	outDir := filepath.Join(cwd, "outputs")
	_ = os.MkdirAll(outDir, 0755)
	path := filepath.Join(outDir, fileName)
	if err := os.WriteFile(path, []byte(code), 0644); err != nil {
		fmt.Printf("❌ 写入文件失败: %v\n", err)
		return
	}
	fmt.Printf("✅ 文件已保存: %s\n", filepath.Join("outputs", fileName))
}

func parseBoolFromLLM(s string) bool {
	res := strings.ToLower(strings.TrimSpace(s))
	// 去除常见的标点与空白
	res = strings.Trim(res, ".!\"'` ")
	return res == "true"
}
