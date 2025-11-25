/*
规划 (Planning) 是 Agent 的“实时导航系统”：它将模糊的最终目标转化为可执行的“动态待办清单 (Dynamic To-Do List)”，并具备在执行过程中根据反馈随时“重新规划 (Re-planning)”的能力。
*/
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
)

// float32Ptr: 辅助函数
func float32Ptr(f float32) *float32 {
	return &f
}

// --- Todo 数据结构 ---
type TodoItem struct {
	ID          string    `json:"id"`
	Title       string    `json:"title"`
	Description string    `json:"description"`
	Status      string    `json:"status"` // "pending", "in_progress", "completed"
	CreatedAt   time.Time `json:"created_at"`
	CompletedAt time.Time `json:"completed_at,omitempty"`
	Result      string    `json:"result,omitempty"`
}

type TodoList struct {
	Items []TodoItem `json:"items"`
}

// --- Todo 管理工具 ---

// TodoManagerTool: Todo List 管理工具
type TodoManagerTool struct {
	todos *TodoList
}

func NewTodoManagerTool() *TodoManagerTool {
	return &TodoManagerTool{
		todos: &TodoList{
			Items: make([]TodoItem, 0),
		},
	}
}

func (t *TodoManagerTool) Info(ctx context.Context) (*schema.ToolInfo, error) {
	return &schema.ToolInfo{
		Name: "todo_manager",
		Desc: "管理 Todo List：添加任务、更新状态、查看列表",
		ParamsOneOf: schema.NewParamsOneOfByParams(map[string]*schema.ParameterInfo{
			"action": {
				Type:     schema.String,
				Desc:     "操作类型：'add'（添加任务）、'update'（更新状态）、'list'（查看列表）、'complete'（完成任务）",
				Required: true,
			},
			"id": {
				Type:     schema.String,
				Desc:     "任务 ID（用于 update 和 complete 操作）",
				Required: false,
			},
			"title": {
				Type:     schema.String,
				Desc:     "任务标题（用于 add 操作）",
				Required: false,
			},
			"description": {
				Type:     schema.String,
				Desc:     "任务描述（用于 add 操作）",
				Required: false,
			},
			"status": {
				Type:     schema.String,
				Desc:     "任务状态：'pending'、'in_progress'、'completed'（用于 update 操作）",
				Required: false,
			},
			"result": {
				Type:     schema.String,
				Desc:     "任务执行结果（用于 complete 操作）",
				Required: false,
			},
		}),
	}, nil
}

func (t *TodoManagerTool) InvokableRun(ctx context.Context, argumentsInJSON string, opts ...tool.Option) (string, error) {
	var args struct {
		Action      string `json:"action"`
		ID          string `json:"id,omitempty"`
		Title       string `json:"title,omitempty"`
		Description string `json:"description,omitempty"`
		Status      string `json:"status,omitempty"`
		Result      string `json:"result,omitempty"`
	}

	if err := json.Unmarshal([]byte(argumentsInJSON), &args); err != nil {
		return "", fmt.Errorf("无效的参数: %w", err)
	}

	fmt.Printf("\n--- 🛠️ 工具调用：todo_manager，操作：'%s' ---\n", args.Action)

	switch args.Action {
	case "add":
		id := fmt.Sprintf("todo-%d", len(t.todos.Items)+1)
		todo := TodoItem{
			ID:          id,
			Title:       args.Title,
			Description: args.Description,
			Status:      "pending",
			CreatedAt:   time.Now(),
		}
		t.todos.Items = append(t.todos.Items, todo)
		fmt.Printf("✅ 已添加任务: %s - %s\n", id, args.Title)
		return fmt.Sprintf("任务已添加: ID=%s, 标题=%s", id, args.Title), nil

	case "update":
		for i := range t.todos.Items {
			if t.todos.Items[i].ID == args.ID {
				if args.Status != "" {
					t.todos.Items[i].Status = args.Status
				}
				fmt.Printf("✅ 已更新任务: %s, 状态=%s\n", args.ID, args.Status)
				return fmt.Sprintf("任务已更新: ID=%s, 状态=%s", args.ID, args.Status), nil
			}
		}
		return "", fmt.Errorf("未找到任务: %s", args.ID)

	case "complete":
		for i := range t.todos.Items {
			if t.todos.Items[i].ID == args.ID {
				t.todos.Items[i].Status = "completed"
				t.todos.Items[i].CompletedAt = time.Now()
				if args.Result != "" {
					t.todos.Items[i].Result = args.Result
				}
				fmt.Printf("✅ 已完成任务: %s\n", args.ID)
				return fmt.Sprintf("任务已完成: ID=%s, 结果=%s", args.ID, args.Result), nil
			}
		}
		return "", fmt.Errorf("未找到任务: %s", args.ID)

	case "list":
		return t.renderTodoList(), nil

	default:
		return "", fmt.Errorf("未知操作: %s", args.Action)
	}
}

// renderTodoList: 渲染 Todo List（类似 Cursor 的展示格式）
func (t *TodoManagerTool) renderTodoList() string {
	if len(t.todos.Items) == 0 {
		return "📋 Todo List 为空"
	}

	var sb strings.Builder
	sb.WriteString("\n")
	sb.WriteString("╔════════════════════════════════════════════════════════════╗\n")
	sb.WriteString("║                    📋 TODO LIST                            ║\n")
	sb.WriteString("╠════════════════════════════════════════════════════════════╣\n")

	for i, item := range t.todos.Items {
		// 状态图标
		var statusIcon string
		switch item.Status {
		case "completed":
			statusIcon = "✅"
		case "in_progress":
			statusIcon = "🔄"
		default:
			statusIcon = "⏳"
		}

		// 任务行
		sb.WriteString(fmt.Sprintf("║ %d. %s [%s] %s\n", i+1, statusIcon, item.ID, item.Title))
		if item.Description != "" {
			sb.WriteString(fmt.Sprintf("║    └─ %s\n", item.Description))
		}
		if item.Status == "completed" && item.Result != "" {
			sb.WriteString(fmt.Sprintf("║    └─ 结果: %s\n", item.Result))
		}
		if i < len(t.todos.Items)-1 {
			sb.WriteString("║\n")
		}
	}

	sb.WriteString("╚════════════════════════════════════════════════════════════╝\n")

	// 统计信息
	completed := 0
	inProgress := 0
	pending := 0
	for _, item := range t.todos.Items {
		switch item.Status {
		case "completed":
			completed++
		case "in_progress":
			inProgress++
		default:
			pending++
		}
	}

	sb.WriteString(fmt.Sprintf("\n📊 统计: 总计 %d | ✅ 已完成 %d | 🔄 进行中 %d | ⏳ 待处理 %d\n",
		len(t.todos.Items), completed, inProgress, pending))

	return sb.String()
}

// --- 规划工具 ---

// PlannerTool: 规划工具，根据目标生成 Todo List
type PlannerTool struct {
	todoManager *TodoManagerTool
}

func NewPlannerTool(todoManager *TodoManagerTool) *PlannerTool {
	return &PlannerTool{
		todoManager: todoManager,
	}
}

func (p *PlannerTool) Info(ctx context.Context) (*schema.ToolInfo, error) {
	return &schema.ToolInfo{
		Name: "planner",
		Desc: "根据用户目标规划并生成 Todo List。输入目标描述，自动分解为可执行的任务列表",
		ParamsOneOf: schema.NewParamsOneOfByParams(map[string]*schema.ParameterInfo{
			"goal": {
				Type:     schema.String,
				Desc:     "用户的目标描述，例如：'开发一个待办事项应用'、'分析公司财报'",
				Required: true,
			},
			"tasks": {
				Type:     schema.String,
				Desc:     "JSON 格式的任务列表，例如：'[{\"title\":\"任务1\",\"description\":\"描述1\"},{\"title\":\"任务2\",\"description\":\"描述2\"}]'",
				Required: true,
			},
		}),
	}, nil
}

func (p *PlannerTool) InvokableRun(ctx context.Context, argumentsInJSON string, opts ...tool.Option) (string, error) {
	var args struct {
		Goal  string `json:"goal"`
		Tasks string `json:"tasks"`
	}

	if err := json.Unmarshal([]byte(argumentsInJSON), &args); err != nil {
		return "", fmt.Errorf("无效的参数: %w", err)
	}

	fmt.Printf("\n--- 🧠 规划工具：目标='%s' ---\n", args.Goal)

	// 解析任务列表
	var tasks []struct {
		Title       string `json:"title"`
		Description string `json:"description"`
	}
	if err := json.Unmarshal([]byte(args.Tasks), &tasks); err != nil {
		return "", fmt.Errorf("无效的任务列表格式: %w", err)
	}

	// 添加任务到 Todo List
	for _, task := range tasks {
		_, err := p.todoManager.InvokableRun(ctx, fmt.Sprintf(`{"action":"add","title":"%s","description":"%s"}`,
			task.Title, task.Description))
		if err != nil {
			return "", fmt.Errorf("添加任务失败: %w", err)
		}
	}

	fmt.Printf("✅ 已规划 %d 个任务\n", len(tasks))
	return fmt.Sprintf("规划完成：已生成 %d 个任务", len(tasks)), nil
}

func main() {
	ctx := context.Background()

	// --- 配置 ---
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		fmt.Println("错误: 请设置 OPENAI_API_KEY 环境变量")
		os.Exit(1)
	}

	baseURL := os.Getenv("OPENAI_BASE_URL")

	// 创建 OpenAI ChatModel 配置
	config := &openai.ChatModelConfig{
		Model:       "qwen/Qwen2.5-Coder-32B-Instruct",
		APIKey:      apiKey,
		Temperature: float32Ptr(0.3),
	}

	if baseURL != "" {
		config.BaseURL = baseURL
	}

	// 初始化 LLM
	llm, err := openai.NewChatModel(ctx, config)
	if err != nil {
		fmt.Printf("初始化语言模型时出错: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("✅ 语言模型已初始化: %s\n\n", config.Model)

	// --- 创建工具 ---
	todoManager := NewTodoManagerTool()
	planner := NewPlannerTool(todoManager)

	tools := []tool.BaseTool{
		todoManager,
		planner,
	}

	// --- 创建 ReAct Agent ---
	agentConfig := &react.AgentConfig{
		ToolCallingModel: llm,
		ToolsConfig: compose.ToolsNodeConfig{
			Tools: tools,
		},
		MaxStep: 20,
	}

	agent, err := react.NewAgent(ctx, agentConfig)
	if err != nil {
		fmt.Printf("创建 Agent 失败: %v\n", err)
		os.Exit(1)
	}

	// --- 系统提示词：指导 Agent 使用规划模式 ---
	systemPrompt := `你是一个智能任务规划助手。当用户提出目标时，你需要：

1. 首先使用 planner 工具将目标分解为具体的任务列表
2. 使用 todo_manager 的 list 操作查看当前任务列表
3. 逐个执行任务，使用 todo_manager 更新任务状态（in_progress -> completed）
4. 每完成一个任务，记录执行结果
5. 定期使用 todo_manager 的 list 操作展示当前进度

请按照这个流程帮助用户完成任务。`

	// --- 示例：用户目标 ---
	userGoals := []string{
		"帮我规划一个简单的待办事项应用开发任务，包括：需求分析、UI设计、后端开发、测试",
	}

	for _, goal := range userGoals {
		fmt.Println(strings.Repeat("=", 70))
		fmt.Printf("🎯 用户目标: %s\n", goal)
		fmt.Println(strings.Repeat("=", 70))

		messages := []*schema.Message{
			schema.SystemMessage(systemPrompt),
			schema.UserMessage(goal),
		}

		// 执行 Agent
		response, err := agent.Generate(ctx, messages)
		if err != nil {
			fmt.Printf("🛑 Agent 执行期间发生错误：%v\n", err)
			continue
		}

		// 显示最终响应
		fmt.Println("\n" + strings.Repeat("-", 70))
		fmt.Println("🤖 Agent 最终响应:")
		fmt.Println(strings.Repeat("-", 70))
		fmt.Println(response.Content)

		// 显示最终的 Todo List
		fmt.Println("\n" + strings.Repeat("=", 70))
		fmt.Println("📋 最终 Todo List 状态:")
		fmt.Println(strings.Repeat("=", 70))
		finalList, _ := todoManager.InvokableRun(ctx, `{"action":"list"}`)
		fmt.Println(finalList)
	}
}
