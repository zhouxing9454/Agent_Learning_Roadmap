/*
记忆管理（Memory Management）是 Agent 系统的"信息持久化机制"，
它让 Agent 能够记住过去的交互、观察和学习经验，从而做出明智决策、
维护对话上下文并随时间持续改进。

记忆类型：
	短期记忆（上下文记忆）：
		- 类似于工作记忆，保存当前处理或最近访问的信息
		- 对于使用 LLM 的 Agent，短期记忆主要存在于上下文窗口中
		- 包含最近消息、Agent 回复、工具使用结果以及当前交互中的 Agent 反思
		- 上下文窗口容量有限，制约了 Agent 可直接访问的近期信息量
		- 实现逻辑：
		  1. 保存最近 N 轮完整对话（如最近 10 轮 user input 和 agent answer）
		  2. 超过 N 轮的部分，生成总结后也保存到短期记忆
		  3. 使用 Redis 作为内存数据库存储短期记忆，支持跨请求的会话持久化

	长期记忆（持久记忆）：
		- 作为 Agent 跨交互、任务或延长期间所需信息的存储库
		- 使用向量数据库（Elasticsearch 8）存储，支持基于语义相似性的检索
		- 当 Agent 需要长期记忆信息时，会查询向量数据库、检索相关数据并集成到短期上下文

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
	"time"

	openaiEmbedding "github.com/cloudwego/eino-ext/components/embedding/openai"
	es8Indexer "github.com/cloudwego/eino-ext/components/indexer/es8"
	openaiModel "github.com/cloudwego/eino-ext/components/model/openai"
	es8Retriever "github.com/cloudwego/eino-ext/components/retriever/es8"
	"github.com/cloudwego/eino-ext/components/retriever/es8/search_mode"
	"github.com/cloudwego/eino/components/embedding"
	"github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/components/prompt"
	"github.com/cloudwego/eino/compose"
	"github.com/cloudwego/eino/schema"
	"github.com/elastic/go-elasticsearch/v8"
	"github.com/elastic/go-elasticsearch/v8/typedapi/types"
	"github.com/go-redis/redis/v8"
)

// float32Ptr: 辅助函数，将 float32 值转换为 *float32 指针
func float32Ptr(f float32) *float32 {
	return &f
}

// ========== 短期记忆：Redis 存储 ==========

// Message: 对话消息
type Message struct {
	Role    string `json:"role"`    // "user" 或 "assistant"
	Content string `json:"content"` // 消息内容
	Time    int64  `json:"time"`    // 时间戳
}

// ShortTermMemory: 短期记忆管理器（使用 Redis）
type ShortTermMemory struct {
	redisClient *redis.Client
	maxHistory  int // 保存的完整对话轮数（如 10 轮）
	// 注意：不使用固定过期时间，而是通过会话管理来控制清理
	// 当会话被明确删除时，通过后台消息队列执行清理操作
}

// NewShortTermMemory: 创建短期记忆管理器
func NewShortTermMemory(redisAddr, redisPassword string, db int, maxHistory int) (*ShortTermMemory, error) {
	rdb := redis.NewClient(&redis.Options{
		Addr:     redisAddr,
		Password: redisPassword,
		DB:       db,
	})

	// 测试连接
	ctx := context.Background()
	if err := rdb.Ping(ctx).Err(); err != nil {
		return nil, fmt.Errorf("无法连接到 Redis: %w", err)
	}

	return &ShortTermMemory{
		redisClient: rdb,
		maxHistory:  maxHistory,
	}, nil
}

// AddMessage: 添加消息到会话历史
func (stm *ShortTermMemory) AddMessage(ctx context.Context, sessionID string, role string, content string) error {
	msg := Message{
		Role:    role,
		Content: content,
		Time:    time.Now().Unix(),
	}

	msgJSON, err := json.Marshal(msg)
	if err != nil {
		return fmt.Errorf("序列化消息失败: %w", err)
	}

	// 使用 Redis List 存储消息历史
	key := fmt.Sprintf("session:%s:messages", sessionID)
	if err := stm.redisClient.RPush(ctx, key, msgJSON).Err(); err != nil {
		return fmt.Errorf("存储消息失败: %w", err)
	}

	// 注意：不设置固定过期时间
	// 理由：
	// 1. 短期记忆应该跟随会话的生命周期，而不是固定的时间
	// 2. 如果用户重新打开会话，应该还能看到之前的对话历史
	// 3. 会话的清理应该由会话管理系统控制，通过后台消息队列执行
	// 4. 固定过期时间可能导致活跃会话被误删

	return nil
}

// GetHistory: 获取会话历史（最近 N 轮完整对话 + 总结）
func (stm *ShortTermMemory) GetHistory(ctx context.Context, sessionID string) ([]Message, string, error) {
	key := fmt.Sprintf("session:%s:messages", sessionID)

	// 获取所有消息
	msgJSONs, err := stm.redisClient.LRange(ctx, key, 0, -1).Result()
	if err != nil {
		return nil, "", fmt.Errorf("获取消息历史失败: %w", err)
	}

	var allMessages []Message
	for _, msgJSON := range msgJSONs {
		var msg Message
		if err := json.Unmarshal([]byte(msgJSON), &msg); err != nil {
			continue
		}
		allMessages = append(allMessages, msg)
	}

	// 计算完整对话轮数（每轮包含 user + assistant）
	totalRounds := len(allMessages) / 2

	// 如果消息数量在限制内，直接返回
	if totalRounds <= stm.maxHistory {
		return allMessages, "", nil
	}

	// 超过限制，需要总结
	// 保留最近 N 轮完整对话
	recentMessages := allMessages[len(allMessages)-stm.maxHistory*2:]

	// 检查是否有已保存的总结
	summaryKey := fmt.Sprintf("session:%s:summary", sessionID)
	summary, err := stm.redisClient.Get(ctx, summaryKey).Result()
	if err == redis.Nil {
		// 没有总结，需要生成（这里返回空，由调用者生成）
		// 注意：调用者会重新获取 allMessages 并计算 oldMessages 来调用 GenerateSummary
		return recentMessages, "", nil
	} else if err != nil {
		return nil, "", fmt.Errorf("获取总结失败: %w", err)
	}

	// 有总结，直接返回最近消息和总结
	return recentMessages, summary, nil
}

// SaveSummary: 保存总结
func (stm *ShortTermMemory) SaveSummary(ctx context.Context, sessionID string, summary string) error {
	summaryKey := fmt.Sprintf("session:%s:summary", sessionID)
	// 不设置过期时间，跟随会话生命周期
	if err := stm.redisClient.Set(ctx, summaryKey, summary, 0).Err(); err != nil {
		return fmt.Errorf("保存总结失败: %w", err)
	}
	return nil
}

// GenerateSummary: 生成旧对话的总结
func (stm *ShortTermMemory) GenerateSummary(ctx context.Context, sessionID string, llm model.BaseChatModel, oldMessages []Message) (string, error) {
	// 构建总结提示词
	var oldText strings.Builder
	for _, msg := range oldMessages {
		oldText.WriteString(fmt.Sprintf("%s: %s\n", msg.Role, msg.Content))
	}

	template := prompt.FromMessages(
		schema.FString,
		schema.SystemMessage("你是一个对话总结助手。请将以下对话历史总结为简洁的要点，保留关键信息和上下文。"),
		schema.UserMessage(fmt.Sprintf("请总结以下对话历史：\n\n%s", oldText.String())),
	)

	chain, err := compose.NewChain[map[string]any, *schema.Message]().
		AppendChatTemplate(template).
		AppendChatModel(llm).
		Compile(ctx)
	if err != nil {
		return "", fmt.Errorf("创建总结链失败: %w", err)
	}

	result, err := chain.Invoke(ctx, map[string]any{})
	if err != nil {
		return "", fmt.Errorf("生成总结失败: %w", err)
	}

	// 保存总结
	if err := stm.SaveSummary(ctx, sessionID, result.Content); err != nil {
		return "", err
	}

	return result.Content, nil
}

// ClearHistory: 清空会话历史
// 这个方法应该在会话被明确删除时调用
// 在实际应用中，可以通过后台消息队列异步执行清理操作
//
// 使用场景示例：
// 1. 用户主动删除会话 -> 立即调用 ClearHistory
// 2. 会话管理系统检测到会话被删除 -> 发送消息到队列 -> 后台任务执行清理
// 3. 定期清理非活跃会话（可选）-> 通过定时任务检测 -> 发送消息到队列 -> 后台任务执行清理
func (stm *ShortTermMemory) ClearHistory(ctx context.Context, sessionID string) error {
	key := fmt.Sprintf("session:%s:messages", sessionID)
	summaryKey := fmt.Sprintf("session:%s:summary", sessionID)
	accessKey := fmt.Sprintf("session:%s:last_access", sessionID)

	// 删除所有相关的 key
	if err := stm.redisClient.Del(ctx, key, summaryKey, accessKey).Err(); err != nil {
		return fmt.Errorf("清空历史失败: %w", err)
	}
	return nil
}

// UpdateLastAccessTime: 更新会话最后访问时间（可选，用于会话活跃度检测）
// 可以用于实现"长时间未访问的会话自动清理"功能
// 注意：这只是一个辅助功能，实际的清理决策应该由会话管理系统做出
func (stm *ShortTermMemory) UpdateLastAccessTime(ctx context.Context, sessionID string) error {
	accessKey := fmt.Sprintf("session:%s:last_access", sessionID)
	// 可以设置一个较长的过期时间（如 30 天），用于检测非活跃会话
	// 但这只是用于检测，实际的清理还是由会话管理系统控制
	if err := stm.redisClient.Set(ctx, accessKey, time.Now().Unix(), 30*24*time.Hour).Err(); err != nil {
		return fmt.Errorf("更新访问时间失败: %w", err)
	}
	return nil
}

// GetLastAccessTime: 获取会话最后访问时间（用于会话活跃度检测）
func (stm *ShortTermMemory) GetLastAccessTime(ctx context.Context, sessionID string) (int64, error) {
	accessKey := fmt.Sprintf("session:%s:last_access", sessionID)
	val, err := stm.redisClient.Get(ctx, accessKey).Int64()
	if err == redis.Nil {
		return 0, nil // 没有记录，返回 0
	}
	if err != nil {
		return 0, fmt.Errorf("获取访问时间失败: %w", err)
	}
	return val, nil
}

// ========== 长期记忆：Elasticsearch 8 向量数据库 ==========

// LongTermMemory: 长期记忆管理器（使用 Elasticsearch 8）
type LongTermMemory struct {
	indexer   *es8Indexer.Indexer
	retriever *es8Retriever.Retriever
	embedder  embedding.Embedder
}

// NewLongTermMemory: 创建长期记忆管理器
func NewLongTermMemory(ctx context.Context, esAddr, esUser, esPassword, indexName string, embedder embedding.Embedder) (*LongTermMemory, error) {
	// 1. 创建 Elasticsearch 客户端
	cfg := elasticsearch.Config{
		Addresses: []string{esAddr},
	}
	if esUser != "" && esPassword != "" {
		cfg.Username = esUser
		cfg.Password = esPassword
	}

	esClient, err := elasticsearch.NewClient(cfg)
	if err != nil {
		return nil, fmt.Errorf("创建 Elasticsearch 客户端失败: %w", err)
	}

	// 2. 测试连接
	res, err := esClient.Info()
	if err != nil {
		return nil, fmt.Errorf("连接 Elasticsearch 失败: %w", err)
	}
	defer res.Body.Close()
	fmt.Println("✅ Elasticsearch 连接成功")

	// 3. 定义字段映射
	const (
		fieldContent       = "content"
		fieldContentVector = "content_vector"
		fieldMetadata      = "metadata"
	)

	// 4. 创建索引器
	indexer, err := es8Indexer.NewIndexer(ctx, &es8Indexer.IndexerConfig{
		Client:    esClient,
		Index:     indexName,
		BatchSize: 5,
		DocumentToFields: func(ctx context.Context, doc *schema.Document) (map[string]es8Indexer.FieldValue, error) {
			fields := make(map[string]es8Indexer.FieldValue)
			// 内容字段：存储原始文本，并设置 EmbedKey 指向向量字段
			// EmbedKey 表示：对这个字段的值进行向量化，并将向量存储到 content_vector 字段
			// 注意：不需要单独定义 content_vector 字段，EmbedKey 会自动创建它
			fields[fieldContent] = es8Indexer.FieldValue{
				Value:    doc.Content,
				EmbedKey: fieldContentVector, // 对 content 的值进行向量化，存储到 content_vector
			}
			// 元数据字段
			if len(doc.MetaData) > 0 {
				fields[fieldMetadata] = es8Indexer.FieldValue{
					Value: doc.MetaData,
				}
			}
			return fields, nil
		},
		Embedding: embedder,
	})
	if err != nil {
		return nil, fmt.Errorf("创建索引器失败: %w", err)
	}

	// 5. 创建检索器
	retriever, err := es8Retriever.NewRetriever(ctx, &es8Retriever.RetrieverConfig{
		Client: esClient,
		Index:  indexName,
		TopK:   5,
		SearchMode: search_mode.SearchModeApproximate(&search_mode.ApproximateConfig{
			QueryFieldName:  fieldContent,
			VectorFieldName: fieldContentVector,
			Hybrid:          true, // 启用混合搜索（文本 + 向量）
			RRF:             false,
		}),
		ResultParser: func(ctx context.Context, hit types.Hit) (doc *schema.Document, err error) {
			doc = &schema.Document{
				ID:       *hit.Id_,
				Content:  "",
				MetaData: make(map[string]any),
			}

			var src map[string]any
			if err = json.Unmarshal(hit.Source_, &src); err != nil {
				return nil, err
			}

			// 解析字段
			if val, ok := src[fieldContent]; ok {
				doc.Content = val.(string)
			}
			if val, ok := src[fieldContentVector]; ok {
				var v []float64
				for _, item := range val.([]interface{}) {
					v = append(v, item.(float64))
				}
				doc.WithDenseVector(v)
			}
			if val, ok := src[fieldMetadata]; ok {
				if metaMap, ok := val.(map[string]any); ok {
					doc.MetaData = metaMap
				}
			}

			if hit.Score_ != nil {
				doc.WithScore(float64(*hit.Score_))
			}

			return doc, nil
		},
		Embedding: embedder,
	})
	if err != nil {
		return nil, fmt.Errorf("创建检索器失败: %w", err)
	}

	return &LongTermMemory{
		indexer:   indexer,
		retriever: retriever,
		embedder:  embedder,
	}, nil
}

// Store: 存储长期记忆
func (ltm *LongTermMemory) Store(ctx context.Context, content string, metadata map[string]interface{}) (string, error) {
	doc := &schema.Document{
		Content:  content,
		MetaData: metadata,
	}

	ids, err := ltm.indexer.Store(ctx, []*schema.Document{doc})
	if err != nil {
		return "", fmt.Errorf("存储长期记忆失败: %w", err)
	}

	if len(ids) == 0 {
		return "", fmt.Errorf("存储失败：未返回 ID")
	}

	return ids[0], nil
}

// Retrieve: 检索长期记忆
func (ltm *LongTermMemory) Retrieve(ctx context.Context, query string) ([]*schema.Document, error) {
	docs, err := ltm.retriever.Retrieve(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("检索长期记忆失败: %w", err)
	}
	return docs, nil
}

// DropIndex: 删除 Elasticsearch 索引（用于清理旧数据）
func DropIndex(ctx context.Context, esAddr, esUser, esPassword, indexName string) error {
	cfg := elasticsearch.Config{
		Addresses: []string{esAddr},
	}
	if esUser != "" && esPassword != "" {
		cfg.Username = esUser
		cfg.Password = esPassword
	}

	esClient, err := elasticsearch.NewClient(cfg)
	if err != nil {
		return fmt.Errorf("创建 Elasticsearch 客户端失败: %w", err)
	}

	// 检查索引是否存在
	res, err := esClient.Indices.Exists([]string{indexName})
	if err != nil {
		return fmt.Errorf("检查索引是否存在失败: %w", err)
	}
	res.Body.Close()

	if res.StatusCode == 404 {
		fmt.Printf("索引 '%s' 不存在，无需删除\n", indexName)
		return nil
	}

	// 删除索引
	res, err = esClient.Indices.Delete([]string{indexName})
	if err != nil {
		return fmt.Errorf("删除索引失败: %w", err)
	}
	res.Body.Close()

	fmt.Printf("✅ 已删除索引 '%s'\n", indexName)
	return nil
}

// ========== 主程序 ==========

func main() {
	ctx := context.Background()

	// --- 环境变量配置 ---
	openaiAPIKey := os.Getenv("OPENAI_API_KEY")
	if openaiAPIKey == "" {
		fmt.Println("错误: 未找到 OPENAI_API_KEY")
		os.Exit(1)
	}

	openaiBaseURL := os.Getenv("OPENAI_BASE_URL")

	// Redis 配置
	redisAddr := os.Getenv("REDIS_ADDR")
	if redisAddr == "" {
		redisAddr = "localhost:6379"
	}
	// redisPassword := os.Getenv("REDIS_PASSWORD")
	redisPassword := "ningzaichun"
	redisDB := 0

	// Elasticsearch 配置
	esAddr := os.Getenv("ES_ADDR")
	if esAddr == "" {
		esAddr = "http://localhost:9200"
	}
	esUser := os.Getenv("ES_USER")
	esPassword := os.Getenv("ES_PASSWORD")
	indexName := os.Getenv("ES_INDEX")
	if indexName == "" {
		indexName = "eino_memory" // Elasticsearch 索引名称
	}

	// --- 初始化 LLM ---
	llmConfig := &openaiModel.ChatModelConfig{
		Model:       "Qwen/Qwen3-VL-8B-Instruct",
		APIKey:      openaiAPIKey,
		Temperature: float32Ptr(0.7),
		BaseURL:     openaiBaseURL, // 直接设置 BaseURL
	}

	llm, err := openaiModel.NewChatModel(ctx, llmConfig)
	if err != nil {
		fmt.Printf("初始化语言模型失败: %v\n", err)
		os.Exit(1)
	}
	fmt.Println("✅ LLM 模型已初始化")

	// --- 初始化 Embedding 模型 ---
	embedderConfig := &openaiEmbedding.EmbeddingConfig{
		APIKey:  openaiAPIKey,
		Model:   "Qwen/Qwen3-Embedding-8B", // 或使用其他 embedding 模型
		Timeout: 30 * time.Second,
		BaseURL: openaiBaseURL, // 直接设置 BaseURL
	}
	embedder, err := openaiEmbedding.NewEmbedder(ctx, embedderConfig)
	if err != nil {
		fmt.Printf("初始化 Embedding 模型失败: %v\n", err)
		os.Exit(1)
	}
	fmt.Println("✅ Embedding 模型已初始化")

	// --- 初始化短期记忆（Redis）---
	shortTermMemory, err := NewShortTermMemory(redisAddr, redisPassword, redisDB, 10) // 保存最近 10 轮完整对话
	if err != nil {
		fmt.Printf("初始化短期记忆失败: %v\n", err)
		fmt.Println("提示: 请确保 Redis 服务正在运行")
		os.Exit(1)
	}
	fmt.Println("✅ 短期记忆（Redis）已初始化")

	// --- 初始化长期记忆（Elasticsearch 8）---
	// 自动删除旧索引（避免字段定义冲突）
	fmt.Printf("正在清理旧的 Elasticsearch 索引 '%s'...\n", indexName)
	if err := DropIndex(ctx, esAddr, esUser, esPassword, indexName); err != nil {
		fmt.Printf("⚠️ 删除索引失败（可能不存在）: %v\n", err)
	}

	fmt.Println("正在初始化长期记忆（Elasticsearch 8）...")
	longTermMemory, err := NewLongTermMemory(ctx, esAddr, esUser, esPassword, indexName, embedder)
	if err != nil {
		fmt.Printf("❌ 初始化长期记忆失败: %v\n", err)
		fmt.Println("\n可能的原因：")
		fmt.Println("1. Elasticsearch 服务未运行（请检查 Docker 容器状态）")
		fmt.Println("2. 网络连接问题（请检查 Elasticsearch 地址和端口，默认 http://localhost:9200）")
		fmt.Println("3. 认证失败（请检查 ES_USER 和 ES_PASSWORD）")
		fmt.Println("4. Embedding 模型配置错误（请检查 API Key 和 BaseURL）")
		os.Exit(1)
	}
	fmt.Println("✅ 长期记忆（Elasticsearch 8）已初始化")

	// ========== 演示：完整的记忆管理流程 ==========
	fmt.Println("\n" + strings.Repeat("=", 70))
	fmt.Println("## 记忆管理演示：结合短期和长期记忆 ##")
	fmt.Println(strings.Repeat("=", 70))

	sessionID := "demo_session_001"

	// 模拟多轮对话
	testQueries := []string{
		"你好，我的名字是张三",
		"我是一名 Go 语言开发者",
		"我喜欢使用 Redis 和 Milvus",
		"我最近在学习 AI Agent 开发",
		"请记住：我的工作年限是 5 年",
		"我刚才说了什么？",
		"我的工作年限是多少？",
	}

	// 创建对话模板
	conversationTemplate := prompt.FromMessages(
		schema.FString,
		schema.SystemMessage(`你是一个智能助手，能够使用短期记忆（当前对话上下文）和长期记忆（用户的历史信息）来提供个性化的回复。
短期记忆包含最近的对话历史和总结，长期记忆包含用户的持久化信息。
请综合使用这两种记忆来提供连贯、个性化的回复。`),
		schema.UserMessage(`短期记忆（对话历史）：
{short_term_history}

长期记忆（用户信息）：
{long_term_memory}

当前用户输入：{user_input}`),
	)

	conversationChain, err := compose.NewChain[map[string]any, *schema.Message]().
		AppendChatTemplate(conversationTemplate).
		AppendChatModel(llm).
		Compile(ctx)
	if err != nil {
		fmt.Printf("创建对话链失败: %v\n", err)
		os.Exit(1)
	}

	for i, query := range testQueries {
		fmt.Printf("\n--- [轮次 %d] 用户输入: %s ---\n", i+1, query)

		// 1. 获取短期记忆（对话历史）
		recentMessages, summary, err := shortTermMemory.GetHistory(ctx, sessionID)
		if err != nil {
			fmt.Printf("获取短期记忆失败: %v\n", err)
			continue
		}

		// 构建短期记忆文本
		var shortTermHistory strings.Builder
		if summary != "" {
			shortTermHistory.WriteString(fmt.Sprintf("【对话总结】\n%s\n\n", summary))
		}
		shortTermHistory.WriteString("【最近对话】\n")
		for _, msg := range recentMessages {
			shortTermHistory.WriteString(fmt.Sprintf("%s: %s\n", msg.Role, msg.Content))
		}

		// 2. 检索长期记忆
		var longTermInfo strings.Builder
		if strings.Contains(query, "记住") || strings.Contains(query, "我的") {
			// 检索相关长期记忆
			docs, err := longTermMemory.Retrieve(ctx, query)
			if err == nil && len(docs) > 0 {
				longTermInfo.WriteString("检索到的相关信息：\n")
				for j, doc := range docs {
					longTermInfo.WriteString(fmt.Sprintf("%d. %s\n", j+1, doc.Content))
				}
			}
		}

		// 3. 生成回复
		result, err := conversationChain.Invoke(ctx, map[string]any{
			"short_term_history": shortTermHistory.String(),
			"long_term_memory":   longTermInfo.String(),
			"user_input":         query,
		})
		if err != nil {
			fmt.Printf("生成回复失败: %v\n", err)
			continue
		}

		response := result.Content
		fmt.Printf("助手回复: %s\n", response)

		// 4. 保存到短期记忆
		if err := shortTermMemory.AddMessage(ctx, sessionID, "user", query); err != nil {
			fmt.Printf("保存用户消息失败: %v\n", err)
		}
		if err := shortTermMemory.AddMessage(ctx, sessionID, "assistant", response); err != nil {
			fmt.Printf("保存助手消息失败: %v\n", err)
		}

		// 5. 检查是否需要生成总结
		allMessages, _, _ := shortTermMemory.GetHistory(ctx, sessionID)
		totalRounds := len(allMessages) / 2
		if totalRounds > shortTermMemory.maxHistory {
			// 需要生成总结
			oldMessages := allMessages[:len(allMessages)-shortTermMemory.maxHistory*2]
			_, err := shortTermMemory.GenerateSummary(ctx, sessionID, llm, oldMessages)
			if err != nil {
				fmt.Printf("生成总结失败: %v\n", err)
			} else {
				fmt.Println("✅ 已生成对话总结")
			}
		}

		// 6. 如果是需要长期记忆的信息，存储到长期记忆
		if strings.Contains(query, "请记住") {
			content := strings.TrimPrefix(query, "请记住：")
			id, err := longTermMemory.Store(ctx, content, map[string]interface{}{
				"session_id": sessionID,
				"type":       "user_fact",
				"timestamp":  time.Now().Unix(),
			})
			if err != nil {
				fmt.Printf("存储长期记忆失败: %v\n", err)
			} else {
				fmt.Printf("✅ 已存储到长期记忆，ID: %s\n", id)
			}
		}
	}

	fmt.Println("\n" + strings.Repeat("=", 70))
	fmt.Println("## 演示完成 ##")
	fmt.Println(strings.Repeat("=", 70))
	fmt.Println("\n关键要点：")
	fmt.Println("1. 短期记忆（Redis）：保存最近 N 轮完整对话，超过部分生成总结")
	fmt.Println("2. 长期记忆（Elasticsearch 8）：使用向量数据库存储用户持久化信息，支持语义检索和混合搜索")
	fmt.Println("3. 两种记忆结合使用，提供连贯、个性化的对话体验")
}
