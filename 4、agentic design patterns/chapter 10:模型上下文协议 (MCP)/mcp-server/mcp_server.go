package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"time"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"
)

const (
	// Server configuration
	serverName    = "Greeter MCP Server"
	serverVersion = "1.0.0"
	serverAddr    = ":8080"

	// Operation types for calculator
	opAdd      = "add"
	opSubtract = "subtract"
	opMultiply = "multiply"
	opDivide   = "divide"

	// Time format types
	timeFormatDateTime = "datetime"
	timeFormatDate     = "date"
	timeFormatTime     = "time"

	// Default values
	defaultTimeFormat = timeFormatDateTime
)

var (
	// ErrNameRequired is returned when the name parameter is missing
	ErrNameRequired = errors.New("name parameter is required")
	// ErrDivisionByZero is returned when attempting to divide by zero
	ErrDivisionByZero = errors.New("division by zero")
	// ErrUnknownOperation is returned for unsupported operations
	ErrUnknownOperation = errors.New("unknown operation")
)

func main() {
	srv := setupServer()
	registerTools(srv)

	if err := startServer(srv); err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
}

// setupServer creates and configures a new MCP server.
func setupServer() *server.MCPServer {
	return server.NewMCPServer(serverName, serverVersion,
		server.WithToolCapabilities(true),
	)
}

// registerTools registers all available tools with the server.
func registerTools(srv *server.MCPServer) {
	srv.AddTool(createGreetTool(), handleGreet)
	srv.AddTool(createCalculateTool(), handleCalculate)
	srv.AddTool(createGetCurrentTimeTool(), handleGetCurrentTime)
}

// startServer starts the HTTP server on the configured address.
func startServer(srv *server.MCPServer) error {
	log.Printf("Starting MCP server on %s", serverAddr)
	httpServer := server.NewStreamableHTTPServer(srv)
	return httpServer.Start(serverAddr)
}

// createGreetTool creates the greet tool definition.
func createGreetTool() mcp.Tool {
	return mcp.NewTool("greet",
		mcp.WithDescription("向指定的人打招呼"),
		mcp.WithString("name",
			mcp.Required(),
			mcp.Description("要打招呼的人的姓名"),
		),
	)
}

// createCalculateTool creates the calculator tool definition.
func createCalculateTool() mcp.Tool {
	return mcp.NewTool("calculate",
		mcp.WithDescription("执行基本的数学运算"),
		mcp.WithString("operation",
			mcp.Required(),
			mcp.Enum(opAdd, opSubtract, opMultiply, opDivide),
			mcp.Description("要执行的操作：add(加), subtract(减), multiply(乘), divide(除)"),
		),
		mcp.WithNumber("x", mcp.Required(), mcp.Description("第一个数字")),
		mcp.WithNumber("y", mcp.Required(), mcp.Description("第二个数字")),
	)
}

// createGetCurrentTimeTool creates the get_current_time tool definition.
func createGetCurrentTimeTool() mcp.Tool {
	return mcp.NewTool("get_current_time",
		mcp.WithDescription("获取当前系统时间"),
		mcp.WithString("format",
			mcp.Enum(timeFormatDateTime, timeFormatDate, timeFormatTime),
			mcp.DefaultString(defaultTimeFormat),
			mcp.Description("时间格式：datetime(日期时间), date(仅日期), time(仅时间)"),
		),
	)
}

// handleGreet processes greet tool calls.
func handleGreet(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	name, err := req.RequireString("name")
	if err != nil {
		return mcp.NewToolResultError(ErrNameRequired.Error()), nil
	}

	greeting := fmt.Sprintf("你好，%s！很高兴见到你！", name)
	return mcp.NewToolResultText(greeting), nil
}

// handleCalculate processes calculator tool calls.
func handleCalculate(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	operation, err := req.RequireString("operation")
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("operation parameter is required: %v", err)), nil
	}

	x, err := req.RequireFloat("x")
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("x parameter is required: %v", err)), nil
	}

	y, err := req.RequireFloat("y")
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("y parameter is required: %v", err)), nil
	}

	result, err := performCalculation(operation, x, y)
	if err != nil {
		return mcp.NewToolResultError(err.Error()), nil
	}

	return mcp.NewToolResultText(result), nil
}

// performCalculation performs the requested mathematical operation.
func performCalculation(operation string, x, y float64) (string, error) {
	var result float64

	switch operation {
	case opAdd:
		result = x + y
	case opSubtract:
		result = x - y
	case opMultiply:
		result = x * y
	case opDivide:
		if y == 0 {
			return "", ErrDivisionByZero
		}
		result = x / y
	default:
		return "", fmt.Errorf("%w: %s", ErrUnknownOperation, operation)
	}

	return formatCalculationResult(operation, x, y, result), nil
}

// formatCalculationResult formats the calculation result as a string.
func formatCalculationResult(operation string, x, y, result float64) string {
	var opSymbol string
	switch operation {
	case opAdd:
		opSymbol = "+"
	case opSubtract:
		opSymbol = "-"
	case opMultiply:
		opSymbol = "×"
	case opDivide:
		opSymbol = "÷"
	}
	return fmt.Sprintf("%.2f %s %.2f = %.2f", x, opSymbol, y, result)
}

// handleGetCurrentTime processes get_current_time tool calls.
func handleGetCurrentTime(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	format := req.GetString("format", defaultTimeFormat)
	timeStr := formatTime(time.Now(), format)

	result := fmt.Sprintf("当前时间 (%s): %s", format, timeStr)
	return mcp.NewToolResultText(result), nil
}

// formatTime formats the given time according to the specified format.
func formatTime(t time.Time, format string) string {
	switch format {
	case timeFormatDateTime:
		return t.Format("2006-01-02 15:04:05")
	case timeFormatDate:
		return t.Format("2006-01-02")
	case timeFormatTime:
		return t.Format("15:04:05")
	default:
		// Fallback to datetime format
		return t.Format("2006-01-02 15:04:05")
	}
}
