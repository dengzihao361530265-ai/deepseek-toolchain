# deepseek-toolchain

允许本地部署的 DeepSeek R1 等推理模型通过 Function Call 的方式调用外部工具。

## 功能特性

- 支持 OpenAI API 兼容接口（如 vLLM、DeepSeek API）
- 支持 LangChain 生态工具集成
- 内置多种工具：网络搜索、维基百科、代码执行
- 可扩展的工具调用框架，方便自定义扩展

## 项目结构

```
deepseek-toolchain/
├── agent_prompt.py        # 工具调用提示词模板
├── openai_agent.py        # OpenAI 兼容 Agent 实现
├── langchain_agent.py     # LangChain Agent 实现
├── openai_test.py         # OpenAI API 测试客户端
└── langchain_test.py      # LangChain 测试客户端
```

## 集成工具

| 工具 | 说明 |
|------|------|
| Tavily Search | 网络搜索（需配置 API Key） |
| Wikipedia | 维基百科查询 |
| DuckDuckGo Search | DuckDuckGo 搜索 |
| Python REPL | Python 代码执行 |

## 安装依赖

```bash
pip install langchain-openai langchain-experimental langchain-community tavily-python
```

## 使用方法

### 方法一：使用 OpenAI 兼容接口（推荐）

编辑 `openai_test.py`，填入你的配置：

```python
api_key = 'your-api-key'           # 你的 API Key
base_url = 'https://api.deepseek.com'  # 或你的 vLLM 地址
model_name = 'deepseek-reasoner'   # 模型名称

tavily_api_key = 'your-tavily-api-key'  # Tavily API Key
```

运行：

```bash
python openai_test.py
```

### 方法二：使用 LangChain 接口

编辑 `langchain_test.py`，填入你的配置：

```python
api_key = 'your-api-key'
base_url = 'your-vllm-url'  # vLLM 部署地址
model_name = 'your-model'

tavily_api_key = 'your-tavily-api-key'
```

运行：

```bash
python langchain_test.py
```

## 自定义扩展

你可以在 `openai_test.py` 或 `langchain_test.py` 中添加更多工具：

```python
from langchain_community.tools import YourCustomTool

tools = [search, wikipedia, duck_search, pythonREPLTool, YourCustomTool()]
```

## 工具调用协议

Agent 通过以下格式调用工具：

```
[TOOL]
tool: 'tool_name'
parameters: {'arg1': value1, 'arg2': value2}
[/TOOL]
```

通过以下格式返回最终结果：

```
[END]
最终的响应内容
[/END]
```

## License

本项目基于 木兰宽松许可证 第2版（Mulan PSL v2）开源。
