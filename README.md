# AI Workbench

AI Workbench 是一个基于 FastAPI、React、SQLite 与 LiteLLM 的前后端分离项目骨架。
当前包含多会话流式聊天，以及 PDF、TXT、Markdown 文件上传与管理基础能力。

## 技术栈

- Backend: FastAPI + SQLAlchemy
- Frontend: React + TypeScript + Vite
- Database: SQLite + SQLAlchemy
- AI Layer: LiteLLM

## 目录结构

```text
ai-workbench/
├── backend/                 # FastAPI 后端工程
│   ├── app/
│   │   ├── api/             # HTTP API 路由
│   │   ├── core/            # 配置等基础设施
│   │   ├── db/              # 数据库连接与会话
│   │   ├── services/        # 外部服务适配层（包括 LiteLLM）
│   │   └── main.py          # FastAPI 应用入口
│   ├── tests/               # 后端测试
│   ├── .env.example         # 后端环境变量示例
│   └── pyproject.toml       # Python 项目与依赖配置
├── frontend/                # React 前端工程
│   ├── src/                 # 前端源代码
│   ├── .env.example         # 前端环境变量示例
│   ├── package.json         # Node.js 脚本与依赖
│   ├── tsconfig*.json       # TypeScript 配置
│   └── vite.config.ts       # Vite 配置
└── .gitignore
```

## 本地开发

### 后端

需要 Python 3.11 或更高版本。

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env
uvicorn app.main:app --reload
```

后端默认运行于 `http://localhost:8000`，交互式 API 文档位于
`http://localhost:8000/docs`。健康检查接口为 `GET /api/v1/health`，对话接口为
`POST /api/v1/chat`。流式对话接口为 `POST /api/v1/chat/stream`。

文件接口：

- `POST /api/v1/files/upload`：上传 PDF、TXT 或 Markdown 文件，单文件上限 20 MiB。
- `GET /api/v1/files`：按上传时间倒序获取文件列表。

上传文件保存在 `backend/data/uploads`，元数据保存在 SQLite 的 `documents` 表中。
上传成功后会自动提取文本，并以约 1000 字符、150 字符重叠的规则写入
`document_chunks` 表。PDF 需要包含可提取的文本层；当前版本不包含 OCR。

每个 chunk 会通过本地确定性 Embedding 服务生成 384 维向量，并持久化到
`backend/data/chroma` 下的 ChromaDB。该基础 Embedding 无需外部 API 或模型下载，
后续可在不改变存储接口的前提下替换为更强的语义模型。

```text
GET /api/v1/search?query=搜索内容&limit=5
```

检索接口使用 cosine 距离返回相关 chunks，`limit` 支持 1–20，默认为 5。

设置 `RAG_ENABLED=true` 后，可由聊天请求的 `rag_enabled` 字段启用知识库模式。
知识库模式会检索相关 chunks、构造临时增强 Prompt，再通过原有 LiteLLM/DeepSeek
聊天链路生成普通或流式回答。前端对话页提供“知识库”开关；关闭时保持普通聊天模式。

## MCP Server

AI Workbench 提供独立的只读/AI stdio MCP Server：

```bash
cd backend
python -m app.mcp_server
```

它只暴露 `search_knowledge`、`list_documents` 和 `ask_workbench`，不提供文件修改、
Shell 或 Git 工具。`ask_workbench` 不创建 conversation/message 记录。

Codex 的 `~/.codex/config.toml` 可增加以下配置；项目不会自动修改该文件：

```toml
[mcp_servers.ai_workbench]
command = "/home/cqz12314/AI/Projects/ai-workbench/backend/.venv/bin/python"
args = ["-m", "app.mcp_server"]
cwd = "/home/cqz12314/AI/Projects/ai-workbench/backend"
```

配置后请启动新的 Codex 会话，使客户端重新加载 MCP server 列表。

### 前端

需要 Node.js 20 或更高版本。

```bash
cd frontend
npm install
cp .env.example .env
npm run dev
```

前端默认运行于 `http://localhost:5173`。开发环境下，`/api` 请求由 Vite
代理至后端。

## 配置原则

- 本地配置写入各工程的 `.env`，不要提交密钥。
- SQLite 数据库默认创建在 `backend/data/ai_workbench.db`，包含 conversation 和
  message 记录；页面启动时自动恢复最近一次对话。
- LiteLLM 的模型通过 `LITELLM_MODEL` 配置。DeepSeek 使用 provider 前缀，例如
  `deepseek/deepseek-v4-flash` 或 `deepseek/deepseek-v4-pro`。
- DeepSeek 使用 `DEEPSEEK_API_KEY`，API Base 默认为官方
  `https://api.deepseek.com`；也可通过 `DEEPSEEK_API_BASE` 显式配置。
- 其他模型仍使用所选供应商的标准环境变量（例如 `OPENAI_API_KEY`）。只在本地
  `.env` 中填写真实值，`.env.example` 不包含真实密钥且 `.env` 已被 Git 忽略。
- 对话历史持久化到本地 SQLite；密钥只从环境变量读取，不会写入数据库。

## 常用命令

```bash
# 后端测试
cd backend && pytest

# 后端代码检查
cd backend && ruff check .

# 前端类型检查与生产构建
cd frontend && npm run build

# 前端 ESLint
cd frontend && npm run lint
```
