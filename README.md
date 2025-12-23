# SuperInsight AI 数据治理与标注平台

SuperInsight 是一款专为 AI 时代设计的企业级语料治理与智能标注平台，深度借鉴龙石数据成熟的"理采存管用"方法论，同时针对大模型（LLM）和生成式 AI（GenAI）应用场景进行全面升级。

## 特性

- 🔒 **安全数据提取**: 只读权限提取各种数据源
- 🤖 **AI 预标注**: 集成多种 LLM 模型进行智能预标注
- 👥 **人机协同**: 支持业务专家、技术专家协作标注
- 📊 **质量管理**: 基于 Ragas 的语义质量评估
- 💰 **计费结算**: 精确的工时和条数统计
- 🛡️ **安全合规**: 企业级安全控制和审计
- ☁️ **多部署**: 支持云托管、私有化、混合云部署

## 技术架构

- **核心引擎**: Label Studio
- **数据存储**: PostgreSQL + JSONB
- **缓存**: Redis
- **Web 框架**: FastAPI
- **AI 集成**: Ollama, HuggingFace, 国产 LLM APIs
- **部署**: Docker Compose, 腾讯云 TCB

## 快速开始

### 环境要求

- Python 3.11+
- PostgreSQL 15+
- Redis 7+
- Docker & Docker Compose (可选)

### 本地开发环境

1. **克隆项目**
```bash
git clone https://github.com/Angus1976/superinsight1225.git
cd superinsight1225
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **配置环境变量**
```bash
cp .env.example .env
# 编辑 .env 文件，配置数据库连接等信息
```

4. **启动数据库服务**
```bash
# 使用 Docker Compose 启动所有服务
docker-compose up -d postgres redis label-studio

# 或者手动启动 PostgreSQL 和 Redis
```

5. **初始化数据库**
```bash
# 数据库会通过 init-db.sql 自动初始化
# 或者手动运行初始化脚本
psql -h localhost -U superinsight -d superinsight -f scripts/init-db.sql
```

6. **启动应用**
```bash
python main.py
```

### Docker 部署

使用 Docker Compose 一键启动完整环境：

```bash
# 启动所有服务
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f superinsight-api
```

服务访问地址：
- SuperInsight API: http://localhost:8000
- Label Studio: http://localhost:8080
- PostgreSQL: localhost:5432
- Redis: localhost:6379

### 腾讯云 TCB 部署

1. **安装 TCB CLI**
```bash
npm install -g @cloudbase/cli
```

2. **配置 TCB 环境**
```bash
# 登录腾讯云
tcb login

# 配置环境变量
cp .env.example .env
# 编辑 .env 文件，配置 TCB 相关信息
```

3. **部署到 TCB**
```bash
# 部署云托管服务
tcb framework deploy
```

## 项目结构

```
superinsight-platform/
├── src/                          # 源代码目录
│   ├── models/                   # 数据模型
│   ├── config/                   # 配置管理
│   ├── database/                 # 数据库连接
│   ├── extractors/               # 数据提取器
│   ├── label_studio/             # Label Studio 集成
│   ├── ai/                       # AI 预标注服务
│   ├── quality/                  # 质量管理
│   ├── billing/                  # 计费系统
│   ├── security/                 # 安全控制
│   ├── api/                      # API 接口
│   └── utils/                    # 工具函数
├── tests/                        # 测试代码
├── scripts/                      # 脚本文件
├── .kiro/specs/                  # 项目规范文档
├── docker-compose.yml            # Docker 编排文件
├── requirements.txt              # Python 依赖
├── .env.example                  # 环境变量模板
└── main.py                       # 应用入口
```

## 配置说明

### 数据库配置

```bash
# PostgreSQL 配置
DATABASE_URL=postgresql://username:password@localhost:5432/superinsight
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_NAME=superinsight
DATABASE_USER=username
DATABASE_PASSWORD=password
```

### Label Studio 配置

```bash
# Label Studio 配置
LABEL_STUDIO_URL=http://localhost:8080
LABEL_STUDIO_API_TOKEN=your_api_token_here
LABEL_STUDIO_PROJECT_ID=1
```

### AI 服务配置

```bash
# Ollama 本地模型
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2

# HuggingFace 模型
HUGGINGFACE_API_TOKEN=your_token_here
HUGGINGFACE_MODEL=bert-base-chinese

# 国产 LLM APIs
ZHIPU_API_KEY=your_zhipu_key_here
BAIDU_API_KEY=your_baidu_key_here
ALIBABA_API_KEY=your_alibaba_key_here
TENCENT_API_KEY=your_tencent_key_here
```

## 开发指南

### 代码规范

项目使用以下工具确保代码质量：

```bash
# 代码格式化
black src/ tests/

# 导入排序
isort src/ tests/

# 类型检查
mypy src/

# 运行测试
pytest tests/ -v --cov=src
```

### 数据库迁移

使用 Alembic 管理数据库迁移：

```bash
# 生成迁移文件
alembic revision --autogenerate -m "描述"

# 执行迁移
alembic upgrade head

# 回滚迁移
alembic downgrade -1
```

### 测试

项目包含单元测试和属性测试：

```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/test_models.py

# 运行属性测试
pytest tests/ -k "property"

# 生成覆盖率报告
pytest --cov=src --cov-report=html
```

## API 文档

启动应用后，访问以下地址查看 API 文档：

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 许可证

[MIT License](LICENSE)

## 贡献

欢迎提交 Issue 和 Pull Request！

## 支持

如有问题，请联系：
- 邮箱: support@superinsight.ai
- 文档: https://docs.superinsight.ai
