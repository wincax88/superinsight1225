# SuperInsight AI 数据治理与标注平台 - 需求文档

## 介绍

SuperInsight 是一款专为 AI 时代设计的企业级语料治理与智能标注平台，深度借鉴龙石数据成熟的"理采存管用"方法论，同时针对大模型（LLM）和生成式 AI（GenAI）应用场景进行全面升级。平台以安全只读提取 + 人机协同标注 + 业务规则智能注入为核心，帮助企业快速构建高质量、AI 友好的语料数据集。

## 术语表

- **SuperInsight_Platform**: 整个 AI 数据治理与标注平台系统
- **Label_Studio**: 开源标注工具，作为平台的核心标注引擎
- **PostgreSQL_Database**: 统一数据存储系统，使用 JSONB 格式
- **TCB_Cloud**: 腾讯云 TCB 云托管服务
- **Data_Extractor**: 安全只读数据提取模块
- **AI_Annotator**: AI 预标注服务
- **Quality_Manager**: 语义质量评估与管理模块
- **Billing_System**: 计费结算系统
- **Security_Controller**: 安全控制与权限管理模块

## 需求

### 需求 1: 安全数据提取

**用户故事:** 作为数据管理员，我希望能够安全地从各种数据源提取语料数据，以便为 AI 标注做准备。

#### 验收标准

1. WHEN 连接到客户数据库时，THE Data_Extractor SHALL 使用只读权限进行连接
2. WHEN 提取结构化数据时，THE Data_Extractor SHALL 支持 MySQL、PostgreSQL、Oracle 等主流数据库
3. WHEN 提取非结构化数据时，THE Data_Extractor SHALL 支持 PDF、Word、Notion、网页等格式
4. WHEN 数据传输过程中，THE Data_Extractor SHALL 使用加密传输协议
5. WHEN 提取完成后，THE Data_Extractor SHALL 在 PostgreSQL_Database 中创建原始数据副本

### 需求 2: 语料存储与管理

**用户故事:** 作为系统架构师，我希望有一个统一的存储系统来管理所有语料数据，以便支持高效的查询和扩展。

#### 验收标准

1. THE PostgreSQL_Database SHALL 使用 JSONB 格式存储原始语料数据
2. THE PostgreSQL_Database SHALL 使用 JSONB 格式存储标注结果和标签
3. THE PostgreSQL_Database SHALL 使用 JSONB 格式存储优质增强数据
4. THE PostgreSQL_Database SHALL 创建 GIN 索引以支持高效查询
5. WHEN 存储元数据时，THE PostgreSQL_Database SHALL 记录数据血缘和审计日志

### 需求 3: 人机协同标注

**用户故事:** 作为标注专家，我希望能够与 AI 协同进行数据标注，以便提高标注效率和质量。

#### 验收标准

1. THE Label_Studio SHALL 提供直观的标注界面
2. WHEN 开始标注任务时，THE AI_Annotator SHALL 提供预标注结果
3. WHEN 人工标注时，THE Label_Studio SHALL 支持业务专家、技术专家和外包人员协作
4. WHEN 标注完成时，THE Label_Studio SHALL 通过 Webhook 触发质量检查
5. THE Label_Studio SHALL 支持实时标注进度跟踪

### 需求 4: 业务规则与质量治理

**用户故事:** 作为质量管理员，我希望能够定义和执行业务规则，以便确保标注数据的质量。

#### 验收标准

1. THE Quality_Manager SHALL 提供内置的质量规则模板
2. WHEN 质量问题被发现时，THE Quality_Manager SHALL 创建质量工单
3. WHEN 质量工单被创建时，THE Quality_Manager SHALL 自动派发给相关专家
4. THE Quality_Manager SHALL 使用 Ragas 框架进行语义质量评估
5. WHEN 质量问题修复后，THE Quality_Manager SHALL 支持源头数据修复

### 需求 5: 数据增强与重构

**用户故事:** 作为 AI 工程师，我希望能够增强和重构语料数据，以便提高 AI 模型的训练效果。

#### 验收标准

1. THE SuperInsight_Platform SHALL 支持填充优质样本数据
2. WHEN 进行数据增强时，THE SuperInsight_Platform SHALL 放大正向激励数据占比
3. THE SuperInsight_Platform SHALL 提供数据重构接口
4. WHEN 数据增强完成时，THE SuperInsight_Platform SHALL 更新数据质量评分
5. THE SuperInsight_Platform SHALL 支持批量数据增强操作

### 需求 6: AI 友好数据集输出

**用户故事:** 作为 AI 开发者，我希望能够导出标准格式的数据集，以便用于 RAG、Agent 等 AI 应用。

#### 验收标准

1. THE SuperInsight_Platform SHALL 支持导出 JSON 格式数据集
2. THE SuperInsight_Platform SHALL 支持导出 CSV 格式数据集
3. THE SuperInsight_Platform SHALL 支持导出 COCO 格式数据集
4. THE SuperInsight_Platform SHALL 提供 RAG 测试接口
5. THE SuperInsight_Platform SHALL 提供 Agent 测试接口

### 需求 7: 计费结算系统

**用户故事:** 作为业务管理员，我希望能够跟踪标注工时和成本，以便进行准确的计费结算。

#### 验收标准

1. THE Billing_System SHALL 统计标注工时
2. THE Billing_System SHALL 统计标注条数
3. WHEN 月度结束时，THE Billing_System SHALL 生成月度账单
4. THE Billing_System SHALL 支持多租户隔离计费
5. THE Billing_System SHALL 提供计费报表和分析

### 需求 8: 安全合规管理

**用户故事:** 作为安全管理员，我希望平台能够满足企业级安全合规要求，以便保护敏感数据。

#### 验收标准

1. THE Security_Controller SHALL 提供项目级别的数据隔离
2. WHEN 处理敏感数据时，THE Security_Controller SHALL 执行数据脱敏
3. THE Security_Controller SHALL 记录所有操作的审计日志
4. THE Security_Controller SHALL 支持 IP 白名单访问控制
5. WHEN 用户访问系统时，THE Security_Controller SHALL 验证用户权限

### 需求 9: 多部署方式支持

**用户故事:** 作为运维工程师，我希望平台能够支持多种部署方式，以便满足不同客户的需求。

#### 验收标准

1. THE SuperInsight_Platform SHALL 支持腾讯云 TCB 云托管部署
2. THE SuperInsight_Platform SHALL 支持 Docker Compose 私有化部署
3. THE SuperInsight_Platform SHALL 支持混合云部署模式
4. WHEN 使用 TCB 部署时，THE SuperInsight_Platform SHALL 支持自动扩缩容
5. WHEN 使用私有化部署时，THE SuperInsight_Platform SHALL 确保数据不出客户环境

### 需求 10: AI 预标注集成

**用户故事:** 作为标注管理员，我希望集成 AI 预标注功能，以便提高标注效率。

#### 验收标准

1. THE AI_Annotator SHALL 集成 Ollama 本地模型
2. THE AI_Annotator SHALL 集成 HuggingFace 模型
3. WHEN 启动标注任务时，THE AI_Annotator SHALL 自动生成预标注结果
4. THE AI_Annotator SHALL 支持自定义模型配置
5. WHEN AI 预标注完成时，THE AI_Annotator SHALL 提供置信度评分