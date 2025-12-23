# SuperInsight 私有化部署指南

## 概述

SuperInsight 支持完全私有化部署，确保数据不离开客户环境。本指南将帮助您在自己的服务器上部署 SuperInsight 平台。

## 系统要求

### 硬件要求

- **CPU**: 4 核心或以上
- **内存**: 8GB RAM 或以上（推荐 16GB）
- **存储**: 100GB 可用磁盘空间或以上
- **网络**: 稳定的网络连接

### 软件要求

- **操作系统**: Linux (Ubuntu 20.04+, CentOS 8+, RHEL 8+)
- **Docker**: 20.10+ 
- **Docker Compose**: 2.0+
- **Git**: 2.0+

## 快速开始

### 1. 克隆代码库

```bash
git clone https://github.com/your-org/superinsight-platform.git
cd superinsight-platform
```

### 2. 一键安装

```bash
# 运行安装脚本
sudo ./deploy/private/deploy.sh install

# 或者手动执行步骤
cp deploy/private/.env.example .env
# 编辑 .env 文件，填入实际配置
vim .env

# 启动服务
./deploy/private/deploy.sh start
```

### 3. 访问服务

- **SuperInsight 主界面**: http://localhost
- **Label Studio**: http://localhost/label-studio
- **API 文档**: http://localhost/api/docs
- **监控面板**: http://localhost:3000 (如果启用了监控)

## 详细部署步骤

### 步骤 1: 环境准备

#### 安装 Docker

```bash
# Ubuntu/Debian
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# CentOS/RHEL
sudo yum install -y docker-ce docker-ce-cli containerd.io
sudo systemctl start docker
sudo systemctl enable docker
```

#### 安装 Docker Compose

```bash
sudo curl -L "https://github.com/docker/compose/releases/download/v2.12.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

### 步骤 2: 配置环境变量

复制并编辑环境变量文件：

```bash
cp deploy/private/.env.example .env
```

**重要配置项**:

```bash
# 数据库密码（必须修改）
POSTGRES_PASSWORD=your_secure_password_here

# Label Studio 配置
LABEL_STUDIO_PASSWORD=your_label_studio_password
LABEL_STUDIO_TOKEN=your_api_token

# 安全密钥（必须修改）
SECRET_KEY=your_32_character_secret_key_here
JWT_SECRET_KEY=your_jwt_secret_key
ENCRYPTION_KEY=your_encryption_key

# AI 服务配置（可选）
HUNYUAN_API_KEY=your_hunyuan_api_key
HUNYUAN_SECRET_KEY=your_hunyuan_secret_key
```

### 步骤 3: 启动服务

```bash
# 基础服务
./deploy/private/deploy.sh start

# 包含监控服务
./deploy/private/deploy.sh start --profile monitoring

# 包含 Ollama 本地 AI
./deploy/private/deploy.sh start --profile ollama
```

### 步骤 4: 验证部署

```bash
# 检查服务状态
./deploy/private/deploy.sh status

# 查看服务日志
./deploy/private/deploy.sh logs

# 健康检查
curl http://localhost/health
```

## 配置说明

### 数据持久化

所有数据默认存储在 `./data` 目录下：

```
data/
├── postgres/          # 数据库文件
├── redis/            # Redis 数据
├── label-studio/     # Label Studio 数据
├── uploads/          # 上传文件
├── exports/          # 导出文件
└── ollama/           # Ollama 模型文件
```

### 网络配置

服务使用两个网络：

- `superinsight-internal`: 内部服务通信
- `superinsight-external`: 外部访问

### SSL/HTTPS 配置

1. 将 SSL 证书放置在 `deploy/private/ssl/` 目录：
   ```
   deploy/private/ssl/
   ├── cert.pem
   └── key.pem
   ```

2. 更新环境变量：
   ```bash
   SSL_CERT_PATH=./deploy/private/ssl/cert.pem
   SSL_KEY_PATH=./deploy/private/ssl/key.pem
   ```

3. 重启服务：
   ```bash
   ./deploy/private/deploy.sh restart
   ```

## 运维管理

### 备份数据

```bash
# 创建备份
./deploy/private/deploy.sh backup

# 备份文件位置
ls backups/
```

### 恢复数据

```bash
# 恢复指定备份
./deploy/private/deploy.sh restore backups/superinsight-backup-20231201-120000
```

### 更新服务

```bash
# 更新到最新版本
./deploy/private/deploy.sh update
```

### 查看日志

```bash
# 查看所有服务日志
./deploy/private/deploy.sh logs

# 查看特定服务日志
./deploy/private/deploy.sh logs api
./deploy/private/deploy.sh logs postgres
```

### 监控服务

启用监控服务后，可以通过以下方式访问：

- **Grafana**: http://localhost:3000
  - 用户名: admin
  - 密码: 在 .env 文件中设置的 GRAFANA_PASSWORD

- **Prometheus**: http://localhost:9090

## 安全配置

### 防火墙设置

```bash
# 只开放必要端口
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

### IP 白名单

在 `.env` 文件中配置允许访问的 IP：

```bash
ALLOWED_IPS=192.168.1.0/24,10.0.0.0/8
```

### 用户认证

SuperInsight 支持多种认证方式：

1. **内置认证**: 默认启用
2. **LDAP 认证**: 企业目录集成
3. **SSO 认证**: SAML/OAuth2/OIDC

## 故障排除

### 常见问题

1. **服务启动失败**
   ```bash
   # 检查日志
   docker-compose -f docker-compose.prod.yml logs
   
   # 检查端口占用
   netstat -tlnp | grep :80
   ```

2. **数据库连接失败**
   ```bash
   # 检查数据库状态
   docker-compose -f docker-compose.prod.yml exec postgres pg_isready
   
   # 重置数据库密码
   docker-compose -f docker-compose.prod.yml exec postgres psql -U postgres -c "ALTER USER superinsight PASSWORD 'new_password';"
   ```

3. **磁盘空间不足**
   ```bash
   # 清理 Docker 资源
   docker system prune -a
   
   # 清理日志文件
   find logs/ -name "*.log" -mtime +7 -delete
   ```

### 性能优化

1. **数据库优化**
   - 调整 PostgreSQL 配置
   - 定期执行 VACUUM 和 ANALYZE

2. **缓存优化**
   - 增加 Redis 内存限制
   - 配置适当的缓存策略

3. **资源限制**
   - 根据服务器配置调整容器资源限制
   - 启用自动扩缩容

## 技术支持

如需技术支持，请联系：

- **邮箱**: support@superinsight.com
- **文档**: https://docs.superinsight.com
- **GitHub**: https://github.com/your-org/superinsight-platform/issues

## 许可证

本软件遵循 [MIT License](LICENSE) 许可证。