#!/bin/bash

# SuperInsight 腾讯云 TCB 部署脚本

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查必要的环境变量
check_env_vars() {
    log_info "检查环境变量..."
    
    required_vars=(
        "TCB_ENV_ID"
        "TCB_SECRET_ID" 
        "TCB_SECRET_KEY"
        "DATABASE_URL"
        "HUNYUAN_API_KEY"
        "HUNYUAN_SECRET_KEY"
    )
    
    for var in "${required_vars[@]}"; do
        if [ -z "${!var}" ]; then
            log_error "环境变量 $var 未设置"
            exit 1
        fi
    done
    
    log_info "环境变量检查通过"
}

# 安装依赖
install_dependencies() {
    log_info "安装部署依赖..."
    
    # 检查是否安装了 cloudbase CLI
    if ! command -v tcb &> /dev/null; then
        log_info "安装 CloudBase CLI..."
        npm install -g @cloudbase/cli
    fi
    
    # 检查是否安装了 Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker 未安装，请先安装 Docker"
        exit 1
    fi
    
    log_info "依赖安装完成"
}

# 构建 Docker 镜像
build_docker_image() {
    log_info "构建 Docker 镜像..."
    
    # 构建 API 服务镜像
    docker build -t superinsight-api:latest -f deploy/tcb/Dockerfile.api .
    
    # 构建 Worker 服务镜像
    docker build -t superinsight-worker:latest -f deploy/tcb/Dockerfile.worker .
    
    log_info "Docker 镜像构建完成"
}

# 推送镜像到腾讯云容器镜像服务
push_images() {
    log_info "推送镜像到腾讯云容器镜像服务..."
    
    # 登录到腾讯云容器镜像服务
    docker login ccr.ccs.tencentyun.com -u ${TCB_SECRET_ID} -p ${TCB_SECRET_KEY}
    
    # 标记镜像
    docker tag superinsight-api:latest ccr.ccs.tencentyun.com/superinsight/api:latest
    docker tag superinsight-worker:latest ccr.ccs.tencentyun.com/superinsight/worker:latest
    
    # 推送镜像
    docker push ccr.ccs.tencentyun.com/superinsight/api:latest
    docker push ccr.ccs.tencentyun.com/superinsight/worker:latest
    
    log_info "镜像推送完成"
}

# 部署数据库迁移
deploy_database() {
    log_info "执行数据库迁移..."
    
    # 运行数据库迁移
    python -m alembic upgrade head
    
    log_info "数据库迁移完成"
}

# 部署云函数
deploy_functions() {
    log_info "部署云函数..."
    
    # 登录 CloudBase
    tcb login --key-file deploy/tcb/tcb-key.json
    
    # 部署函数
    tcb functions:deploy data-extractor --code-secret ${TCB_SECRET_ID}
    tcb functions:deploy ai-annotator --code-secret ${TCB_SECRET_ID}
    tcb functions:deploy quality-manager --code-secret ${TCB_SECRET_ID}
    
    log_info "云函数部署完成"
}

# 部署云托管服务
deploy_cloudbase() {
    log_info "部署云托管服务..."
    
    # 使用 cloudbaserc.json 配置部署
    tcb framework:deploy --envId ${TCB_ENV_ID}
    
    log_info "云托管服务部署完成"
}

# 配置域名和 SSL
configure_domain() {
    log_info "配置自定义域名..."
    
    if [ -n "$CUSTOM_DOMAIN" ]; then
        tcb hosting:domain:add ${CUSTOM_DOMAIN} --envId ${TCB_ENV_ID}
        log_info "自定义域名配置完成: ${CUSTOM_DOMAIN}"
    else
        log_warn "未设置自定义域名，使用默认域名"
    fi
}

# 健康检查
health_check() {
    log_info "执行健康检查..."
    
    # 获取部署的服务 URL
    SERVICE_URL=$(tcb hosting:detail --envId ${TCB_ENV_ID} | grep -o 'https://[^"]*')
    
    if [ -n "$SERVICE_URL" ]; then
        # 检查 API 健康状态
        if curl -f "${SERVICE_URL}/health" > /dev/null 2>&1; then
            log_info "健康检查通过: ${SERVICE_URL}"
        else
            log_error "健康检查失败: ${SERVICE_URL}"
            exit 1
        fi
    else
        log_warn "无法获取服务 URL，跳过健康检查"
    fi
}

# 清理临时文件
cleanup() {
    log_info "清理临时文件..."
    
    # 清理 Docker 镜像
    docker rmi superinsight-api:latest superinsight-worker:latest || true
    
    log_info "清理完成"
}

# 主部署流程
main() {
    log_info "开始 SuperInsight TCB 部署..."
    
    # 检查环境
    check_env_vars
    install_dependencies
    
    # 构建和推送
    build_docker_image
    push_images
    
    # 部署服务
    deploy_database
    deploy_functions
    deploy_cloudbase
    
    # 配置和检查
    configure_domain
    health_check
    
    # 清理
    cleanup
    
    log_info "SuperInsight TCB 部署完成！"
    
    if [ -n "$SERVICE_URL" ]; then
        log_info "服务访问地址: ${SERVICE_URL}"
    fi
}

# 错误处理
trap 'log_error "部署过程中发生错误，退出码: $?"' ERR

# 执行主流程
main "$@"