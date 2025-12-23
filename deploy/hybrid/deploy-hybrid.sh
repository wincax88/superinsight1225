#!/bin/bash

# SuperInsight 混合云部署脚本

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_FILE="$SCRIPT_DIR/hybrid-config.yaml"
ENV_FILE="$PROJECT_ROOT/.env.hybrid"

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

log_debug() {
    echo -e "${BLUE}[DEBUG]${NC} $1"
}

# 显示帮助信息
show_help() {
    cat << EOF
SuperInsight 混合云部署脚本

用法: $0 [选项] [命令]

命令:
    init        初始化混合云环境
    deploy      部署混合云架构
    sync        执行数据同步
    status      查看部署状态
    test        测试连接
    cleanup     清理资源

选项:
    -h, --help          显示帮助信息
    -v, --verbose       详细输出
    -c, --config FILE   指定配置文件
    --local-only        仅部署本地环境
    --cloud-only        仅部署云端环境

示例:
    $0 init                     # 初始化混合云环境
    $0 deploy                   # 部署完整混合云架构
    $0 deploy --local-only      # 仅部署本地环境
    $0 sync                     # 执行数据同步
    $0 test                     # 测试连接
EOF
}

# 检查依赖
check_dependencies() {
    log_info "检查依赖..."
    
    # 检查必要的工具
    local tools=("docker" "docker-compose" "kubectl" "python3" "openssl")
    
    for tool in "${tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            log_error "$tool 未安装"
            exit 1
        fi
    done
    
    # 检查 Python 依赖
    if ! python3 -c "import yaml, cryptography, requests" 2>/dev/null; then
        log_warn "Python 依赖不完整，正在安装..."
        pip3 install pyyaml cryptography requests
    fi
    
    log_info "依赖检查完成"
}

# 生成证书
generate_certificates() {
    log_info "生成 SSL 证书..."
    
    local cert_dir="$PROJECT_ROOT/deploy/hybrid/certs"
    mkdir -p "$cert_dir"
    
    # 生成 CA 证书
    if [ ! -f "$cert_dir/ca.key" ]; then
        openssl genrsa -out "$cert_dir/ca.key" 4096
        openssl req -new -x509 -days 365 -key "$cert_dir/ca.key" -out "$cert_dir/ca.crt" \
            -subj "/C=CN/ST=Shanghai/L=Shanghai/O=SuperInsight/CN=SuperInsight-CA"
        log_info "生成 CA 证书"
    fi
    
    # 生成服务器证书
    if [ ! -f "$cert_dir/server.key" ]; then
        openssl genrsa -out "$cert_dir/server.key" 2048
        openssl req -new -key "$cert_dir/server.key" -out "$cert_dir/server.csr" \
            -subj "/C=CN/ST=Shanghai/L=Shanghai/O=SuperInsight/CN=superinsight-hybrid"
        openssl x509 -req -days 365 -in "$cert_dir/server.csr" -CA "$cert_dir/ca.crt" \
            -CAkey "$cert_dir/ca.key" -CAcreateserial -out "$cert_dir/server.crt"
        rm "$cert_dir/server.csr"
        log_info "生成服务器证书"
    fi
    
    # 生成客户端证书
    if [ ! -f "$cert_dir/client.key" ]; then
        openssl genrsa -out "$cert_dir/client.key" 2048
        openssl req -new -key "$cert_dir/client.key" -out "$cert_dir/client.csr" \
            -subj "/C=CN/ST=Shanghai/L=Shanghai/O=SuperInsight/CN=superinsight-client"
        openssl x509 -req -days 365 -in "$cert_dir/client.csr" -CA "$cert_dir/ca.crt" \
            -CAkey "$cert_dir/ca.key" -CAcreateserial -out "$cert_dir/client.crt"
        rm "$cert_dir/client.csr"
        log_info "生成客户端证书"
    fi
    
    # 设置权限
    chmod 600 "$cert_dir"/*.key
    chmod 644 "$cert_dir"/*.crt
    
    log_info "SSL 证书生成完成"
}

# 初始化环境
init_environment() {
    log_info "初始化混合云环境..."
    
    # 检查依赖
    check_dependencies
    
    # 生成证书
    generate_certificates
    
    # 创建环境变量文件
    if [ ! -f "$ENV_FILE" ]; then
        cat > "$ENV_FILE" << EOF
# SuperInsight 混合云环境变量

# 基础配置
ENVIRONMENT=hybrid
HYBRID_MODE=local_primary

# 本地环境配置
LOCAL_ENDPOINT=http://localhost:8000
LOCAL_DATABASE_URL=postgresql://superinsight:password@localhost:5432/superinsight

# 云端环境配置
CLOUD_ENDPOINT=https://your-cloud-endpoint.com
CLOUD_REGION=ap-shanghai
CLOUD_ACCESS_KEY=your_cloud_access_key
CLOUD_SECRET_KEY=your_cloud_secret_key

# 混合云配置
HYBRID_CONFIG_PATH=$CONFIG_FILE
PROXY_KEY=$(openssl rand -base64 32)
PROXY_ENCRYPTION=true
SYNC_ENABLED=true
SYNC_INTERVAL=300

# 安全配置
HYBRID_PRIVATE_KEY_PATH=$PROJECT_ROOT/deploy/hybrid/certs/client.key
HYBRID_PUBLIC_KEY_PATH=$PROJECT_ROOT/deploy/hybrid/certs/client.crt
HYBRID_CA_CERT_PATH=$PROJECT_ROOT/deploy/hybrid/certs/ca.crt
HYBRID_CLIENT_CERT_PATH=$PROJECT_ROOT/deploy/hybrid/certs/client.crt
HYBRID_CLIENT_KEY_PATH=$PROJECT_ROOT/deploy/hybrid/certs/client.key

# JWT 配置
HYBRID_JWT_SECRET=$(openssl rand -base64 32)
HYBRID_JWT_EXPIRY=3600

# 监控配置
MONITORING_ENABLED=true
METRICS_ENDPOINT=http://localhost:9090
ALERT_WEBHOOK_URL=https://your-alert-webhook.com
EOF
        
        log_warn "已创建环境变量文件 $ENV_FILE，请编辑并填入实际值"
    fi
    
    # 创建必要的目录
    mkdir -p "$PROJECT_ROOT/data/hybrid"
    mkdir -p "$PROJECT_ROOT/logs/hybrid"
    mkdir -p "$PROJECT_ROOT/backups/hybrid"
    
    log_info "混合云环境初始化完成"
}

# 部署本地环境
deploy_local() {
    log_info "部署本地环境..."
    
    cd "$PROJECT_ROOT"
    
    # 加载环境变量
    set -a
    source "$ENV_FILE"
    set +a
    
    # 启动本地服务
    docker-compose -f docker-compose.prod.yml up -d
    
    # 等待服务启动
    log_info "等待本地服务启动..."
    sleep 30
    
    # 健康检查
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        log_info "本地环境部署成功"
    else
        log_error "本地环境部署失败"
        exit 1
    fi
}

# 部署云端环境
deploy_cloud() {
    log_info "部署云端环境..."
    
    # 加载环境变量
    set -a
    source "$ENV_FILE"
    set +a
    
    # 根据云服务提供商部署
    case "${CLOUD_PROVIDER:-tencent_cloud}" in
        "tencent_cloud")
            deploy_tencent_cloud
            ;;
        "aws")
            deploy_aws
            ;;
        "azure")
            deploy_azure
            ;;
        *)
            log_error "不支持的云服务提供商: ${CLOUD_PROVIDER}"
            exit 1
            ;;
    esac
}

# 部署腾讯云环境
deploy_tencent_cloud() {
    log_info "部署腾讯云环境..."
    
    # 检查腾讯云 CLI
    if ! command -v tcb &> /dev/null; then
        log_error "腾讯云 CLI 未安装，请先安装 @cloudbase/cli"
        exit 1
    fi
    
    # 部署云函数
    tcb functions:deploy hybrid-sync --envId "$TCB_ENV_ID"
    tcb functions:deploy hybrid-api --envId "$TCB_ENV_ID"
    
    # 部署云托管服务
    tcb framework:deploy --envId "$TCB_ENV_ID"
    
    log_info "腾讯云环境部署完成"
}

# 部署 AWS 环境
deploy_aws() {
    log_info "部署 AWS 环境..."
    
    # 检查 AWS CLI
    if ! command -v aws &> /dev/null; then
        log_error "AWS CLI 未安装"
        exit 1
    fi
    
    # TODO: 实现 AWS 部署逻辑
    log_warn "AWS 部署功能开发中..."
}

# 部署 Azure 环境
deploy_azure() {
    log_info "部署 Azure 环境..."
    
    # 检查 Azure CLI
    if ! command -v az &> /dev/null; then
        log_error "Azure CLI 未安装"
        exit 1
    fi
    
    # TODO: 实现 Azure 部署逻辑
    log_warn "Azure 部署功能开发中..."
}

# 执行数据同步
run_sync() {
    log_info "执行混合云数据同步..."
    
    # 加载环境变量
    set -a
    source "$ENV_FILE"
    set +a
    
    # 运行同步脚本
    python3 << EOF
import asyncio
import sys
import os
sys.path.append('$PROJECT_ROOT')

from src.hybrid.sync_manager import get_sync_manager

async def main():
    sync_manager = get_sync_manager()
    results = await sync_manager.run_full_sync()
    
    print("同步结果:")
    for result in results:
        status = "成功" if result.success else "失败"
        print(f"  {result.data_type}: {status} ({result.records_synced}/{result.records_processed})")
        if result.errors:
            for error in result.errors:
                print(f"    错误: {error}")

if __name__ == "__main__":
    asyncio.run(main())
EOF
    
    log_info "数据同步完成"
}

# 测试连接
test_connection() {
    log_info "测试混合云连接..."
    
    # 加载环境变量
    set -a
    source "$ENV_FILE"
    set +a
    
    # 测试本地连接
    if curl -f "$LOCAL_ENDPOINT/health" > /dev/null 2>&1; then
        log_info "✓ 本地环境连接正常"
    else
        log_error "✗ 本地环境连接失败"
    fi
    
    # 测试云端连接
    if curl -f "$CLOUD_ENDPOINT/health" > /dev/null 2>&1; then
        log_info "✓ 云端环境连接正常"
    else
        log_error "✗ 云端环境连接失败"
    fi
    
    # 测试安全通道
    python3 << EOF
import sys
sys.path.append('$PROJECT_ROOT')

from src.hybrid.secure_channel import get_secure_channel

try:
    channel = get_secure_channel()
    result = channel.test_connection('$CLOUD_ENDPOINT')
    
    if result['success']:
        print("✓ 安全通道连接正常")
        print(f"  响应时间: {result['response_time']:.2f}s")
        if 'ssl_version' in result:
            print(f"  SSL 版本: {result['ssl_version']}")
    else:
        print(f"✗ 安全通道连接失败: {result['error']}")
except Exception as e:
    print(f"✗ 安全通道测试异常: {e}")
EOF
}

# 查看状态
show_status() {
    log_info "混合云部署状态:"
    
    # 加载环境变量
    set -a
    source "$ENV_FILE"
    set +a
    
    echo
    echo "本地环境:"
    docker-compose -f "$PROJECT_ROOT/docker-compose.prod.yml" ps
    
    echo
    echo "同步状态:"
    python3 << EOF
import sys
sys.path.append('$PROJECT_ROOT')

from src.hybrid.sync_manager import get_sync_manager

try:
    sync_manager = get_sync_manager()
    status = sync_manager.get_sync_status()
    
    print(f"  状态: {status['status']}")
    print(f"  最后同步: {status['last_sync']}")
    print(f"  成功率: {status['success_rate']:.1f}%")
    print(f"  总同步数: {status['total_synced']}")
    
    if status['recent_errors']:
        print("  最近错误:")
        for error in status['recent_errors']:
            print(f"    - {error}")
except Exception as e:
    print(f"  获取同步状态失败: {e}")
EOF
}

# 清理资源
cleanup() {
    log_warn "清理将删除所有混合云资源，确认继续？(y/N)"
    read -r confirm
    if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
        log_info "取消清理操作"
        return
    fi
    
    log_info "清理混合云资源..."
    
    # 停止本地服务
    cd "$PROJECT_ROOT"
    docker-compose -f docker-compose.prod.yml down -v
    
    # 清理数据目录
    rm -rf "$PROJECT_ROOT/data/hybrid"
    rm -rf "$PROJECT_ROOT/logs/hybrid"
    
    # 清理证书
    rm -rf "$PROJECT_ROOT/deploy/hybrid/certs"
    
    log_info "清理完成"
}

# 主函数
main() {
    local command=""
    local verbose=false
    local local_only=false
    local cloud_only=false
    
    # 解析参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -v|--verbose)
                verbose=true
                set -x
                shift
                ;;
            -c|--config)
                CONFIG_FILE="$2"
                shift 2
                ;;
            --local-only)
                local_only=true
                shift
                ;;
            --cloud-only)
                cloud_only=true
                shift
                ;;
            init|deploy|sync|status|test|cleanup)
                command="$1"
                shift
                ;;
            *)
                log_error "未知参数: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 检查命令
    if [ -z "$command" ]; then
        log_error "请指定命令"
        show_help
        exit 1
    fi
    
    # 执行命令
    case $command in
        init)
            init_environment
            ;;
        deploy)
            if [ "$local_only" = true ]; then
                deploy_local
            elif [ "$cloud_only" = true ]; then
                deploy_cloud
            else
                deploy_local
                deploy_cloud
            fi
            ;;
        sync)
            run_sync
            ;;
        status)
            show_status
            ;;
        test)
            test_connection
            ;;
        cleanup)
            cleanup
            ;;
    esac
}

# 错误处理
trap 'log_error "脚本执行失败，退出码: $?"' ERR

# 执行主函数
main "$@"