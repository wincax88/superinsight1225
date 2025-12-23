#!/bin/bash

# SuperInsight 私有化部署脚本

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
ENV_FILE="$PROJECT_ROOT/.env"
COMPOSE_FILE="$PROJECT_ROOT/docker-compose.prod.yml"
DATA_DIR="$PROJECT_ROOT/data"
BACKUP_DIR="$PROJECT_ROOT/backups"

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
SuperInsight 私有化部署脚本

用法: $0 [选项] [命令]

命令:
    install     初始化安装
    start       启动服务
    stop        停止服务
    restart     重启服务
    update      更新服务
    backup      备份数据
    restore     恢复数据
    logs        查看日志
    status      查看状态
    cleanup     清理资源

选项:
    -h, --help          显示帮助信息
    -v, --verbose       详细输出
    -f, --force         强制执行
    --env-file FILE     指定环境变量文件
    --profile PROFILE   指定 Docker Compose profile

示例:
    $0 install                    # 初始化安装
    $0 start --profile monitoring # 启动服务并启用监控
    $0 backup                     # 备份数据
    $0 logs api                   # 查看 API 服务日志
EOF
}

# 检查系统要求
check_requirements() {
    log_info "检查系统要求..."
    
    # 检查 Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker 未安装，请先安装 Docker"
        exit 1
    fi
    
    # 检查 Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose 未安装，请先安装 Docker Compose"
        exit 1
    fi
    
    # 检查磁盘空间（至少需要 10GB）
    available_space=$(df "$PROJECT_ROOT" | awk 'NR==2 {print $4}')
    required_space=$((10 * 1024 * 1024))  # 10GB in KB
    
    if [ "$available_space" -lt "$required_space" ]; then
        log_warn "磁盘空间不足，建议至少有 10GB 可用空间"
    fi
    
    # 检查内存（至少需要 4GB）
    total_memory=$(free -m | awk 'NR==2{print $2}')
    if [ "$total_memory" -lt 4096 ]; then
        log_warn "内存不足，建议至少有 4GB 内存"
    fi
    
    log_info "系统要求检查完成"
}

# 初始化环境
init_environment() {
    log_info "初始化环境..."
    
    # 创建必要的目录
    mkdir -p "$DATA_DIR"/{postgres,redis,label-studio,label-studio-media,uploads,exports,ollama}
    mkdir -p "$BACKUP_DIR"
    mkdir -p "$PROJECT_ROOT/logs"/{api,worker,nginx,postgres,redis,label-studio,ollama}
    
    # 设置目录权限
    chmod 755 "$DATA_DIR"
    chmod 755 "$BACKUP_DIR"
    chmod 755 "$PROJECT_ROOT/logs"
    
    # 复制环境变量文件
    if [ ! -f "$ENV_FILE" ]; then
        cp "$SCRIPT_DIR/.env.example" "$ENV_FILE"
        log_warn "已创建环境变量文件 $ENV_FILE，请编辑并填入实际值"
        log_warn "特别注意设置安全的密码和密钥"
    fi
    
    # 生成随机密钥（如果未设置）
    if ! grep -q "SECRET_KEY=" "$ENV_FILE" || grep -q "your_secret_key_here" "$ENV_FILE"; then
        SECRET_KEY=$(openssl rand -base64 32)
        sed -i "s/your_secret_key_here_minimum_32_characters/$SECRET_KEY/" "$ENV_FILE"
        log_info "已生成随机 SECRET_KEY"
    fi
    
    if ! grep -q "JWT_SECRET_KEY=" "$ENV_FILE" || grep -q "your_jwt_secret_key_here" "$ENV_FILE"; then
        JWT_SECRET_KEY=$(openssl rand -base64 32)
        sed -i "s/your_jwt_secret_key_here/$JWT_SECRET_KEY/" "$ENV_FILE"
        log_info "已生成随机 JWT_SECRET_KEY"
    fi
    
    if ! grep -q "ENCRYPTION_KEY=" "$ENV_FILE" || grep -q "your_encryption_key_here" "$ENV_FILE"; then
        ENCRYPTION_KEY=$(openssl rand -base64 32)
        sed -i "s/your_encryption_key_here_32_bytes_base64/$ENCRYPTION_KEY/" "$ENV_FILE"
        log_info "已生成随机 ENCRYPTION_KEY"
    fi
    
    log_info "环境初始化完成"
}

# 构建镜像
build_images() {
    log_info "构建 Docker 镜像..."
    
    cd "$PROJECT_ROOT"
    
    # 构建 API 镜像
    docker build -t superinsight-api:latest -f deploy/private/Dockerfile.api .
    
    # 构建 Worker 镜像
    docker build -t superinsight-worker:latest -f deploy/private/Dockerfile.worker .
    
    log_info "镜像构建完成"
}

# 启动服务
start_services() {
    log_info "启动 SuperInsight 服务..."
    
    cd "$PROJECT_ROOT"
    
    # 加载环境变量
    set -a
    source "$ENV_FILE"
    set +a
    
    # 启动服务
    if [ -n "$PROFILE" ]; then
        docker-compose -f "$COMPOSE_FILE" --profile "$PROFILE" up -d
    else
        docker-compose -f "$COMPOSE_FILE" up -d
    fi
    
    log_info "等待服务启动..."
    sleep 30
    
    # 检查服务状态
    check_services_health
    
    log_info "SuperInsight 服务启动完成"
}

# 停止服务
stop_services() {
    log_info "停止 SuperInsight 服务..."
    
    cd "$PROJECT_ROOT"
    docker-compose -f "$COMPOSE_FILE" down
    
    log_info "服务已停止"
}

# 重启服务
restart_services() {
    log_info "重启 SuperInsight 服务..."
    stop_services
    start_services
}

# 更新服务
update_services() {
    log_info "更新 SuperInsight 服务..."
    
    # 备份数据
    backup_data
    
    # 拉取最新代码
    git pull origin main
    
    # 重新构建镜像
    build_images
    
    # 重启服务
    restart_services
    
    log_info "服务更新完成"
}

# 检查服务健康状态
check_services_health() {
    log_info "检查服务健康状态..."
    
    services=("postgres" "redis" "label-studio" "superinsight-api")
    
    for service in "${services[@]}"; do
        if docker-compose -f "$COMPOSE_FILE" ps "$service" | grep -q "Up (healthy)"; then
            log_info "✓ $service 服务健康"
        else
            log_warn "✗ $service 服务异常"
        fi
    done
}

# 查看服务状态
show_status() {
    log_info "SuperInsight 服务状态:"
    
    cd "$PROJECT_ROOT"
    docker-compose -f "$COMPOSE_FILE" ps
    
    echo
    log_info "磁盘使用情况:"
    du -sh "$DATA_DIR"/* 2>/dev/null || true
    
    echo
    log_info "内存使用情况:"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"
}

# 查看日志
show_logs() {
    local service="$1"
    
    cd "$PROJECT_ROOT"
    
    if [ -n "$service" ]; then
        docker-compose -f "$COMPOSE_FILE" logs -f "$service"
    else
        docker-compose -f "$COMPOSE_FILE" logs -f
    fi
}

# 备份数据
backup_data() {
    log_info "备份数据..."
    
    local backup_name="superinsight-backup-$(date +%Y%m%d-%H%M%S)"
    local backup_path="$BACKUP_DIR/$backup_name"
    
    mkdir -p "$backup_path"
    
    # 备份数据库
    docker-compose -f "$COMPOSE_FILE" exec -T postgres pg_dump -U superinsight superinsight > "$backup_path/database.sql"
    
    # 备份文件数据
    tar -czf "$backup_path/data.tar.gz" -C "$DATA_DIR" .
    
    # 备份配置文件
    cp "$ENV_FILE" "$backup_path/"
    
    log_info "备份完成: $backup_path"
}

# 恢复数据
restore_data() {
    local backup_path="$1"
    
    if [ ! -d "$backup_path" ]; then
        log_error "备份目录不存在: $backup_path"
        exit 1
    fi
    
    log_warn "恢复数据将覆盖现有数据，确认继续？(y/N)"
    read -r confirm
    if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
        log_info "取消恢复操作"
        return
    fi
    
    log_info "恢复数据..."
    
    # 停止服务
    stop_services
    
    # 恢复数据库
    if [ -f "$backup_path/database.sql" ]; then
        docker-compose -f "$COMPOSE_FILE" up -d postgres
        sleep 10
        docker-compose -f "$COMPOSE_FILE" exec -T postgres psql -U superinsight -d superinsight < "$backup_path/database.sql"
    fi
    
    # 恢复文件数据
    if [ -f "$backup_path/data.tar.gz" ]; then
        tar -xzf "$backup_path/data.tar.gz" -C "$DATA_DIR"
    fi
    
    # 启动服务
    start_services
    
    log_info "数据恢复完成"
}

# 清理资源
cleanup() {
    log_warn "清理将删除所有容器、镜像和数据，确认继续？(y/N)"
    read -r confirm
    if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
        log_info "取消清理操作"
        return
    fi
    
    log_info "清理资源..."
    
    # 停止并删除容器
    docker-compose -f "$COMPOSE_FILE" down -v --rmi all
    
    # 删除数据目录
    rm -rf "$DATA_DIR"
    
    # 删除日志
    rm -rf "$PROJECT_ROOT/logs"
    
    log_info "清理完成"
}

# 主函数
main() {
    local command=""
    local verbose=false
    local force=false
    
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
            -f|--force)
                force=true
                shift
                ;;
            --env-file)
                ENV_FILE="$2"
                shift 2
                ;;
            --profile)
                PROFILE="$2"
                shift 2
                ;;
            install|start|stop|restart|update|backup|restore|logs|status|cleanup)
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
        install)
            check_requirements
            init_environment
            build_images
            start_services
            ;;
        start)
            start_services
            ;;
        stop)
            stop_services
            ;;
        restart)
            restart_services
            ;;
        update)
            update_services
            ;;
        backup)
            backup_data
            ;;
        restore)
            restore_data "$1"
            ;;
        logs)
            show_logs "$1"
            ;;
        status)
            show_status
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