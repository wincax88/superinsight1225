#!/bin/bash

# Setup script for SuperInsight Platform Monitoring Stack
# This script sets up Prometheus, Grafana, and Alertmanager for monitoring

set -e

echo "🚀 Setting up SuperInsight Platform Monitoring Stack..."

# Check if Docker and Docker Compose are installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create necessary directories
echo "📁 Creating monitoring directories..."
mkdir -p grafana/provisioning/datasources
mkdir -p grafana/provisioning/dashboards
mkdir -p grafana/dashboards

# Set permissions for Grafana
echo "🔐 Setting up permissions..."
sudo chown -R 472:472 grafana/ || echo "⚠️  Could not set Grafana permissions (may need to run as root)"

# Create .env file for monitoring stack
echo "⚙️  Creating environment configuration..."
cat > .env << EOF
# SuperInsight Monitoring Configuration

# Grafana Configuration
GF_SECURITY_ADMIN_USER=admin
GF_SECURITY_ADMIN_PASSWORD=superinsight123
GF_USERS_ALLOW_SIGN_UP=false

# Prometheus Configuration
PROMETHEUS_RETENTION=200h

# Alertmanager Configuration
ALERTMANAGER_SMTP_HOST=localhost:587
ALERTMANAGER_SMTP_FROM=alerts@superinsight.local
ALERTMANAGER_SMTP_USER=alerts@superinsight.local
ALERTMANAGER_SMTP_PASS=password

# Slack Configuration (optional)
SLACK_WEBHOOK_URL=YOUR_SLACK_WEBHOOK_URL

# Email Configuration
ADMIN_EMAIL=admin@superinsight.local
OPS_EMAIL=ops@superinsight.local
BUSINESS_EMAIL=business@superinsight.local
EOF

# Start the monitoring stack
echo "🐳 Starting monitoring services..."
docker-compose -f docker-compose.monitoring.yml up -d

# Wait for services to start
echo "⏳ Waiting for services to start..."
sleep 30

# Check service health
echo "🏥 Checking service health..."

# Check Prometheus
if curl -s http://localhost:9090/-/healthy > /dev/null; then
    echo "✅ Prometheus is healthy"
else
    echo "❌ Prometheus is not responding"
fi

# Check Grafana
if curl -s http://localhost:3000/api/health > /dev/null; then
    echo "✅ Grafana is healthy"
else
    echo "❌ Grafana is not responding"
fi

# Check Alertmanager
if curl -s http://localhost:9093/-/healthy > /dev/null; then
    echo "✅ Alertmanager is healthy"
else
    echo "❌ Alertmanager is not responding"
fi

echo ""
echo "🎉 Monitoring stack setup complete!"
echo ""
echo "📊 Access URLs:"
echo "   Grafana:      http://localhost:3000 (admin/superinsight123)"
echo "   Prometheus:   http://localhost:9090"
echo "   Alertmanager: http://localhost:9093"
echo ""
echo "📈 Default Dashboards:"
echo "   - SuperInsight Platform Overview"
echo "   - Business Metrics Dashboard"
echo ""
echo "🔔 Alert Configuration:"
echo "   - Edit alertmanager.yml to configure email/Slack notifications"
echo "   - Update .env file with your SMTP and Slack settings"
echo ""
echo "🔧 Next Steps:"
echo "   1. Configure your email/Slack settings in .env"
echo "   2. Restart Alertmanager: docker-compose -f docker-compose.monitoring.yml restart alertmanager"
echo "   3. Import additional dashboards in Grafana if needed"
echo "   4. Set up SSL certificates for production use"
echo ""
echo "📚 Documentation:"
echo "   - Prometheus: https://prometheus.io/docs/"
echo "   - Grafana: https://grafana.com/docs/"
echo "   - Alertmanager: https://prometheus.io/docs/alerting/latest/alertmanager/"