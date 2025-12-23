# SuperInsight Platform Monitoring

This directory contains the complete monitoring stack for the SuperInsight Platform, including Prometheus for metrics collection, Grafana for visualization, and Alertmanager for alerting.

## Quick Start

1. **Setup the monitoring stack:**
   ```bash
   cd deploy/monitoring
   ./setup-monitoring.sh
   ```

2. **Access the services:**
   - Grafana: http://localhost:3000 (admin/superinsight123)
   - Prometheus: http://localhost:9090
   - Alertmanager: http://localhost:9093

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  SuperInsight   │    │   Prometheus    │    │    Grafana      │
│   Platform      │───▶│   (Metrics)     │───▶│ (Visualization) │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │  Alertmanager   │
                       │   (Alerting)    │
                       └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │ Email/Slack/etc │
                       │ (Notifications) │
                       └─────────────────┘
```

## Components

### Prometheus
- **Port:** 9090
- **Purpose:** Metrics collection and storage
- **Configuration:** `prometheus.yml`
- **Alert Rules:** `alert_rules.yml`

### Grafana
- **Port:** 3000
- **Purpose:** Metrics visualization and dashboards
- **Default Login:** admin/superinsight123
- **Dashboards:** Pre-configured with SuperInsight-specific dashboards

### Alertmanager
- **Port:** 9093
- **Purpose:** Alert routing and notifications
- **Configuration:** `alertmanager.yml`

### Node Exporter
- **Port:** 9100
- **Purpose:** System-level metrics (CPU, memory, disk, network)

### PostgreSQL Exporter
- **Port:** 9187
- **Purpose:** Database metrics

## Metrics Collected

### System Metrics
- CPU usage percentage
- Memory usage percentage
- Disk usage percentage
- Network I/O statistics

### Application Metrics
- HTTP request rate and duration
- Database query performance
- Error rates and response codes

### Business Metrics
- Annotation efficiency (annotations per hour)
- Annotation quality scores
- User activity and engagement
- AI model performance and confidence
- Project progress and completion rates
- Quality issues and work orders
- Billing and cost metrics

## Dashboards

### SuperInsight Platform Overview
- System resource utilization
- HTTP request metrics
- Database performance
- Overall health status

### Business Metrics Dashboard
- Annotation efficiency trends
- Quality score monitoring
- User activity patterns
- AI model performance
- Project progress tracking
- Quality issue trends

## Alerting

### Alert Categories

1. **System Alerts**
   - High CPU/Memory/Disk usage
   - Service availability

2. **Application Alerts**
   - High error rates
   - Slow response times
   - Database issues

3. **Business Alerts**
   - Low annotation efficiency
   - Poor quality scores
   - No active users
   - AI model failures

### Alert Severity Levels

- **Critical:** Immediate attention required (service down, critical errors)
- **Warning:** Attention needed (performance degradation, resource usage)

### Notification Channels

- **Email:** Configured for different severity levels
- **Slack:** Real-time notifications (requires webhook configuration)
- **Webhook:** Custom integrations

## Configuration

### Environment Variables

Edit `.env` file to configure:

```bash
# Grafana
GF_SECURITY_ADMIN_USER=admin
GF_SECURITY_ADMIN_PASSWORD=superinsight123

# Email notifications
ALERTMANAGER_SMTP_HOST=localhost:587
ALERTMANAGER_SMTP_FROM=alerts@superinsight.local
ADMIN_EMAIL=admin@superinsight.local

# Slack notifications
SLACK_WEBHOOK_URL=YOUR_SLACK_WEBHOOK_URL
```

### Customizing Alerts

1. **Edit alert rules:** Modify `alert_rules.yml`
2. **Configure notifications:** Update `alertmanager.yml`
3. **Restart services:**
   ```bash
   docker-compose -f docker-compose.monitoring.yml restart
   ```

### Adding Custom Dashboards

1. Create dashboard JSON in `grafana/dashboards/`
2. Restart Grafana:
   ```bash
   docker-compose -f docker-compose.monitoring.yml restart grafana
   ```

## Maintenance

### Backup

Important data to backup:
- Grafana dashboards: `grafana_data` volume
- Prometheus data: `prometheus_data` volume
- Alert configurations: `alertmanager.yml`, `alert_rules.yml`

### Updates

```bash
# Pull latest images
docker-compose -f docker-compose.monitoring.yml pull

# Restart services
docker-compose -f docker-compose.monitoring.yml up -d
```

### Troubleshooting

#### Prometheus not scraping metrics
1. Check SuperInsight Platform `/metrics` endpoint
2. Verify network connectivity
3. Check Prometheus logs:
   ```bash
   docker logs superinsight-prometheus
   ```

#### Grafana dashboards not loading
1. Check datasource configuration
2. Verify Prometheus connectivity
3. Check Grafana logs:
   ```bash
   docker logs superinsight-grafana
   ```

#### Alerts not firing
1. Verify alert rules syntax
2. Check Alertmanager configuration
3. Test notification channels

### Performance Tuning

#### Prometheus
- Adjust retention period in `prometheus.yml`
- Configure storage settings for large deployments
- Optimize scrape intervals based on needs

#### Grafana
- Configure caching for better performance
- Optimize dashboard queries
- Set appropriate refresh intervals

## Security

### Production Deployment

1. **Change default passwords**
2. **Enable HTTPS/TLS**
3. **Configure authentication** (LDAP, OAuth, etc.)
4. **Set up proper firewall rules**
5. **Use secrets management** for sensitive data

### Network Security

- Use internal networks for service communication
- Expose only necessary ports
- Implement proper access controls

## Integration

### SuperInsight Platform Integration

The monitoring stack automatically integrates with SuperInsight Platform through:

1. **Metrics endpoint:** `/metrics` (Prometheus format)
2. **Business metrics API:** `/api/business-metrics/*`
3. **Health checks:** `/health`, `/health/ready`, `/health/live`

### Custom Metrics

Add custom metrics in your application:

```python
from src.system.prometheus_exporter import prometheus_exporter

# Track custom events
prometheus_exporter.track_http_request("GET", "/api/custom", 200, 0.5)
prometheus_exporter.track_ai_inference("custom-model", True, 2.3)
```

## Support

For issues and questions:

1. Check logs: `docker-compose -f docker-compose.monitoring.yml logs`
2. Verify configuration files
3. Consult official documentation:
   - [Prometheus Documentation](https://prometheus.io/docs/)
   - [Grafana Documentation](https://grafana.com/docs/)
   - [Alertmanager Documentation](https://prometheus.io/docs/alerting/latest/alertmanager/)

## License

This monitoring configuration is part of the SuperInsight Platform and follows the same licensing terms.