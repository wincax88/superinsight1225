# Label Studio Django Settings Override for TCB Container
# This file extends/overrides Label Studio's default settings

import os

# Database configuration for embedded PostgreSQL
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': os.getenv('POSTGRE_NAME', 'label_studio'),
        'USER': os.getenv('POSTGRE_USER', 'superinsight'),
        'PASSWORD': os.getenv('POSTGRE_PASSWORD', 'superinsight_secret'),
        'HOST': os.getenv('POSTGRE_HOST', 'localhost'),
        'PORT': os.getenv('POSTGRE_PORT', '5432'),
        'CONN_MAX_AGE': 600,
        'OPTIONS': {
            'connect_timeout': 10,
        }
    }
}

# Redis configuration for caching
CACHES = {
    'default': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': os.getenv('REDIS_URL', 'redis://localhost:6379/1'),
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
            'CONNECTION_POOL_KWARGS': {
                'max_connections': 50,
                'retry_on_timeout': True,
            },
            'SOCKET_CONNECT_TIMEOUT': 5,
            'SOCKET_TIMEOUT': 5,
        }
    }
}

# Session configuration
SESSION_ENGINE = 'django.contrib.sessions.backends.cache'
SESSION_CACHE_ALIAS = 'default'

# Static and media files
STATIC_URL = '/static/'
MEDIA_URL = '/data/'
MEDIA_ROOT = os.getenv('LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT', '/app/label-studio-data')

# Local file serving
LOCAL_FILES_SERVING_ENABLED = os.getenv('LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED', 'true').lower() == 'true'
LOCAL_FILES_DOCUMENT_ROOT = os.getenv('LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT', '/app/label-studio-data')

# Security settings
ALLOWED_HOSTS = ['*']
CSRF_TRUSTED_ORIGINS = os.getenv('CSRF_TRUSTED_ORIGINS', '').split(',') if os.getenv('CSRF_TRUSTED_ORIGINS') else []

# Debug mode (should be False in production)
DEBUG = os.getenv('DEBUG', 'false').lower() == 'true'

# Logging configuration
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '[{asctime}] {levelname} {name} {message}',
            'style': '{',
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'verbose',
        },
    },
    'root': {
        'handlers': ['console'],
        'level': os.getenv('LOG_LEVEL', 'INFO'),
    },
    'loggers': {
        'django': {
            'handlers': ['console'],
            'level': os.getenv('DJANGO_LOG_LEVEL', 'INFO'),
            'propagate': False,
        },
        'label_studio': {
            'handlers': ['console'],
            'level': os.getenv('LOG_LEVEL', 'INFO'),
            'propagate': False,
        },
    },
}

# Performance settings
DATA_UPLOAD_MAX_MEMORY_SIZE = 250 * 1024 * 1024  # 250MB
FILE_UPLOAD_MAX_MEMORY_SIZE = 250 * 1024 * 1024  # 250MB

# Celery configuration (if using task queue)
CELERY_BROKER_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
CELERY_RESULT_BACKEND = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
