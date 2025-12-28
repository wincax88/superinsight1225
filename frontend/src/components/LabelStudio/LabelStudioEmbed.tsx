// Label Studio iframe embed component
import { useEffect, useRef, useState, useCallback } from 'react';
import { Card, Spin, Alert, Button, Space } from 'antd';
import { ReloadOutlined, ExpandOutlined, CompressOutlined } from '@ant-design/icons';

interface LabelStudioEmbedProps {
  projectId: string;
  taskId?: string;
  baseUrl?: string;
  token?: string;
  onAnnotationCreate?: (annotation: unknown) => void;
  onAnnotationUpdate?: (annotation: unknown) => void;
  onTaskComplete?: (taskId: string) => void;
  height?: number | string;
}

interface LabelStudioMessage {
  type: string;
  payload?: unknown;
}

export const LabelStudioEmbed: React.FC<LabelStudioEmbedProps> = ({
  projectId,
  taskId,
  baseUrl = '/label-studio',
  token,
  onAnnotationCreate,
  onAnnotationUpdate,
  onTaskComplete,
  height = 600,
}) => {
  const iframeRef = useRef<HTMLIFrameElement>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [fullscreen, setFullscreen] = useState(false);

  // Build Label Studio URL
  const getLabelStudioUrl = useCallback(() => {
    let url = `${baseUrl}/projects/${projectId}`;
    if (taskId) {
      url += `/data?task=${taskId}`;
    }
    if (token) {
      url += `${taskId ? '&' : '?'}token=${token}`;
    }
    return url;
  }, [baseUrl, projectId, taskId, token]);

  // Handle messages from Label Studio iframe
  useEffect(() => {
    const handleMessage = (event: MessageEvent) => {
      // Verify origin for security
      if (!event.origin.includes(window.location.hostname) && !event.origin.includes('label-studio')) {
        return;
      }

      try {
        const message: LabelStudioMessage = event.data;

        switch (message.type) {
          case 'labelStudio:annotationCreated':
            onAnnotationCreate?.(message.payload);
            break;
          case 'labelStudio:annotationUpdated':
            onAnnotationUpdate?.(message.payload);
            break;
          case 'labelStudio:taskCompleted':
            if (taskId) {
              onTaskComplete?.(taskId);
            }
            break;
          case 'labelStudio:ready':
            setLoading(false);
            break;
          case 'labelStudio:error':
            setError(String(message.payload));
            setLoading(false);
            break;
        }
      } catch {
        // Ignore non-JSON messages
      }
    };

    window.addEventListener('message', handleMessage);
    return () => window.removeEventListener('message', handleMessage);
  }, [onAnnotationCreate, onAnnotationUpdate, onTaskComplete, taskId]);

  // Send message to Label Studio iframe
  const sendMessage = useCallback((type: string, payload?: unknown) => {
    if (iframeRef.current?.contentWindow) {
      iframeRef.current.contentWindow.postMessage({ type, payload }, '*');
    }
  }, []);

  // Reload iframe
  const handleReload = useCallback(() => {
    setLoading(true);
    setError(null);
    if (iframeRef.current) {
      iframeRef.current.src = getLabelStudioUrl();
    }
  }, [getLabelStudioUrl]);

  // Toggle fullscreen
  const handleFullscreenToggle = useCallback(() => {
    setFullscreen((prev) => !prev);
  }, []);

  // Handle iframe load
  const handleIframeLoad = useCallback(() => {
    // Give Label Studio time to initialize
    setTimeout(() => {
      setLoading(false);
    }, 1000);
  }, []);

  // Handle iframe error
  const handleIframeError = useCallback(() => {
    setError('Failed to load Label Studio. Please check if the service is running.');
    setLoading(false);
  }, []);

  const containerStyle: React.CSSProperties = fullscreen
    ? {
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        zIndex: 1000,
        background: '#fff',
      }
    : {};

  const iframeHeight = fullscreen ? '100vh' : height;

  return (
    <Card
      style={containerStyle}
      styles={{ body: { padding: 0, height: iframeHeight } }}
      title="Label Studio"
      extra={
        <Space>
          <Button
            type="text"
            icon={<ReloadOutlined />}
            onClick={handleReload}
            title="Reload"
          />
          <Button
            type="text"
            icon={fullscreen ? <CompressOutlined /> : <ExpandOutlined />}
            onClick={handleFullscreenToggle}
            title={fullscreen ? 'Exit Fullscreen' : 'Fullscreen'}
          />
        </Space>
      }
    >
      {loading && (
        <div
          style={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            zIndex: 10,
          }}
        >
          <Spin size="large" tip="Loading Label Studio..." />
        </div>
      )}

      {error && (
        <Alert
          type="error"
          message="Label Studio Error"
          description={error}
          showIcon
          action={
            <Button size="small" onClick={handleReload}>
              Retry
            </Button>
          }
          style={{ margin: 16 }}
        />
      )}

      <iframe
        ref={iframeRef}
        src={getLabelStudioUrl()}
        style={{
          width: '100%',
          height: '100%',
          border: 'none',
          display: error ? 'none' : 'block',
        }}
        onLoad={handleIframeLoad}
        onError={handleIframeError}
        title="Label Studio"
        allow="clipboard-read; clipboard-write"
      />
    </Card>
  );
};

// Export utility functions for external use
export const labelStudioUtils = {
  // Send annotation to Label Studio
  submitAnnotation: (iframe: HTMLIFrameElement, annotation: unknown) => {
    iframe.contentWindow?.postMessage(
      { type: 'labelStudio:submitAnnotation', payload: annotation },
      '*'
    );
  },

  // Navigate to specific task
  navigateToTask: (iframe: HTMLIFrameElement, taskId: string) => {
    iframe.contentWindow?.postMessage(
      { type: 'labelStudio:navigateToTask', payload: { taskId } },
      '*'
    );
  },

  // Skip current task
  skipTask: (iframe: HTMLIFrameElement) => {
    iframe.contentWindow?.postMessage({ type: 'labelStudio:skipTask' }, '*');
  },
};
