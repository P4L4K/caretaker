import React, { useEffect, useRef, useState } from 'react';
import { Button, Card, Progress, Alert, Tabs, TabsProps, Space, Typography } from 'antd';
import { AudioOutlined, VideoCameraOutlined, SmileOutlined } from '@ant-design/icons';
import { Line } from '@ant-design/plots';

const { Title, Text } = Typography;

interface DetectionResult {
  type: 'audio' | 'fall' | 'emotion';
  data: any;
  timestamp: number;
}

const RealTimeMonitor: React.FC = () => {
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const [results, setResults] = useState<DetectionResult[]>([]);
  const [activeTab, setActiveTab] = useState<string>('1');
  const [audioLevel, setAudioLevel] = useState<number>(0);
  const [cpuUsage, setCpuUsage] = useState<number>(0);
  const [gpuUsage, setGpuUsage] = useState<number>(0);
  const [error, setError] = useState<string | null>(null);
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const animationFrameRef = useRef<number>();
  const audioAnalyserRef = useRef<AnalyserNode | null>(null);
  const dataArrayRef = useRef<Uint8Array | null>(null);
  
  // WebSocket connection
  useEffect(() => {
    connectWebSocket();
    
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      stopMediaStreams();
    };
  }, []);

  const connectWebSocket = () => {
    const token = localStorage.getItem('token');
    const ws = new WebSocket(`ws://localhost:8000/ws/unified?token=${token}`);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('Connected to WebSocket');
      setIsConnected(true);
      setError(null);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.error) {
          setError(data.error);
          return;
        }
        
        const newResults = data.results.map((result: any) => ({
          ...result,
          timestamp: Date.now(),
        }));
        
        setResults(prev => [...newResults, ...prev].slice(0, 100)); // Keep last 100 results
      } catch (err) {
        console.error('Error processing WebSocket message:', err);
      }
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected');
      setIsConnected(false);
      setIsRecording(false);
      setError('Connection lost. Attempting to reconnect...');
      setTimeout(connectWebSocket, 3000);
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setError('Connection error. Check if the server is running.');
    };
  };

  const startMonitoring = async () => {
    try {
      setError(null);
      
      // Request camera and microphone access
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
        audio: true,
      });
      
      mediaStreamRef.current = stream;
      
      // Setup video element
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }
      
      // Setup audio context and analyzer
      const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
      audioContextRef.current = audioContext;
      
      const source = audioContext.createMediaStreamSource(stream);
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 256;
      source.connect(analyser);
      audioAnalyserRef.current = analyser;
      
      const bufferLength = analyser.frequencyBinCount;
      dataArrayRef.current = new Uint8Array(bufferLength);
      
      // Start processing frames
      setIsRecording(true);
      processVideoFrame();
      processAudio();
      
    } catch (err) {
      console.error('Error starting monitoring:', err);
      setError('Failed to access camera or microphone. Please check permissions.');
    }
  };
  
  const stopMonitoring = () => {
    stopMediaStreams();
    setIsRecording(false);
    
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = undefined;
    }
  };
  
  const stopMediaStreams = () => {
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(track => track.stop());
      mediaStreamRef.current = null;
    }
    
    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
  };
  
  const processVideoFrame = () => {
    if (!videoRef.current || !canvasRef.current || !isRecording) return;
    
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    if (!ctx) return;
    
    // Draw video frame to canvas
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Get frame data and send to WebSocket
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      const frameData = canvas.toDataURL('image/jpeg', 0.7).split(',')[1];
      wsRef.current.send(JSON.stringify({
        type: 'video',
        data: frameData
      }));
    }
    
    // Continue processing frames
    animationFrameRef.current = requestAnimationFrame(processVideoFrame);
  };
  
  const processAudio = () => {
    if (!audioAnalyserRef.current || !dataArrayRef.current || !isRecording) return;
    
    const analyser = audioAnalyserRef.current;
    const dataArray = dataArrayRef.current;
    
    // Get audio data
    analyser.getByteFrequencyData(dataArray);
    
    // Calculate average volume
    const sum = dataArray.reduce((a, b) => a + b, 0);
    const avg = sum / dataArray.length;
    setAudioLevel(avg);
    
    // Send audio data to WebSocket if above threshold
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN && avg > 5) {
      wsRef.current.send(JSON.stringify({
        type: 'audio',
        data: Array.from(dataArray)
      }));
    }
    
    // Continue processing audio
    setTimeout(processAudio, 100); // Process audio every 100ms
  };
  
  // Process results for display
  const audioResults = results.filter(r => r.type === 'audio');
  const fallResults = results.filter(r => r.type === 'fall');
  const emotionResults = results.filter(r => r.type === 'emotion');
  
  // Prepare chart data
  const audioChartData = audioResults.map((result, index) => ({
    time: new Date(result.timestamp).toLocaleTimeString(),
    value: result.data.volume || 0,
  }));
  
  const fallChartData = fallResults.map((result, index) => ({
    time: new Date(result.timestamp).toLocaleTimeString(),
    value: result.data.confidence || 0,
  }));
  
  const emotionChartData = emotionResults.map((result, index) => ({
    time: new Date(result.timestamp).toLocaleTimeString(),
    ...result.data.emotions,
  }));
  
  // Tabs configuration
  const items: TabsProps['items'] = [
    {
      key: '1',
      label: 'Dashboard',
      children: (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Card title="Audio Levels" className="h-80">
            <div className="h-48">
              <Line 
                data={audioChartData}
                xField="time"
                yField="value"
                smooth
                animation={false}
              />
            </div>
            <Progress 
              percent={Math.min(100, audioLevel)} 
              showInfo={false}
              strokeColor="#1890ff"
            />
            <div className="flex justify-between mt-2">
              <Text type="secondary">0%</Text>
              <Text>Current: {audioLevel.toFixed(1)}</Text>
              <Text type="secondary">100%</Text>
            </div>
          </Card>
          
          <Card title="Fall Detection" className="h-80">
            <div className="h-48">
              <Line 
                data={fallChartData}
                xField="time"
                yField="value"
                smooth
                animation={false}
              />
            </div>
            <div className="mt-4">
              <Text>Last Detection: {fallResults[0]?.data.detected ? 'Fall Detected!' : 'No falls detected'}</Text>
              <Text type="secondary" className="block mt-2">
                Confidence: {fallResults[0]?.data.confidence?.toFixed(2) || 'N/A'}
              </Text>
            </div>
          </Card>
          
          <Card title="Emotion Analysis" className="h-80">
            <div className="h-48">
              <Line 
                data={emotionChartData}
                xField="time"
                yField="happy"
                seriesField="type"
                smooth
                animation={false}
              />
            </div>
            <div className="mt-4">
              <Text>Dominant Emotion: {emotionResults[0]?.data.dominant_emotion || 'N/A'}</Text>
              <Text type="secondary" className="block mt-2">
                Confidence: {emotionResults[0]?.data.confidence?.toFixed(2) || 'N/A'}
              </Text>
            </div>
          </Card>
          
          <Card title="System Status" className="h-80">
            <div className="space-y-4">
              <div>
                <Text strong>Connection Status:</Text>
                <div className="flex items-center mt-1">
                  <div className={`w-3 h-3 rounded-full mr-2 ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}></div>
                  <Text>{isConnected ? 'Connected' : 'Disconnected'}</Text>
                </div>
              </div>
              
              <div>
                <Text strong>CPU Usage:</Text>
                <Progress percent={cpuUsage} status={cpuUsage > 80 ? 'exception' : 'active'} />
              </div>
              
              <div>
                <Text strong>GPU Usage:</Text>
                <Progress percent={gpuUsage} status={gpuUsage > 80 ? 'exception' : 'active'} />
              </div>
              
              <div className="mt-4">
                {!isRecording ? (
                  <Button 
                    type="primary" 
                    icon={<VideoCameraOutlined />} 
                    onClick={startMonitoring}
                    disabled={!isConnected}
                  >
                    Start Monitoring
                  </Button>
                ) : (
                  <Button 
                    danger 
                    icon={<VideoCameraOutlined />} 
                    onClick={stopMonitoring}
                  >
                    Stop Monitoring
                  </Button>
                )}
              </div>
            </div>
          </Card>
        </div>
      ),
    },
    {
      key: '2',
      label: 'Live Feed',
      children: (
        <div className="relative">
          <video 
            ref={videoRef} 
            autoPlay 
            playsInline 
            muted 
            className="w-full max-w-2xl mx-auto rounded-lg shadow-lg"
          />
          <canvas ref={canvasRef} className="hidden" />
          
          <div className="mt-4 p-4 bg-gray-100 dark:bg-gray-800 rounded-lg">
            <Title level={4} className="flex items-center">
              <AudioOutlined className="mr-2" />
              Audio Level
            </Title>
            <Progress 
              percent={Math.min(100, audioLevel)} 
              showInfo={false}
              strokeColor="#1890ff"
              className="mb-2"
            />
            
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6">
              <StatCard 
                title="Audio Events" 
                value={audioResults.length} 
                icon={<AudioOutlined />} 
                color="#1890ff"
              />
              <StatCard 
                title="Fall Detections" 
                value={fallResults.filter(r => r.data.detected).length} 
                icon={<VideoCameraOutlined />} 
                color="#ff4d4f"
              />
              <StatCard 
                title="Emotion Analysis" 
                value={emotionResults.length} 
                icon={<SmileOutlined />} 
                color="#52c41a"
              />
              <StatCard 
                title="Processing" 
                value={`${((results.length / 100) * 100).toFixed(1)}%`} 
                icon={<SmileOutlined />} 
                color="#faad14"
              />
            </div>
          </div>
        </div>
      ),
    },
  ];

  return (
    <div className="p-4">
      <div className="flex justify-between items-center mb-6">
        <Title level={2} className="mb-0">
          <span className="text-blue-600">Care</span>Taker
          <Text type="secondary" className="ml-2 text-sm font-normal">Real-time Monitoring</Text>
        </Title>
        
        <div className="flex items-center space-x-2">
          <div className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}></div>
          <Text type="secondary">
            {isConnected ? 'Connected' : 'Disconnected'}
          </Text>
          
          <Button 
            type={isRecording ? 'default' : 'primary'} 
            danger={isRecording}
            onClick={isRecording ? stopMonitoring : startMonitoring}
            disabled={!isConnected}
            className="ml-4"
          >
            {isRecording ? 'Stop Monitoring' : 'Start Monitoring'}
          </Button>
        </div>
      </div>
      
      {error && (
        <Alert 
          message="Error" 
          description={error} 
          type="error" 
          showIcon 
          closable 
          className="mb-4"
          onClose={() => setError(null)}
        />
      )}
      
      <Tabs 
        activeKey={activeTab}
        onChange={setActiveTab}
        items={items}
        className="bg-white p-4 rounded-lg shadow"
      />
    </div>
  );
};

// Helper component for stats cards
const StatCard: React.FC<{
  title: string;
  value: string | number;
  icon: React.ReactNode;
  color?: string;
}> = ({ title, value, icon, color = '#1890ff' }) => (
  <Card className="text-center hover:shadow-lg transition-shadow">
    <div className="flex flex-col items-center">
      <div 
        className="w-12 h-12 rounded-full flex items-center justify-center mb-2"
        style={{ backgroundColor: `${color}15`, color }}
      >
        {React.cloneElement(icon as React.ReactElement, { 
          style: { fontSize: '24px' } 
        })}
      </div>
      <Text type="secondary">{title}</Text>
      <Title level={3} className="mt-1 mb-0">{value}</Title>
    </div>
  </Card>
);

export default RealTimeMonitor;
