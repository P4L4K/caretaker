// Unified Real-Time Monitoring System
const API_BASE = 'http://localhost:8000';
let monitorSocket = null;
let videoStream = null;
let isMonitoring = false;
let detections = [];
let emotionCount = 0;
let fallCount = 0;
let coughCount = 0;
let coughConfidences = [];

// Check authentication
document.addEventListener('DOMContentLoaded', () => {
    const token = localStorage.getItem('token');
    if (!token) {
        window.location.href = 'index.html';
        return;
    }

    const username = localStorage.getItem('username') || 'User';
    document.getElementById('username-display').textContent = username;

    setupEventListeners();
});

function setupEventListeners() {
    document.getElementById('start-monitoring').addEventListener('click', startMonitoring);
    document.getElementById('stop-monitoring').addEventListener('click', stopMonitoring);
    document.getElementById('clear-detections').addEventListener('click', clearDetections);
    document.getElementById('logout-btn').addEventListener('click', handleLogout);
}

async function startMonitoring() {
    try {
        // Get video stream first
        videoStream = await navigator.mediaDevices.getUserMedia({ 
            video: { width: 640, height: 480 },
            audio: false 
        });
        
        const videoElement = document.getElementById('video-feed');
        videoElement.srcObject = videoStream;
        
        // Connect to unified WebSocket
        connectMonitorWebSocket();
        
        // Start sending video frames
        startVideoFrameCapture();
        
        // Update UI
        document.getElementById('start-monitoring').disabled = true;
        document.getElementById('stop-monitoring').disabled = false;
        document.getElementById('video-status-text').textContent = 'Active';
        document.getElementById('video-status-dot').classList.add('active');
        
        isMonitoring = true;
        
        console.log('✓ Unified monitoring started');
    } catch (error) {
        console.error('Failed to start monitoring:', error);
        alert('Could not access camera/microphone: ' + error.message);
    }
}

function stopMonitoring() {
    // Stop video stream
    if (videoStream) {
        videoStream.getTracks().forEach(track => track.stop());
        videoStream = null;
        document.getElementById('video-feed').srcObject = null;
    }
    
    // Close WebSocket
    if (monitorSocket) {
        monitorSocket.close();
        monitorSocket = null;
    }
    
    // Stop video frame capture
    if (window.videoFrameInterval) {
        clearInterval(window.videoFrameInterval);
        window.videoFrameInterval = null;
    }
    
    // Update UI
    document.getElementById('start-monitoring').disabled = false;
    document.getElementById('stop-monitoring').disabled = true;
    document.getElementById('video-status-text').textContent = 'Stopped';
    document.getElementById('video-status-dot').classList.remove('active');
    document.getElementById('audio-status-text').textContent = 'Stopped';
    document.getElementById('audio-status-dot').classList.remove('active');
    
    isMonitoring = false;
    
    console.log('✓ Monitoring stopped');
}

function connectMonitorWebSocket() {
    const token = localStorage.getItem('token');
    
    document.getElementById('audio-status-text').textContent = 'Connecting...';
    console.log('Connecting to unified monitoring WebSocket...');
    
    monitorSocket = new WebSocket(`ws://localhost:8000/ws/unified?token=${token}`);
    
    monitorSocket.onopen = () => {
        document.getElementById('audio-status-text').textContent = 'Active';
        document.getElementById('audio-status-dot').classList.add('active');
        console.log('✓ Unified monitoring WebSocket connected');
    };
    
    monitorSocket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        if (data.type === 'audio_waveform') {
            handleAudioWaveform(data);
        } else if (data.type === 'cough_detection') {
            handleCoughDetection(data);
        } else if (data.type === 'video_analysis') {
            handleVideoAnalysis(data);
        }
    };
    
    monitorSocket.onclose = () => {
        document.getElementById('audio-status-text').textContent = 'Disconnected';
        document.getElementById('audio-status-dot').classList.remove('active');
        monitorSocket = null;
        console.log('WebSocket disconnected');
    };
    
    monitorSocket.onerror = (error) => {
        console.error('WebSocket error:', error);
        document.getElementById('audio-status-text').textContent = 'Error';
        document.getElementById('audio-status-dot').classList.remove('active');
    };
}

function startVideoFrameCapture() {
    const videoElement = document.getElementById('video-feed');
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    canvas.width = 640;
    canvas.height = 480;
    
    // Wait for video to be ready
    videoElement.addEventListener('loadeddata', () => {
        console.log('Video ready, starting frame capture...');
        
        // Capture and send frames every 1.5 seconds
        window.videoFrameInterval = setInterval(() => {
            if (!isMonitoring || !monitorSocket || monitorSocket.readyState !== WebSocket.OPEN) {
                return;
            }
            
            try {
                // Draw current video frame to canvas
                ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
                
                // Convert to base64
                canvas.toBlob((blob) => {
                    if (!blob) return;
                    
                    const reader = new FileReader();
                    reader.onloadend = () => {
                        const base64data = reader.result.split(',')[1];
                        
                        // Send frame to backend via WebSocket
                        if (monitorSocket && monitorSocket.readyState === WebSocket.OPEN) {
                            monitorSocket.send(JSON.stringify({
                                type: 'video_frame',
                                frame: base64data
                            }));
                        }
                    };
                    reader.readAsDataURL(blob);
                }, 'image/jpeg', 0.8);
            } catch (error) {
                console.error('Frame capture error:', error);
            }
        }, 1500); // Send frame every 1.5 seconds
    }, { once: true });
}

function handleAudioWaveform(data) {
    // Draw waveform
    const canvas = document.getElementById('waveform');
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    const center = height / 2;
    
    ctx.fillStyle = '#f8f9fa';
    ctx.fillRect(0, 0, width, height);
    ctx.strokeStyle = '#2196F3';
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    const waveform = data.waveform;
    const sliceWidth = width / waveform.length;
    let x = 0;
    
    for (let i = 0; i < waveform.length; i++) {
        const v = waveform[i] / 32768.0;
        const y = center + (v * height * 0.8);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
        x += sliceWidth;
    }
    ctx.stroke();
    
    // Update audio metrics
    document.getElementById('rms-value').textContent = Math.round(data.rms);
    document.getElementById('db-value').textContent = data.db.toFixed(1);
    document.getElementById('gate-status').textContent = data.gate_open ? 'Open' : 'Closed';
}

function handleCoughDetection(data) {
    console.log('Cough detected:', data);
    
    coughCount++;
    coughConfidences.push(data.probability);
    
    // Update stats
    document.getElementById('cough-count').textContent = coughCount;
    const avgConf = coughConfidences.reduce((a, b) => a + b, 0) / coughConfidences.length;
    document.getElementById('avg-confidence').textContent = Math.round(avgConf * 100) + '%';
    
    // Add to detections list
    addDetection({
        type: 'cough',
        timestamp: data.timestamp,
        confidence: data.probability,
        mediaUrl: data.media_url
    });
    
    // Play notification sound
    playNotificationSound();
    
    // Show browser notification
    if (Notification.permission === 'granted') {
        new Notification('Cough Detected', {
            body: `Confidence: ${Math.round(data.probability * 100)}%`,
            icon: '/favicon.ico'
        });
    }
}

function handleVideoAnalysis(data) {
    // Update emotion display
    document.getElementById('current-emotion').textContent = data.emotion;
    emotionCount++;
    document.getElementById('emotion-count').textContent = emotionCount;
    
    // Handle fall detection
    if (data.fall_detected) {
        console.log('Fall detected:', data);
        
        fallCount++;
        document.getElementById('fall-count').textContent = fallCount;
        
        // Show fall alert badge
        const fallAlert = document.getElementById('fall-alert');
        fallAlert.classList.add('active');
        setTimeout(() => {
            fallAlert.classList.remove('active');
        }, 5000);
        
        // Add to detections list
        addDetection({
            type: 'fall',
            timestamp: data.fall_timestamp || data.timestamp,
            emotion: data.emotion
        });
        
        // Play alert sound
        playAlertSound();
        
        // Show browser notification
        if (Notification.permission === 'granted') {
            new Notification('⚠️ FALL DETECTED!', {
                body: 'Immediate attention required!',
                icon: '/favicon.ico',
                requireInteraction: true
            });
        }
    }
}

function addDetection(detection) {
    detections.unshift(detection);
    if (detections.length > 50) {
        detections.pop();
    }
    
    updateDetectionsList();
}

function updateDetectionsList() {
    const container = document.getElementById('detections-list');
    
    if (detections.length === 0) {
        container.innerHTML = `
            <div style="text-align: center; padding: 40px; color: #999;">
                <i class="fas fa-inbox" style="font-size: 3rem; margin-bottom: 15px;"></i>
                <p>No detections yet. Start monitoring to see results.</p>
            </div>
        `;
        return;
    }
    
    container.innerHTML = detections.map(d => {
        const time = new Date(d.timestamp).toLocaleString();
        
        if (d.type === 'cough') {
            return `
                <div class="detection-item cough">
                    <div class="detection-time">${time}</div>
                    <div class="detection-type">
                        <i class="fas fa-lungs"></i> Cough Detected
                    </div>
                    <div class="detection-confidence">
                        Confidence: ${Math.round(d.confidence * 100)}%
                    </div>
                    ${d.mediaUrl ? `<audio controls src="${API_BASE}${d.mediaUrl}" style="width: 100%; margin-top: 10px;"></audio>` : ''}
                </div>
            `;
        } else if (d.type === 'fall') {
            return `
                <div class="detection-item fall">
                    <div class="detection-time">${time}</div>
                    <div class="detection-type">
                        <i class="fas fa-exclamation-triangle"></i> Fall Detected
                    </div>
                    <div class="detection-confidence">
                        Emotion at time: ${d.emotion || 'unknown'}
                    </div>
                </div>
            `;
        }
    }).join('');
}

function clearDetections() {
    if (confirm('Clear all detections?')) {
        detections = [];
        updateDetectionsList();
    }
}

function playNotificationSound() {
    // Create a simple beep sound
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const oscillator = audioContext.createOscillator();
    const gainNode = audioContext.createGain();
    
    oscillator.connect(gainNode);
    gainNode.connect(audioContext.destination);
    
    oscillator.frequency.value = 800;
    oscillator.type = 'sine';
    
    gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
    gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.5);
    
    oscillator.start(audioContext.currentTime);
    oscillator.stop(audioContext.currentTime + 0.5);
}

function playAlertSound() {
    // Create an urgent alert sound
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    
    for (let i = 0; i < 3; i++) {
        const oscillator = audioContext.createOscillator();
        const gainNode = audioContext.createGain();
        
        oscillator.connect(gainNode);
        gainNode.connect(audioContext.destination);
        
        oscillator.frequency.value = 1000;
        oscillator.type = 'square';
        
        const startTime = audioContext.currentTime + (i * 0.3);
        gainNode.gain.setValueAtTime(0.5, startTime);
        gainNode.gain.exponentialRampToValueAtTime(0.01, startTime + 0.2);
        
        oscillator.start(startTime);
        oscillator.stop(startTime + 0.2);
    }
}

function handleLogout() {
    const token = localStorage.getItem('token');
    
    // Stop monitoring first
    if (isMonitoring) {
        stopMonitoring();
    }
    
    fetch(`${API_BASE}/logout`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}`
        }
    })
    .then(() => {
        localStorage.removeItem('token');
        localStorage.removeItem('username');
        window.location.href = 'index.html';
    })
    .catch(error => {
        console.error('Logout error:', error);
        // Still redirect even if logout fails
        localStorage.removeItem('token');
        localStorage.removeItem('username');
        window.location.href = 'index.html';
    });
}

// Request notification permission on load
if (Notification.permission === 'default') {
    Notification.requestPermission();
}
