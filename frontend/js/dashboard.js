// CareTaker AI Dashboard - Complete Implementation
const API_BASE = 'http://localhost:8000';
let audioSocket = null;
let videoStream = null;
let emotionChart = null;
let allDetections = [];
let recentDetections = [];
let statsUpdateInterval = null;

document.addEventListener('DOMContentLoaded', () => {
    const token = localStorage.getItem('token');
    if (!token) {
        window.location.href = 'index.html';
        return;
    }

    initializeDashboard();
    setupEventListeners();
    // Don't auto-connect audio - let user start it manually
    // connectAudioWebSocket();
    loadDetectionHistory();
});

function initializeDashboard() {
    const username = localStorage.getItem('username') || 'User';
    document.getElementById('username-display').textContent = username;
    initializeEmotionChart();
    
    const today = new Date().toISOString().split('T')[0];
    document.getElementById('date-to').value = today;
    const weekAgo = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString().split('T')[0];
    document.getElementById('date-from').value = weekAgo;
    
    // Start real-time stats updates
    fetchAndUpdateStats();
    statsUpdateInterval = setInterval(fetchAndUpdateStats, 5000); // Update every 5 seconds
}

function setupEventListeners() {
    // Sidebar navigation
    document.querySelectorAll('.menu-item').forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            switchSection(item.dataset.section);
        });
    });

    const logoutBtn = document.getElementById('logout-btn');
    if (logoutBtn) {
        logoutBtn.addEventListener('click', (e) => {
            e.preventDefault();
            console.log('Logout button clicked');
            handleLogout();
        });
        console.log('Logout button event listener attached');
    } else {
        console.error('Logout button not found!');
    }
    
    document.getElementById('clear-alerts').addEventListener('click', () => {
        recentDetections = [];
        updateRecentDetectionsList();
    });
    document.getElementById('toggle-video').addEventListener('click', toggleVideo);
    document.getElementById('apply-filters').addEventListener('click', applyFilters);
    document.getElementById('export-history').addEventListener('click', exportHistory);
}

function switchSection(sectionName) {
    document.querySelectorAll('.menu-item').forEach(item => item.classList.remove('active'));
    document.querySelector(`[data-section="${sectionName}"]`).classList.add('active');
    document.querySelectorAll('.content-section').forEach(section => section.classList.remove('active'));
    document.getElementById(`${sectionName}-section`).classList.add('active');
    
    // Connect audio when switching to audio section
    if (sectionName === 'audio' && !audioSocket) {
        connectAudioWebSocket();
    }
    
    if (sectionName === 'history') loadDetectionHistory();
}

function connectAudioWebSocket() {
    if (audioSocket) {
        const state = audioSocket.readyState;
        const stateStr = ['CONNECTING', 'OPEN', 'CLOSING', 'CLOSED'][state] || state;
        console.log(`Audio WebSocket already exists with state: ${stateStr}`);
        return;
    }
    
    const token = localStorage.getItem('token');
    if (!token) {
        console.error('No authentication token found');
        return;
    }
    
    const statusDot = document.getElementById('audio-status-dot');
    const statusText = document.getElementById('audio-status-text');

    statusText.textContent = 'Connecting...';
    console.log('Connecting to WebSocket endpoint: ws://localhost:8000/ws/unified');

    try {
        const wsUrl = `ws://localhost:8000/ws/unified?token=${encodeURIComponent(token)}`;
        console.log('Creating WebSocket with URL:', wsUrl);
        audioSocket = new WebSocket(wsUrl);

        audioSocket.onopen = (event) => {
            console.log('WebSocket connection established successfully', event);
            statusText.textContent = 'Connected';
            statusDot.classList.add('connected');
            
            // Send a test message to verify the connection
            audioSocket.send(JSON.stringify({
                type: 'handshake',
                message: 'Client connected',
                timestamp: new Date().toISOString()
            }));
        };
    } catch (error) {
        console.error('Failed to create WebSocket:', error);
        statusText.textContent = 'Connection failed';
        statusDot.classList.remove('connected');
        return;
    }

    audioSocket.onmessage = (event) => {
        console.log('Received WebSocket message:', event.data);
        try {
            const data = JSON.parse(event.data);
            console.log('Parsed WebSocket data:', data);
            
            // Handle different message types
            if (data.type === 'debug') {
                console.log('Server debug:', data.message);
                return;
            }
            
            if (data.waveform) {
                drawWaveform(data.waveform);
                updateAudioLevels(data.rms, data.db);
            }
            
            if (data.event === 'prediction') {
                console.log('Prediction received:', data);
                if (data.label === 'Cough') {
                    handleCoughDetection(data);
                }
            }
        } catch (error) {
            console.error('Error processing WebSocket message:', error, 'Raw data:', event.data);
        }
    };

    audioSocket.onclose = (event) => {
        console.log('WebSocket connection closed:', {
            code: event.code,
            reason: event.reason,
            wasClean: event.wasClean
        });
        
        statusText.textContent = 'Disconnected';
        statusDot.classList.remove('connected');
        
        // Clear the socket reference
        const oldSocket = audioSocket;
        audioSocket = null;
        
        // Attempt to reconnect after a delay if the closure was unexpected
        if (!event.wasClean) {
            console.log('Attempting to reconnect in 3 seconds...');
            setTimeout(connectAudioWebSocket, 3000);
        }
    };

    audioSocket.onerror = (error) => {
        console.error('WebSocket error:', {
            type: error.type,
            message: error.message || 'Unknown WebSocket error',
            timestamp: new Date().toISOString()
        });
        
        statusText.textContent = 'Connection Error';
        statusDot.classList.remove('connected');
        
        // The socket will be closed after an error, so we'll rely on onclose to handle reconnection
    };
}

function disconnectAudioWebSocket() {
    if (audioSocket) {
        audioSocket.close();
        audioSocket = null;
        console.log('Audio WebSocket manually disconnected');
    }
}

function drawWaveform(data) {
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

    const sliceWidth = width / data.length;
    let x = 0;

    for (let i = 0; i < data.length; i++) {
        const v = data[i] / 32768.0;
        const y = center + (v * height * 0.8);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
        x += sliceWidth;
    }
    ctx.stroke();
}

function updateAudioLevels(rms, db) {
    document.getElementById('db-level').textContent = db.toFixed(1) + ' dB';
    const percentage = Math.min((rms / 5000) * 100, 100);
    document.getElementById('rms-level').style.width = percentage + '%';
}

function handleCoughDetection(data) {
    recentDetections.unshift(data);
    if (recentDetections.length > 10) recentDetections.pop();
    updateRecentDetectionsList();
    updateStats();
    allDetections.unshift(data);
}

function updateRecentDetectionsList() {
    const container = document.getElementById('recent-detections');
    if (recentDetections.length === 0) {
        container.innerHTML = '<div class="empty-state"><i class="fas fa-inbox"></i><p>No cough detections yet...</p></div>';
        return;
    }
    
    console.log('Updating recent detections list:', recentDetections);
    
    container.innerHTML = recentDetections.map(d => {
        console.log('Detection media_url:', d.media_url);
        return `
        <div class="detection-item">
            <div class="detection-info">
                <div class="detection-time">${new Date(d.timestamp).toLocaleString()}</div>
                <div class="detection-confidence">Confidence: ${(d.probability * 100).toFixed(1)}%</div>
            </div>
            <button class="btn-icon" onclick="playAudio('${d.media_url}')" title="Play"><i class="fas fa-play"></i></button>
        </div>
    `;
    }).join('');
}

function updateStats() {
    document.getElementById('total-coughs').textContent = allDetections.length;
    if (allDetections.length > 0) {
        const lastTime = new Date(allDetections[0].timestamp);
        document.getElementById('last-cough-time').textContent = lastTime.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
        const avgConf = allDetections.reduce((sum, d) => sum + d.probability, 0) / allDetections.length;
        document.getElementById('avg-confidence').textContent = (avgConf * 100).toFixed(0) + '%';
    }
    const today = new Date().toDateString();
    const todayCoughs = allDetections.filter(d => new Date(d.timestamp).toDateString() === today).length;
    document.getElementById('today-coughs').textContent = todayCoughs;
}

async function fetchAndUpdateStats() {
    try {
        const response = await fetch(`${API_BASE}/api/cough/stats`);
        if (!response.ok) {
            console.error('Failed to fetch stats:', response.status);
            return;
        }
        const stats = await response.json();
        
        // Update dashboard stats
        document.getElementById('total-coughs').textContent = stats.total_detections || 0;
        document.getElementById('today-coughs').textContent = stats.today_detections || 0;
        
        if (stats.last_detection_time) {
            const lastTime = new Date(stats.last_detection_time);
            document.getElementById('last-cough-time').textContent = lastTime.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
        } else {
            document.getElementById('last-cough-time').textContent = '--:--';
        }
        
        if (stats.average_probability) {
            document.getElementById('avg-confidence').textContent = (stats.average_probability * 100).toFixed(0) + '%';
        } else {
            document.getElementById('avg-confidence').textContent = '0%';
        }
        
        // Update recent detections list
        if (stats.recent_detections && stats.recent_detections.length > 0) {
            recentDetections = stats.recent_detections;
            updateRecentDetectionsList();
        }
    } catch (error) {
        console.error('Error fetching stats:', error);
    }
}

async function toggleVideo() {
    const button = document.getElementById('toggle-video');
    const videoElement = document.getElementById('video-feed');
    const statusDot = document.getElementById('video-status-dot');
    const statusText = document.getElementById('video-status-text');
    
    if (!videoStream) {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            videoElement.srcObject = stream;
            videoStream = stream;
            button.innerHTML = '<i class="fas fa-stop"></i> Stop Video';
            statusText.textContent = 'Active - Analyzing';
            statusDot.classList.add('connected');
            connectVideoStream();
        } catch (error) {
            alert('Could not access webcam: ' + error.message);
            statusText.textContent = 'Error';
            statusDot.classList.remove('connected');
        }
    } else {
        videoStream.getTracks().forEach(track => track.stop());
        videoElement.srcObject = null;
        videoStream = null;
        button.innerHTML = '<i class="fas fa-play"></i> Start Video';
        statusText.textContent = 'Stopped';
        statusDot.classList.remove('connected');
        if (window.videoAnalyzeInterval) {
            clearInterval(window.videoAnalyzeInterval);
            window.videoAnalyzeInterval = null;
        }
    }
}

function connectVideoStream() {
    const videoElement = document.getElementById('video-feed');
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    // Set canvas size to match video
    canvas.width = 640;
    canvas.height = 480;
    
    // Wait for video to be ready before starting analysis
    const startAnalysis = () => {
        console.log('Starting video analysis...');
        
        // Capture and analyze frames periodically
        const analyzeInterval = setInterval(async () => {
            if (!videoStream) {
                clearInterval(analyzeInterval);
                return;
            }
            
            // Check if video is ready
            if (videoElement.readyState < 2) {
                console.log('Video not ready yet, skipping frame...');
                return;
            }
            
            try {
                // Draw current video frame to canvas
                ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
                
                // Convert canvas to blob
                canvas.toBlob(async (blob) => {
                    if (!blob) {
                        console.warn('Failed to create blob from canvas');
                        return;
                    }
                    
                    // Send frame to backend for analysis
                    const formData = new FormData();
                    formData.append('file', blob, 'frame.jpg');
                    
                    try {
                        const response = await fetch(`${API_BASE}/video/analyze`, {
                            method: 'POST',
                            body: formData
                        });
                        
                        if (response.ok) {
                            const data = await response.json();
                            console.log('Video analysis result:', data);
                            document.getElementById('current-emotion').textContent = data.mood || 'neutral';
                            updateEmotionChart(data.mood || 'neutral');
                            if (data.fall_detected) showFallAlert(data.timestamp);
                        } else {
                            const errorText = await response.text();
                            console.error('Video analysis failed:', response.status, errorText);
                        }
                    } catch (fetchError) {
                        console.error('Fetch error:', fetchError);
                    }
                }, 'image/jpeg', 0.8);
            } catch (error) {
                console.error('Frame capture error:', error);
            }
        }, 1500); // Analyze every 1.5 seconds
        
        // Store interval ID for cleanup
        window.videoAnalyzeInterval = analyzeInterval;
    };
    
    // Wait for video to be ready
    if (videoElement.readyState >= 2) {
        startAnalysis();
    } else {
        videoElement.addEventListener('loadeddata', startAnalysis, { once: true });
        // Fallback timeout in case loadeddata never fires
        setTimeout(startAnalysis, 2000);
    }
}

function initializeEmotionChart() {
    const ctx = document.getElementById('emotion-chart').getContext('2d');
    emotionChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Happy', 'Sad', 'Angry', 'Neutral', 'Surprise', 'Fear', 'Disgust'],
            datasets: [{ 
                data: [0, 0, 0, 0, 0, 0, 0], 
                backgroundColor: ['#4CAF50', '#2196F3', '#F44336', '#9E9E9E', '#FF9800', '#9C27B0', '#795548'] 
            }]
        },
        options: { 
            responsive: true, 
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom'
                }
            }
        }
    });
}

function updateEmotionChart(emotion) {
    const map = { 
        'happy': 0, 
        'sad': 1, 
        'angry': 2, 
        'neutral': 3, 
        'surprise': 4, 
        'fear': 5,
        'disgust': 6
    };
    const emotionLower = emotion.toLowerCase();
    const index = map[emotionLower];
    
    if (index !== undefined) {
        emotionChart.data.datasets[0].data[index]++;
        emotionChart.update();
    } else {
        console.warn('Unknown emotion:', emotion);
    }
}

function showFallAlert(timestamp) {
    document.getElementById('fall-timestamp').textContent = new Date(timestamp).toLocaleString();
    document.getElementById('fall-alert').style.display = 'block';
}

async function loadDetectionHistory() {
    const tbody = document.getElementById('history-tbody');
    tbody.innerHTML = '<tr class="loading-row"><td colspan="5"><div class="loading-spinner"></div><p>Loading...</p></td></tr>';
    
    try {
        const response = await fetch(`${API_BASE}/api/cough/detections/all`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        allDetections = data.items || [];
        displayHistory(allDetections);
        updateStats();
    } catch (error) {
        console.error('Error loading detection history:', error);
        tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;color:#F44336;">Error loading history</td></tr>';
    }
}

function displayHistory(detections) {
    const tbody = document.getElementById('history-tbody');
    if (detections.length === 0) {
        tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;padding:2rem;">No detections found</td></tr>';
        return;
    }
    tbody.innerHTML = detections.map(d => {
        const conf = (d.probability * 100).toFixed(1);
        const confClass = conf >= 80 ? 'confidence-high' : conf >= 60 ? 'confidence-medium' : 'confidence-low';
        return `
            <tr>
                <td>${new Date(d.timestamp).toLocaleString()}</td>
                <td>${d.username || 'Unknown'}</td>
                <td><span class="confidence-badge ${confClass}">${conf}%</span></td>
                <td><button class="btn-icon" onclick="playAudio('${d.media_url}')"><i class="fas fa-play"></i></button></td>
                <td><button class="btn-icon" onclick="downloadAudio('${d.media_url}')"><i class="fas fa-download"></i></button></td>
            </tr>
        `;
    }).join('');
}

function applyFilters() {
    const search = document.getElementById('search-input').value.toLowerCase();
    const dateFrom = document.getElementById('date-from').value;
    const dateTo = document.getElementById('date-to').value;
    
    let filtered = [...allDetections];
    if (search) filtered = filtered.filter(d => (d.username || '').toLowerCase().includes(search));
    if (dateFrom) filtered = filtered.filter(d => new Date(d.timestamp) >= new Date(dateFrom));
    if (dateTo) {
        const to = new Date(dateTo);
        to.setHours(23, 59, 59);
        filtered = filtered.filter(d => new Date(d.timestamp) <= to);
    }
    displayHistory(filtered);
}

function exportHistory() {
    const csv = [['Timestamp', 'Username', 'Confidence', 'Audio URL'], ...allDetections.map(d => [
        d.timestamp, d.username || 'Unknown', (d.probability * 100).toFixed(1) + '%', API_BASE + d.media_url
    ])].map(row => row.join(',')).join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `cough-detections-${new Date().toISOString().split('T')[0]}.csv`;
    a.click();
    URL.revokeObjectURL(url);
}

function playAudio(mediaUrl) {
    const modal = document.getElementById('audio-modal');
    const player = document.getElementById('audio-player');
    
    console.log('Playing audio:', API_BASE + mediaUrl);
    
    // Set the audio source
    player.src = API_BASE + mediaUrl;
    
    // Show the modal
    modal.classList.add('active');
    modal.style.display = 'flex';
    
    // Play the audio with error handling
    player.play().catch(error => {
        console.error('Error playing audio:', error);
        alert('Failed to play audio. Please check if the file exists.');
    });
}

function closeAudioModal() {
    const modal = document.getElementById('audio-modal');
    const player = document.getElementById('audio-player');
    player.pause();
    player.src = '';
    modal.classList.remove('active');
    modal.style.display = 'none';
}

function downloadAudio(mediaUrl) {
    const a = document.createElement('a');
    a.href = API_BASE + mediaUrl;
    a.download = mediaUrl.split('/').pop();
    a.click();
}

// Expose functions to global scope for inline onclick handlers
window.playAudio = playAudio;
window.closeAudioModal = closeAudioModal;
window.downloadAudio = downloadAudio;

async function handleLogout() {
    console.log('Logout initiated...');
    
    // Clean up active connections and intervals
    if (audioSocket) {
        console.log('Closing audio WebSocket...');
        audioSocket.close();
        audioSocket = null;
    }
    
    if (videoStream) {
        console.log('Stopping video stream...');
        videoStream.getTracks().forEach(track => track.stop());
        videoStream = null;
        
        // Clear video element
        const videoElement = document.getElementById('video-feed');
        if (videoElement) {
            videoElement.srcObject = null;
        }
    }
    
    if (window.videoAnalyzeInterval) {
        console.log('Clearing video analysis interval...');
        clearInterval(window.videoAnalyzeInterval);
        window.videoAnalyzeInterval = null;
    }
    
    if (statsUpdateInterval) {
        console.log('Clearing stats update interval...');
        clearInterval(statsUpdateInterval);
        statsUpdateInterval = null;
    }
    
    // Destroy Chart.js instance
    if (emotionChart) {
        console.log('Destroying emotion chart...');
        emotionChart.destroy();
        emotionChart = null;
    }
    
    // Call logout endpoint to blocklist the token
    try {
        const token = localStorage.getItem('token');
        if (token) {
            console.log('Calling logout endpoint...');
            const response = await fetch(`${API_BASE}/logout`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ token: token })
            });
            
            if (response.ok) {
                const data = await response.json();
                console.log('Logout successful:', data);
            } else {
                const errorText = await response.text();
                console.warn('Logout endpoint returned error:', response.status, errorText);
            }
        } else {
            console.warn('No token found in localStorage');
        }
    } catch (error) {
        console.error('Logout API call failed:', error);
        // Continue with logout even if API call fails
    }
    
    // Clear local storage
    console.log('Clearing localStorage...');
    localStorage.removeItem('token');
    localStorage.removeItem('username');
    
    // Redirect to login page
    console.log('Redirecting to login page...');
    window.location.href = 'index.html';
}
