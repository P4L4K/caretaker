
document.addEventListener('DOMContentLoaded', () => {
    const recordingStatus = document.getElementById('recording-status');
    const processingStatus = document.getElementById('processing-status');
    const silenceStatus = document.getElementById('silence-status');
    const waveformCanvas = document.getElementById('waveform');
    const rmsLevel = document.getElementById('rms-level');
    const logoutLink = document.querySelector('a[href="index.html"]');

    let socket;
    const canvasCtx = waveformCanvas.getContext('2d');
    let animationFrameId;

    function drawWaveform(data) {
        const width = waveformCanvas.width;
        const height = waveformCanvas.height;
        const center = height / 2;
        const sliceWidth = width * 1.0 / data.length;
        let x = 0;

        canvasCtx.fillStyle = 'rgb(200, 200, 200)';
        canvasCtx.fillRect(0, 0, width, height);

        canvasCtx.lineWidth = 2;
        canvasCtx.strokeStyle = 'rgb(0, 0, 0)';

        canvasCtx.beginPath();

        for (let i = 0; i < data.length; i++) {
            const v = data[i] / 32768.0;
            const lineHeight = v * height;

            const y1 = center - lineHeight / 2;
            const y2 = center + lineHeight / 2;

            canvasCtx.moveTo(x, y1);
            canvasCtx.lineTo(x, y2);

            x += sliceWidth;
        }

        canvasCtx.stroke();
    }

    function updateRMS(rms) {
        const percentage = (rms / 800) * 100;
        rmsLevel.style.width = `${Math.min(percentage, 100)}%`;
    }

    function connectWebSocket() {
        const token = localStorage.getItem('token');
        if (!token) {
            window.location.href = 'index.html';
            return;
        }
        socket = new WebSocket(`ws://localhost:8000/ws/audio?token=${token}`);

        socket.onopen = () => {
            console.log('WebSocket connection established.');
            recordingStatus.textContent = 'Recording';
            processingStatus.textContent = 'Processing';
        };

        socket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            drawWaveform(data.waveform);
            updateRMS(data.rms);

            if (data.rms < 100) {
                silenceStatus.textContent = 'Silence Detected';
            } else {
                silenceStatus.textContent = '';
            }
        };

        socket.onclose = () => {
            console.log('WebSocket connection closed.');
            recordingStatus.textContent = 'Not Recording';
            processingStatus.textContent = '';
            silenceStatus.textContent = '';
            cancelAnimationFrame(animationFrameId);
        };

        socket.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
    }

    // Auto-start recording on page load
    connectWebSocket();

    // Stop recording on logout
    logoutLink.addEventListener('click', async (e) => {
        e.preventDefault(); // Prevent default link behavior

        if (socket) {
            socket.close();
        }

        try {
            const token = localStorage.getItem('token');
            const response = await fetch('http://localhost:8000/logout', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ token })
            });

            if (response.ok) {
                localStorage.removeItem('token');
                window.location.href = 'index.html'; // Redirect to login page
            } else {
                console.error('Logout failed:', await response.text());
            }
        } catch (error) {
            console.error('Error during logout:', error);
        }
    });
});
