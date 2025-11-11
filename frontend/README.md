# CareTaker - Real-time Monitoring System

A real-time monitoring system for fall detection, audio analysis, and emotion recognition using multiprocessing and GPU acceleration.

## Features

- **Real-time Video Processing**: Fall detection and emotion recognition
- **Audio Analysis**: Background sound classification and analysis
- **GPU Acceleration**: Optimized for performance with CUDA support
- **WebSocket API**: Real-time communication between frontend and backend
- **Responsive Dashboard**: Modern UI with real-time visualizations

## Prerequisites

- Python 3.8+
- Node.js 16+
- CUDA-compatible GPU (optional but recommended)
- Webcam and microphone

## Backend Setup

1. **Create and activate virtual environment**:
   ```bash
   cd backend
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the backend server**:
   ```bash
   python multiprocess_models.py
   ```

## Frontend Setup

1. **Install dependencies**:
   ```bash
   cd frontend
   npm install
   ```

2. **Start the development server**:
   ```bash
   npm start
   ```

3. **Open in browser**:
   ```
   http://localhost:3000
   ```

## Project Structure

```
caretaker/
├── backend/
│   ├── multiprocess_models.py  # Main backend server
│   ├── requirements.txt        # Python dependencies
│   └── models/                 # Model definitions
│       ├── audio/             # Audio processing models
│       └── video/             # Video processing models
├── frontend/
│   ├── public/                # Static files
│   ├── src/
│   │   ├── components/        # React components
│   │   ├── App.tsx            # Main app component
│   │   └── index.tsx          # Entry point
│   ├── package.json           # Frontend dependencies
│   └── tsconfig.json          # TypeScript config
└── README.md                  # This file
```

## Configuration

### Backend Configuration

Edit `multiprocess_models.py` to adjust:
- WebSocket port (default: 8000)
- Model parameters
- Processing settings

### Frontend Configuration

Edit `.env` to configure:
- API endpoints
- WebSocket URL
- UI settings

## Troubleshooting

### Common Issues

1. **WebSocket Connection Failed**
   - Ensure backend server is running
   - Check for firewall/port conflicts
   - Verify CORS settings

2. **GPU Not Detected**
   - Install CUDA Toolkit
   - Verify GPU drivers are up to date
   - Check TensorFlow/PyTorch GPU support

3. **High CPU Usage**
   - Reduce video resolution
   - Increase frame processing interval
   - Disable unused models

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
