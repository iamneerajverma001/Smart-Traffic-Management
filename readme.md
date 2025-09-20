# Smart Traffic Management System v3.0

![System Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![AI Powered](https://img.shields.io/badge/AI-YOLOv8-orange)

A professional AI-driven traffic management system that uses computer vision and machine learning to optimize traffic flow through intelligent vehicle detection, emergency vehicle prioritization, and automated traffic light control.

## 🚦 Key Features

### 🤖 **Advanced AI Detection**
- **YOLOv8 Object Detection**: Real-time vehicle classification (cars, trucks, buses, motorcycles, bicycles)
- **Emergency Vehicle Recognition**: OCR-based text detection for ambulances, fire trucks, and police vehicles
- **Siren Detection**: Audio-based emergency vehicle detection using microphone input
- **CUDA Acceleration**: GPU support for enhanced performance

### 🎛️ **Intelligent Traffic Control**
- **Dynamic Priority System**: Weight-based traffic prioritization algorithm
- **Emergency Override**: Automatic priority switching for emergency vehicles
- **Manual Control**: Override capabilities for traffic operators
- **Arduino Integration**: Hardware traffic light control via serial communication

### 🖥️ **Professional Dashboard**
- **Real-time Monitoring**: Live video feeds from multiple camera sources
- **Performance Analytics**: FPS monitoring, detection confidence tracking
- **System Status**: Comprehensive health monitoring and error reporting
- **Modern UI**: PyQt5-based professional interface with dark theme

### 🔧 **Production-Ready Features**
- **Graceful Shutdown**: Safe system termination with progress tracking
- **Error Recovery**: Robust error handling and automatic recovery
- **Comprehensive Logging**: Detailed system logs with rotation
- **Resource Management**: Optimized memory and CPU usage

## 📋 System Requirements

### **Hardware Requirements**
- **CPU**: Intel i5 or AMD Ryzen 5 (minimum)
- **RAM**: 8GB (16GB recommended for optimal performance)
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)
- **Storage**: 2GB free space
- **Cameras**: USB webcams or IP cameras (up to 4 routes supported)
- **Arduino**: Compatible board for traffic light control
- **Microphone**: For siren detection (optional)

### **Software Requirements**
- **Operating System**: Windows 10/11, Linux, or macOS
- **Python**: 3.8 or higher
- **CUDA**: 11.0+ (for GPU acceleration)
- **Tesseract OCR**: For emergency vehicle text recognition

## 🚀 Installation

### 1. **Clone Repository**
```bash
git clone https://github.com/your-username/smart-traffic-management.git
cd smart-traffic-management
```

### 2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 3. **Download YOLOv8 Model**
The system requires the YOLOv8s model file:
```bash
# The model will be automatically downloaded on first run
# Or manually download yolov8s.pt to the project directory
```

### 4. **Install Tesseract OCR**
- **Windows**: Download from [GitHub Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
- **Linux**: `sudo apt-get install tesseract-ocr`
- **macOS**: `brew install tesseract`

### 5. **Arduino Setup** (Optional)
Upload the traffic light control sketch to your Arduino board. Pin configuration:
- **Route 1**: Pins 2 (RED), 3 (GREEN)
- **Route 2**: Pins 4 (RED), 5 (GREEN)  
- **Route 3**: Pins 6 (RED), 7 (GREEN)
- **Route 4**: Pins 8 (RED), 9 (GREEN)

## 🎮 Usage

### **Starting the System**
```bash
python main.py
```

### **Configuration Options**
The setup window allows you to configure:
- **Camera Sources**: USB cameras or IP/RTSP streams
- **Detection Settings**: Confidence thresholds, OCR sensitivity
- **Audio Settings**: Siren detection parameters
- **Traffic Routes**: Number of routes (1-4)

### **Operating Modes**
1. **Automatic Mode**: AI-driven traffic prioritization
2. **Manual Mode**: Operator-controlled traffic lights
3. **Emergency Override**: Priority for emergency vehicles

## 📊 System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Camera Feeds  │───▶│  AI Detection    │───▶│ Traffic Control │
│  (USB/IP/RTSP)  │    │ (YOLOv8 + OCR)   │    │   (Arduino)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Video Processing│    │ Priority Engine  │    │ Light Controller│
│   (OpenCV)      │    │ (Weight-based)   │    │  (Serial Comm)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                        │
         └────────────────────────┼────────────────────────┘
                                  ▼
                     ┌──────────────────┐
                     │   Dashboard UI   │
                     │    (PyQt5)       │
                     └──────────────────┘
```

## 🔧 Configuration

### **Camera Configuration**
- **USB Cameras**: Automatically detected (0, 1, 2, 3...)
- **IP Cameras**: RTSP URLs (rtsp://192.168.1.100:554/stream1)
- **Resolution**: Automatically optimized for performance

### **Detection Parameters**
- **Confidence Threshold**: 0.3-0.9 (default: 0.5)
- **OCR Threshold**: 0.1-1.0 (default: 0.5)
- **Siren Threshold**: 0.01-1.0 (default: 0.08)

### **Traffic Weights**
| Vehicle Type | Weight | Priority Impact |
|--------------|--------|-----------------|
| Cars         | 2      | Standard        |
| Motorcycles  | 1      | Low            |
| Bicycles     | 1      | Low            |
| Buses        | 3      | High           |
| Trucks       | 4      | Highest        |

## 📈 Performance Monitoring

### **Real-time Metrics**
- **Detection FPS**: Processing speed per camera
- **Vehicle Counts**: Live counting by category
- **Priority Scores**: Weight-based traffic analysis
- **System Health**: Resource usage and error tracking

### **Logging System**
```
logs/
├── traffic_system_20250920.log    # Daily log rotation
├── traffic_system_20250919.log
└── ...
```

## 🚨 Emergency Vehicle Detection

### **Detection Methods**
1. **OCR Text Recognition**: Scans for "AMBULANCE", "FIRE", "POLICE"
2. **Siren Audio Detection**: Microphone-based siren identification
3. **Visual Indicators**: Color-coded emergency alerts

### **Emergency Response**
- **Immediate Priority**: Emergency vehicles get instant green light
- **Audio Alerts**: System notification sounds
- **Visual Alerts**: Red highlighting and emergency indicators
- **Override Duration**: Configurable emergency priority time

## 🛠️ Arduino Integration

### **Hardware Setup**
```cpp
// Pin Configuration
int routes[4][2] = {
  {2, 3},   // Route 1: RED, GREEN
  {4, 5},   // Route 2: RED, GREEN  
  {6, 7},   // Route 3: RED, GREEN
  {8, 9}    // Route 4: RED, GREEN
};
```

### **Communication Protocol**
- **Baud Rate**: 9600
- **Command Format**: `PIN:STATE\n` (e.g., "2:1\n" for RED ON)
- **Acknowledgment**: Arduino responds with confirmation

## 📋 File Structure

```
Smart-Traffic-Management/
├── main.py                      # Main application entry point
├── ambulance_detection.py       # Ambulance-specific detection
├── fire_brigade_detection.py    # Fire brigade detection
├── requirements.txt             # Python dependencies
├── yolov8s.pt                  # YOLOv8 model file
├── logs/                       # System logs directory
│   ├── traffic_system_20250920.log
│   └── ...
└── README.md                   # This file
```

## 🔍 Troubleshooting

### **Common Issues**

**🔴 Camera Not Detected**
```bash
# Check camera permissions and connections
# For Linux: sudo usermod -a -G video $USER
# Restart system after permission changes
```

**🔴 Arduino Connection Failed**
```bash
# Check COM port and permissions
# Windows: Device Manager → Ports (COM & LPT)
# Linux: ls /dev/ttyUSB* or ls /dev/ttyACM*
```

**🔴 CUDA Not Available**
```bash
# Install CUDA toolkit and compatible PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**🔴 OCR Not Working**
```bash
# Verify Tesseract installation
tesseract --version
# Update Tesseract path in code if necessary
```

## 📊 Performance Optimization

### **GPU Acceleration**
- Enable CUDA for 3-5x performance improvement
- Automatic fallback to CPU if GPU unavailable
- Memory optimization for multiple camera streams

### **Multi-threading**
- Separate threads for each camera stream
- Parallel detection processing
- Non-blocking UI updates

### **Resource Management**
- Automatic frame queue management
- Memory leak prevention
- CPU usage optimization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Developer

**Er. NEERAJ VERMA**
- Professional Traffic Management System Developer
- AI/ML Engineering Specialist
- Computer Vision Expert

## 🙏 Acknowledgments

- **Ultralytics**: YOLOv8 object detection framework
- **OpenCV**: Computer vision library
- **PyQt5**: Professional GUI framework
- **Arduino Community**: Hardware integration support

---


**🚦 Making Traffic Smarter, One Intersection at a Time! 🚦**