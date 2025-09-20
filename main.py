"""
Smart Traffic Management System v3.0
Production-Ready Implementation with Enhanced Shutdown and Priority Logic

Features:
- Smooth graceful shutdown with progress tracking
- Fixed vehicle detection weight-based priority system
- Professional error handling and recovery
- Optimized performance and resource management
- Comprehensive logging and monitoring
"""

import sys
import os
import time
import threading
import queue
import logging
import signal
import atexit
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import cv2
import torch
import easyocr
import sounddevice as sd
import serial
import serial.tools.list_ports
import psutil
from datetime import datetime, timedelta
from ultralytics import YOLO

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QRadioButton, QSpinBox, QFormLayout, QGroupBox, QMessageBox,
    QMainWindow, QGridLayout, QInputDialog, QProgressBar, QTextEdit,
    QSplitter, QFrame, QTabWidget, QScrollArea, QSlider, QCheckBox,
    QSystemTrayIcon, QMenu, QAction, QDialog, QDialogButtonBox
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QThread, pyqtSlot, QMutex, QWaitCondition
from PyQt5.QtGui import QPixmap, QImage, QColor, QFont, QIcon, QPalette, QCloseEvent

# Configure comprehensive logging
def setup_logging():
    """Setup comprehensive logging system"""
    log_format = '%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s'
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Setup file handler with rotation
    file_handler = logging.FileHandler(f'logs/traffic_system_{datetime.now().strftime("%Y%m%d")}.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format))
    
    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format))
    
    # Configure root logger
    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[file_handler, console_handler]
    )
    
    return logging.getLogger(__name__)

logger = setup_logging()

class SystemState(Enum):
    """System state enumeration"""
    INITIALIZING = "Initializing"
    READY = "Ready"
    RUNNING = "Running"
    PAUSED = "Paused"
    SHUTTING_DOWN = "Shutting Down"
    ERROR = "Error"
    OFFLINE = "Offline"

class Priority(Enum):
    """Traffic priority levels"""
    EMERGENCY = 1
    HIGH_TRAFFIC = 2
    NORMAL = 3
    LOW_TRAFFIC = 4

@dataclass
class VehicleMetrics:
    """Enhanced vehicle metrics with weights"""
    cars: int = 0
    trucks: int = 0
    buses: int = 0
    motorcycles: int = 0
    bicycles: int = 0
    timestamp: float = 0.0
    
    # Traffic weights for priority calculation
    WEIGHTS = {
        'cars': 2,
        'trucks': 4, 
        'buses': 3,
        'motorcycles': 1,
        'bicycles': 1
    }
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()
    
    def total_vehicles(self) -> int:
        """Total number of vehicles"""
        return self.cars + self.trucks + self.buses + self.motorcycles + self.bicycles
    
    def weighted_score(self) -> float:
        """Calculate weighted traffic score"""
        score = (
            self.cars * self.WEIGHTS['cars'] +
            self.trucks * self.WEIGHTS['trucks'] +
            self.buses * self.WEIGHTS['buses'] +
            self.motorcycles * self.WEIGHTS['motorcycles'] +
            self.bicycles * self.WEIGHTS['bicycles']
        )
        return float(score)
    
    def get_priority(self) -> Priority:
        """Determine traffic priority based on weighted score"""
        score = self.weighted_score()
        if score >= 20:
            return Priority.HIGH_TRAFFIC
        elif score >= 10:
            return Priority.NORMAL
        else:
            return Priority.LOW_TRAFFIC
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

class GracefulShutdown:
    """Graceful shutdown manager"""
    
    def __init__(self):
        self.shutdown_event = threading.Event()
        self.components = []
        self.shutdown_timeout = 30.0  # seconds
        self.shutdown_progress = 0
        self.shutdown_callback = None
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        atexit.register(self.shutdown)
    
    def register_component(self, component, name: str):
        """Register component for graceful shutdown"""
        self.components.append((component, name))
        logger.debug(f"Registered component for shutdown: {name}")
    
    def set_progress_callback(self, callback):
        """Set progress callback for UI updates"""
        self.shutdown_callback = callback
    
    def _signal_handler(self, signum, frame):
        """Handle system signals"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown")
        self.shutdown()
    
    def _update_progress(self, progress: int, message: str):
        """Update shutdown progress"""
        self.shutdown_progress = progress
        if self.shutdown_callback:
            self.shutdown_callback(progress, message)
        logger.info(f"Shutdown progress: {progress}% - {message}")
    
    def shutdown(self):
        """Perform graceful shutdown"""
        if self.shutdown_event.is_set():
            return
            
        self.shutdown_event.set()
        logger.info("Initiating graceful system shutdown...")
        
        try:
            self._update_progress(10, "Stopping detection engines...")
            self._shutdown_detection_engines()
            
            self._update_progress(30, "Closing camera streams...")
            self._shutdown_camera_streams()
            
            self._update_progress(50, "Disconnecting Arduino...")
            self._shutdown_arduino()
            
            self._update_progress(70, "Cleaning up resources...")
            self._cleanup_resources()
            
            self._update_progress(90, "Finalizing shutdown...")
            self._finalize_shutdown()
            
            self._update_progress(100, "Shutdown complete")
            logger.info("Graceful shutdown completed successfully")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        
        # Force exit if needed
        QTimer.singleShot(1000, lambda: QApplication.quit())
    
    def _shutdown_detection_engines(self):
        """Shutdown detection engines"""
        for component, name in self.components:
            if hasattr(component, 'stop') and 'detector' in name.lower():
                try:
                    component.stop()
                    logger.debug(f"Stopped {name}")
                except Exception as e:
                    logger.error(f"Error stopping {name}: {e}")
    
    def _shutdown_camera_streams(self):
        """Shutdown camera streams"""
        for component, name in self.components:
            if hasattr(component, 'stop') and 'camera' in name.lower():
                try:
                    component.stop()
                    logger.debug(f"Stopped {name}")
                except Exception as e:
                    logger.error(f"Error stopping {name}: {e}")
    
    def _shutdown_arduino(self):
        """Shutdown Arduino connection"""
        for component, name in self.components:
            if hasattr(component, 'emergency_shutdown') and 'arduino' in name.lower():
                try:
                    component.emergency_shutdown()
                    component.close()
                    logger.debug(f"Shutdown {name}")
                except Exception as e:
                    logger.error(f"Error shutting down {name}: {e}")
    
    def _cleanup_resources(self):
        """Clean up system resources"""
        try:
            # Clean up temporary files
            temp_files = ['*.tmp', '*.temp']
            for pattern in temp_files:
                import glob
                for file in glob.glob(pattern):
                    try:
                        os.remove(file)
                    except:
                        pass
        except Exception as e:
            logger.error(f"Resource cleanup error: {e}")
    
    def _finalize_shutdown(self):
        """Finalize shutdown process"""
        logger.info("System shutdown completed successfully")

# Global shutdown manager
shutdown_manager = GracefulShutdown()

class ArduinoController:
    """Enhanced Arduino controller with robust communication"""
    
    PIN_MAP = [(2, 3), (4, 5), (6, 7), (8, 9)]  # (RED, GREEN) for 4 routes
    
    def __init__(self):
        self.baudrate = 9600
        self.timeout = 2.0
        self.serial = None
        self.port = None
        self.connected = False
        self.lock = threading.RLock()
        self.last_command_time = 0
        self.command_delay = 0.1  # Minimum delay between commands
        self.connection_attempts = 0
        self.max_connection_attempts = 3
        
        # Register for shutdown
        shutdown_manager.register_component(self, "Arduino Controller")
    
    def find_arduino_port(self) -> Optional[str]:
        """Enhanced Arduino port detection"""
        ports = list(serial.tools.list_ports.comports())
        logger.debug(f"Scanning {len(ports)} serial ports for Arduino")
        
        arduino_indicators = [
            'arduino', 'ch340', 'ch341', 'usb serial', 'acm', 'usb2.0-serial'
        ]
        
        # First pass: Look for explicit Arduino indicators
        for port in ports:
            desc = f"{port.manufacturer or ''} {port.description or ''}".lower()
            device = port.device.lower()
            
            if any(indicator in desc or indicator in device for indicator in arduino_indicators):
                logger.info(f"Arduino candidate found: {port.device} ({port.description})")
                if self._test_port(port.device):
                    return port.device
        
        # Second pass: Test all available ports
        logger.warning("No explicit Arduino found, testing all ports...")
        for port in ports:
            if self._test_port(port.device):
                return port.device
        
        logger.error("No Arduino found on any serial port")
        return None
    
    def _test_port(self, port_name: str) -> bool:
        """Test if port has Arduino with correct sketch"""
        try:
            with serial.Serial(port_name, self.baudrate, timeout=self.timeout) as test_serial:
                time.sleep(2.5)  # Wait for Arduino reset
                
                # Look for "Ready" message
                start_time = time.time()
                while time.time() - start_time < 5:
                    if test_serial.in_waiting:
                        response = test_serial.readline().decode(errors='ignore').strip()
                        logger.debug(f"Port {port_name} response: {response}")
                        if "Ready" in response:
                            logger.info(f"Arduino confirmed on {port_name}")
                            return True
                    time.sleep(0.1)
                
        except Exception as e:
            logger.debug(f"Port {port_name} test failed: {e}")
        
        return False
    
    def connect(self) -> bool:
        """Connect to Arduino with retry logic"""
        with self.lock:
            if self.connected:
                return True
            
            self.connection_attempts += 1
            if self.connection_attempts > self.max_connection_attempts:
                logger.error("Maximum connection attempts exceeded")
                return False
            
            self.port = self.find_arduino_port()
            if not self.port:
                return False
            
            try:
                self.serial = serial.Serial(
                    self.port, 
                    self.baudrate, 
                    timeout=self.timeout,
                    write_timeout=self.timeout
                )
                
                time.sleep(2.5)  # Arduino reset delay
                
                # Wait for handshake
                if self._wait_for_handshake():
                    self.connected = True
                    self.connection_attempts = 0
                    logger.info(f"Arduino connected successfully on {self.port}")
                    return True
                else:
                    self.serial.close()
                    self.serial = None
                    
            except Exception as e:
                logger.error(f"Arduino connection error: {e}")
                if self.serial:
                    try:
                        self.serial.close()
                    except:
                        pass
                    self.serial = None
            
            return False
    
    def _wait_for_handshake(self) -> bool:
        """Wait for Arduino handshake with timeout"""
        start_time = time.time()
        while time.time() - start_time < 6:
            try:
                if self.serial.in_waiting:
                    response = self.serial.readline().decode(errors='ignore').strip()
                    logger.debug(f"Handshake response: {response}")
                    if "Ready" in response:
                        return True
            except Exception as e:
                logger.error(f"Handshake error: {e}")
                return False
            time.sleep(0.1)
        
        logger.error("Arduino handshake timeout")
        return False
    
    def send_command(self, pin: int, state: int) -> bool:
        """Send command with enhanced error handling"""
        if not self.connected or not self.serial:
            logger.warning("Arduino not connected - command ignored")
            return False
        
        # Rate limiting
        current_time = time.time()
        if current_time - self.last_command_time < self.command_delay:
            time.sleep(self.command_delay - (current_time - self.last_command_time))
        
        command = f"{pin}:{state}\n"
        max_retries = 3
        
        with self.lock:
            for attempt in range(max_retries):
                try:
                    self.serial.reset_input_buffer()
                    self.serial.write(command.encode())
                    self.serial.flush()
                    
                    # Wait for acknowledgment
                    start_time = time.time()
                    while time.time() - start_time < 1.5:
                        if self.serial.in_waiting:
                            response = self.serial.readline().decode(errors='ignore').strip()
                            if response.startswith("Set pin"):
                                self.last_command_time = time.time()
                                return True
                        time.sleep(0.05)
                    
                    logger.warning(f"No ACK for {command.strip()}, attempt {attempt + 1}")
                    
                except Exception as e:
                    logger.error(f"Command send error (attempt {attempt + 1}): {e}")
                    if attempt == max_retries - 1:
                        self.connected = False
                        return False
                    time.sleep(0.2)
            
            logger.error(f"Failed to send command after {max_retries} attempts: {command.strip()}")
            return False
    
    def set_traffic_lights(self, active_route: int, total_routes: int) -> bool:
        """Set traffic lights with validation and feedback"""
        if not self.connected:
            logger.warning("Cannot control traffic lights - Arduino disconnected")
            return False
        
        if active_route >= total_routes or active_route < 0:
            logger.error(f"Invalid route index: {active_route} (max: {total_routes-1})")
            return False
        
        success_count = 0
        total_commands = 0
        
        # Set lights for active routes
        for idx, (red_pin, green_pin) in enumerate(self.PIN_MAP[:total_routes]):
            if idx == active_route:
                # Green route
                if self.send_command(red_pin, 0):  # RED OFF
                    success_count += 1
                total_commands += 1
                
                if self.send_command(green_pin, 1):  # GREEN ON
                    success_count += 1
                total_commands += 1
            else:
                # Red routes
                if self.send_command(red_pin, 1):  # RED ON
                    success_count += 1
                total_commands += 1
                
                if self.send_command(green_pin, 0):  # GREEN OFF
                    success_count += 1
                total_commands += 1
        
        # Turn off unused pins
        for red_pin, green_pin in self.PIN_MAP[total_routes:]:
            if self.send_command(red_pin, 0):
                success_count += 1
            total_commands += 1
            
            if self.send_command(green_pin, 0):
                success_count += 1
            total_commands += 1
        
        success_rate = success_count / total_commands if total_commands > 0 else 0
        logger.info(f"Traffic light update: Route {active_route + 1} active, success rate: {success_rate:.1%}")
        
        return success_rate > 0.8  # Consider successful if >80% commands succeeded
    
    def emergency_shutdown(self):
        """Emergency shutdown - all lights to safe state"""
        logger.warning("Emergency shutdown - setting all lights to RED")
        try:
            for red_pin, green_pin in self.PIN_MAP:
                self.send_command(red_pin, 1)  # RED ON
                self.send_command(green_pin, 0)  # GREEN OFF
        except Exception as e:
            logger.error(f"Emergency shutdown error: {e}")
    
    def close(self):
        """Close Arduino connection gracefully"""
        with self.lock:
            if self.connected and self.serial:
                try:
                    self.emergency_shutdown()
                    time.sleep(0.5)  # Allow commands to complete
                    self.serial.close()
                    logger.info("Arduino connection closed")
                except Exception as e:
                    logger.error(f"Error closing Arduino: {e}")
                finally:
                    self.connected = False
                    self.serial = None

class AIModels:
    """AI Models manager with enhanced loading and error recovery"""
    
    def __init__(self):
        self.yolo = None
        self.ocr = None
        self.cuda_available = False
        self.models_loaded = False
        self.model_path = "yolov8s.pt"
        self.load_attempts = 0
        self.max_load_attempts = 3
    
    def load_models(self, model_path: str = None) -> bool:
        """Load AI models with comprehensive error handling"""
        if model_path:
            self.model_path = model_path
        
        self.load_attempts += 1
        if self.load_attempts > self.max_load_attempts:
            logger.error("Maximum model loading attempts exceeded")
            return False
        
        try:
            logger.info(f"Loading AI models (attempt {self.load_attempts})...")
            
            # Check CUDA availability
            self.cuda_available = torch.cuda.is_available()
            if self.cuda_available:
                torch.cuda.empty_cache()  # Clear GPU memory
                logger.info(f"CUDA available: {torch.cuda.get_device_name()}")
            else:
                logger.info("Using CPU inference")
            
            # Load YOLO model
            if os.path.exists(self.model_path):
                self.yolo = YOLO(self.model_path)
                # Warm up the model
                dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
                _ = self.yolo(dummy_frame, device='cuda' if self.cuda_available else 'cpu', verbose=False)
                logger.info(f"YOLO model loaded and warmed up: {self.model_path}")
            else:
                logger.error(f"YOLO model file not found: {self.model_path}")
                return False
            
            # Load OCR
            self.ocr = easyocr.Reader(['en'], gpu=self.cuda_available, verbose=False)
            logger.info("EasyOCR loaded successfully")
            
            self.models_loaded = True
            self.load_attempts = 0
            logger.info("All AI models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load AI models: {e}")
            # Clean up partial loads
            self.yolo = None
            self.ocr = None
            self.models_loaded = False
            return False
    
    def is_ready(self) -> bool:
        """Check if models are ready for inference"""
        return self.models_loaded and self.yolo is not None

class EnhancedDetectionEngine(QThread):
    """Enhanced detection engine with fixed priority logic"""
    
    detection_result = pyqtSignal(int, dict)
    error_occurred = pyqtSignal(int, str)
    
    def __init__(self, camera_id: int, ai_models: AIModels, settings: Dict):
        super().__init__()
        self.camera_id = camera_id
        self.ai_models = ai_models
        self.settings = settings
        self.enable_ocr = settings.get('enable_ocr', True)
        self.ocr_threshold = settings.get('ocr_threshold', 0.5)
        self.enable_audio = settings.get('enable_audio', True)
        self.siren_threshold = settings.get('siren_threshold', 0.08)
        self.confidence_threshold = settings.get('confidence_threshold', 0.5)
        self.running = False
        self.detection_enabled = True
        self.frame_queue = queue.Queue(maxsize=3)  # Smaller queue for better performance
        
        # Detection parameters
        self.vehicle_classes = {1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
        self.emergency_keywords = ['ambulance', 'fire', 'police', 'emergency', 'rescue', 'medical']
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        
        # Detection history for smoother results
        self.detection_history = []
        self.history_size = 5
        
        # Current detection state
        self.current_metrics = VehicleMetrics()
        self.emergency_detected = False
        self.siren_detected = False
        
        # Register for shutdown
        shutdown_manager.register_component(self, f"Detection Engine {camera_id}")
        
        logger.info(f"Detection engine initialized for camera {camera_id}")
    
    def set_siren_status(self, detected: bool):
        """Update siren detection status"""
        self.siren_detected = detected
        if detected:
            logger.info(f"Siren detected on camera {self.camera_id}")
    
    def process_frame(self, frame: np.ndarray):
        """Add frame to processing queue"""
        if not self.running or not self.detection_enabled:
            return
        
        try:
            # Non-blocking put - drop oldest frame if queue is full
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            
            self.frame_queue.put_nowait(frame.copy())
        except queue.Full:
            pass  # Skip frame if queue is full
    
    def run(self):
        """Main detection loop with enhanced error handling"""
        self.running = True
        logger.info(f"Detection engine started for camera {self.camera_id}")
        
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.5)
                
                if not self.detection_enabled or not self.ai_models.is_ready():
                    continue
                
                # Perform detection
                result = self._detect_objects(frame)
                
                # Update FPS
                self.fps_counter += 1
                current_time = time.time()
                if current_time - self.fps_start_time >= 1.0:
                    self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
                    self.fps_counter = 0
                    self.fps_start_time = current_time
                
                result['fps'] = self.current_fps
                result['camera_id'] = self.camera_id
                
                # Emit result
                self.detection_result.emit(self.camera_id, result)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Detection error for camera {self.camera_id}: {e}")
                self.error_occurred.emit(self.camera_id, str(e))
                time.sleep(0.1)  # Brief pause on error
    
    def _detect_objects(self, frame: np.ndarray) -> Dict[str, Any]:
        """Enhanced object detection with proper metrics calculation"""
        result = {
            'vehicle_metrics': VehicleMetrics(),
            'emergency_detected': False,
            'siren_detected': self.siren_detected,
            'annotated_frame': frame.copy(),
            'detection_confidence': 0.0,
            'processing_time': 0.0,
            'priority': Priority.LOW_TRAFFIC
        }
        
        start_time = time.time()
        
        try:
            if not self.ai_models.yolo:
                return result
            
            # YOLO Detection
            device = 'cuda' if self.ai_models.cuda_available else 'cpu'
            detections = self.ai_models.yolo(
                frame, 
                device=device, 
                verbose=False,
                conf=self.confidence_threshold
            )
            
            vehicle_counts = VehicleMetrics()
            confidences = []
            emergency_detected = False
            
            for detection in detections:
                boxes = detection.boxes
                if boxes is None or len(boxes) == 0:
                    continue
                
                class_ids = boxes.cls.cpu().numpy()
                coordinates = boxes.xyxy.cpu().numpy()
                confidence_scores = boxes.conf.cpu().numpy()
                
                for class_id, coord, conf in zip(class_ids, coordinates, confidence_scores):
                    if conf < self.confidence_threshold:
                        continue
                    
                    vehicle_type = self.vehicle_classes.get(int(class_id))
                    if not vehicle_type:
                        continue
                    
                    # Count vehicles by type
                    if vehicle_type == 'car':
                        vehicle_counts.cars += 1
                    elif vehicle_type == 'truck':
                        vehicle_counts.trucks += 1
                    elif vehicle_type == 'bus':
                        vehicle_counts.buses += 1
                    elif vehicle_type == 'motorcycle':
                        vehicle_counts.motorcycles += 1
                    elif vehicle_type == 'bicycle':
                        vehicle_counts.bicycles += 1
                    
                    confidences.append(float(conf))
                    
                    # Draw bounding box
                    x1, y1, x2, y2 = map(int, coord[:4])
                    color = (0, 255, 0) if not emergency_detected else (0, 0, 255)
                    cv2.rectangle(result['annotated_frame'], (x1, y1), (x2, y2), color, 2)
                    
                    # Add label with confidence
                    label = f"{vehicle_type} {conf:.2f}"
                    cv2.putText(result['annotated_frame'], label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # Emergency vehicle detection via OCR
                    if (self.ai_models.ocr and 
                        vehicle_type in ['car', 'bus', 'truck'] and 
                        y2 > y1 and x2 > x1):
                        
                        try:
                            roi = frame[max(0, y1):min(frame.shape[0], y2), 
                                       max(0, x1):min(frame.shape[1], x2)]
                            
                            if roi.size > 100:  # Minimum ROI size
                                texts = self.ai_models.ocr.readtext(roi, detail=0)
                                for text in texts:
                                    text_lower = text.lower()
                                    if any(keyword in text_lower for keyword in self.emergency_keywords):
                                        emergency_detected = True
                                        # Highlight emergency vehicle
                                        cv2.rectangle(result['annotated_frame'], (x1, y1), (x2, y2), (0, 0, 255), 4)
                                        cv2.putText(result['annotated_frame'], "EMERGENCY", 
                                                   (x1, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 
                                                   0.8, (0, 0, 255), 2)
                                        logger.info(f"Emergency vehicle detected: {text}")
                                        break
                        except Exception as ocr_e:
                            logger.debug(f"OCR processing error: {ocr_e}")
            
            # Update result with detection data
            result['vehicle_metrics'] = vehicle_counts
            result['emergency_detected'] = emergency_detected or self.siren_detected
            result['detection_confidence'] = np.mean(confidences) if confidences else 0.0
            result['priority'] = self._calculate_priority(vehicle_counts, emergency_detected or self.siren_detected)
            
            # Add siren indicator to frame
            if self.siren_detected:
                cv2.putText(result['annotated_frame'], "SIREN DETECTED", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                result['emergency_detected'] = True
            
            # Store current metrics
            self.current_metrics = vehicle_counts
            self.emergency_detected = emergency_detected or self.siren_detected
            
            # Add to detection history for smoothing
            self.detection_history.append(vehicle_counts)
            if len(self.detection_history) > self.history_size:
                self.detection_history.pop(0)
            
        except Exception as e:
            logger.error(f"Object detection processing error: {e}")
        
        result['processing_time'] = time.time() - start_time
        return result
    
    def _calculate_priority(self, metrics: VehicleMetrics, emergency: bool) -> Priority:
        """Calculate traffic priority based on metrics and emergency status"""
        if emergency:
            return Priority.EMERGENCY
        
        return metrics.get_priority()
    
    def get_smoothed_metrics(self) -> VehicleMetrics:
        """Get smoothed vehicle metrics from history"""
        if not self.detection_history:
            return VehicleMetrics()
        
        # Average the recent detections
        avg_cars = sum(m.cars for m in self.detection_history) / len(self.detection_history)
        avg_trucks = sum(m.trucks for m in self.detection_history) / len(self.detection_history)
        avg_buses = sum(m.buses for m in self.detection_history) / len(self.detection_history)
        avg_motorcycles = sum(m.motorcycles for m in self.detection_history) / len(self.detection_history)
        avg_bicycles = sum(m.bicycles for m in self.detection_history) / len(self.detection_history)
        
        return VehicleMetrics(
            cars=int(avg_cars),
            trucks=int(avg_trucks),
            buses=int(avg_buses),
            motorcycles=int(avg_motorcycles),
            bicycles=int(avg_bicycles)
        )
    
    def enable_detection(self, enabled: bool):
        """Enable/disable detection"""
        self.detection_enabled = enabled
        status = "enabled" if enabled else "disabled"
        logger.info(f"Detection {status} for camera {self.camera_id}")
    
    def stop(self):
        """Stop detection engine gracefully"""
        logger.info(f"Stopping detection engine for camera {self.camera_id}")
        self.running = False
        
        # Clear queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        
        self.wait(3000)  # Wait up to 3 seconds for thread to finish
        if self.isRunning():
            logger.warning(f"Force terminating detection engine {self.camera_id}")
            self.terminate()
        
        logger.info(f"Detection engine stopped for camera {self.camera_id}")

class EnhancedCameraStream(QThread):
    """Enhanced camera stream with robust error handling"""
    
    frame_ready = pyqtSignal(int, np.ndarray)
    error_occurred = pyqtSignal(int, str)
    status_changed = pyqtSignal(int, str)
    
    def __init__(self, camera_id: int, source, name: str):
        super().__init__()
        self.camera_id = camera_id
        self.source = source
        self.name = name
        self.running = False
        
        # Camera properties
        self.cap = None
        self.frame_lock = threading.RLock()
        self.latest_frame = None
        self.frame_count = 0
        self.error_count = 0
        self.max_consecutive_errors = 10
        
        # Performance tracking
        self.fps_target = 30
        self.frame_interval = 1.0 / self.fps_target
        self.last_frame_time = 0
        
        # Audio/siren detection
        self.audio_stream = None
        self.siren_detector = None
        self.enable_audio = True
        
        # Register for shutdown
        shutdown_manager.register_component(self, f"Camera Stream {camera_id}")
        
        logger.info(f"Camera stream initialized: {name}")
    
    def setup_siren_detection(self, detection_engine):
        """Setup siren detection for this camera"""
        if not self.enable_audio:
            return
        
        try:
            def audio_callback(indata, frames, time_info, status):
                if self.running and detection_engine:
                    # Simple RMS-based siren detection
                    rms = np.sqrt(np.mean(indata[:, 0] ** 2))
                    detected = rms > 0.08  # Adjustable threshold
                    detection_engine.set_siren_status(detected)
            
            self.audio_stream = sd.InputStream(
                channels=1,
                samplerate=44100,
                blocksize=1024,
                callback=audio_callback
            )
            
            logger.info(f"Siren detection setup for camera {self.camera_id}")
            
        except Exception as e:
            logger.warning(f"Could not setup siren detection for camera {self.camera_id}: {e}")
            self.enable_audio = False
    
    def run(self):
        """Main camera streaming loop with dynamic FPS management"""
        self.running = True
        logger.info(f"Starting camera stream: {self.name}")

        # Initialize camera (directly with OpenCV)
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap or not self.cap.isOpened():
            self.error_occurred.emit(self.camera_id, f"Failed to initialize camera: {self.source}")
            return

        # Start audio stream if enabled
        if self.enable_audio and self.audio_stream:
            try:
                self.audio_stream.start()
                logger.debug(f"Audio stream started for camera {self.camera_id}")
            except Exception as e:
                logger.warning(f"Failed to start audio stream: {e}")

        self.status_changed.emit(self.camera_id, "Running")

        # Dynamic FPS management variables
        min_interval = 1.0 / 60  # Max 60 FPS
        max_interval = 1.0 / 5   # Min 5 FPS

        while self.running:
            try:
                start_time = time.time()
                ret, frame = self.cap.read()

                if ret and frame is not None:
                    with self.frame_lock:
                        self.latest_frame = frame.copy()

                    self.frame_ready.emit(self.camera_id, frame)
                    self.frame_count += 1
                    self.error_count = 0
                    self.last_frame_time = start_time

                else:
                    self.error_count += 1
                    if self.error_count > self.max_consecutive_errors:
                        error_msg = f"Too many consecutive read errors from {self.name}"
                        logger.error(error_msg)
                        self.error_occurred.emit(self.camera_id, error_msg)
                        break
                    time.sleep(0.1)

                # Dynamic FPS: sleep based on processing time
                elapsed = time.time() - start_time
                sleep_time = max(min_interval, min(max_interval, elapsed))
                time.sleep(sleep_time)

            except Exception as e:
                logger.error(f"Camera {self.camera_id} streaming error: {e}")
                self.error_count += 1
                if self.error_count > self.max_consecutive_errors:
                    self.error_occurred.emit(self.camera_id, str(e))
                    break
                time.sleep(0.5)

        self._cleanup()
        logger.info(f"Camera stream stopped: {self.name}")

    def stop(self):
        """Stop the camera stream and release resources"""
        self.running = False
        self.wait()  # Wait for the thread to finish
        self._cleanup()

    def _cleanup(self):
        """Release camera resources"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        logger.info(f"Camera {self.camera_id} resources released.")

class EnhancedVideoWidget(QWidget):
    """Enhanced video widget with professional styling"""
    
    def __init__(self, camera_id: int, title: str):
        super().__init__()
        self.camera_id = camera_id
        self.title = title
        self.current_frame = None
        self.is_active_route = False
        self.detection_data = {}
        self.setup_ui()
        
    def setup_ui(self):
        """Setup enhanced UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        # Header with route status
        header_layout = QHBoxLayout()
        
        self.title_label = QLabel(self.title)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("""
            QLabel {
                font-weight: bold;
                font-size: 16px;
                color: #2E8B57;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #F0F8FF, stop:1 #E6F3FF);
                padding: 8px;
                border-radius: 6px;
                border: 2px solid #2E8B57;
            }
        """)
        
        self.status_indicator = QLabel("●")
        self.status_indicator.setFixedSize(20, 20)
        self.status_indicator.setAlignment(Qt.AlignCenter)
        self.status_indicator.setStyleSheet("color: #ff6b6b; font-size: 16px; font-weight: bold;")
        
        header_layout.addWidget(self.title_label)
        header_layout.addWidget(self.status_indicator)
        layout.addLayout(header_layout)
        
        # Video display with enhanced styling
        self.video_label = QLabel("Camera Initializing...")
        self.video_label.setFixedSize(500, 340)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("""
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #1a1a1a, stop:1 #2d2d2d);
                border: 3px solid #404040;
                border-radius: 10px;
                color: #ffffff;
                font-size: 14px;
                font-weight: bold;
            }
        """)
        layout.addWidget(self.video_label)
        
        # Enhanced statistics panel
        self.setup_stats_panel(layout)
        
    def setup_stats_panel(self, layout):
        """Setup enhanced statistics panel"""
        stats_frame = QFrame()
        stats_frame.setFrameStyle(QFrame.StyledPanel)
        stats_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f8f9fa, stop:1 #e9ecef);
                border: 1px solid #dee2e6;
                border-radius: 8px;
                padding: 8px;
            }
        """)
        
        stats_layout = QVBoxLayout(stats_frame)
        stats_layout.setSpacing(6)
        
        # Vehicle counts with icons
        counts_layout = QHBoxLayout()
        self.counts_labels = {
            'cars': QLabel("🚗 0"),
            'trucks': QLabel("🚛 0"), 
            'buses': QLabel("🚌 0"),
            'motorcycles': QLabel("🏍️ 0"),
            'bicycles': QLabel("🚴 0")
        }
        
        for label in self.counts_labels.values():
            label.setStyleSheet("font-weight: bold; color: #495057; font-size: 12px;")
            counts_layout.addWidget(label)
        
        stats_layout.addLayout(counts_layout)
        
        # Performance and status info
        perf_layout = QHBoxLayout()
        
        self.fps_label = QLabel("FPS: 0.0")
        self.fps_label.setStyleSheet("color: #6c757d; font-weight: bold;")
        
        self.weight_label = QLabel("Weight: 0")
        self.weight_label.setStyleSheet("color: #6c757d; font-weight: bold;")
        
        self.priority_label = QLabel("Priority: LOW")
        self.priority_label.setStyleSheet("color: #6c757d; font-weight: bold;")
        
        self.emergency_label = QLabel("")
        self.emergency_label.setStyleSheet("color: #dc3545; font-weight: bold; font-size: 14px;")
        
        perf_layout.addWidget(self.fps_label)
        perf_layout.addWidget(self.weight_label)
        perf_layout.addWidget(self.priority_label)
        perf_layout.addStretch()
        perf_layout.addWidget(self.emergency_label)
        
        stats_layout.addLayout(perf_layout)
        layout.addWidget(stats_frame)
    
    def update_frame(self, frame: np.ndarray):
        """Update video frame with enhanced processing"""
        try:
            # Convert and scale frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            
            # Scale to fit widget with aspect ratio
            scaled_pixmap = pixmap.scaled(
                self.video_label.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            
            self.video_label.setPixmap(scaled_pixmap)
            self.current_frame = frame
            
        except Exception as e:
            logger.error(f"Error updating frame for camera {self.camera_id}: {e}")
    
    def update_detection_data(self, detection_data: Dict[str, Any]):
        """Update detection information with enhanced display"""
        try:
            self.detection_data = detection_data
            
            # Update vehicle counts
            metrics = detection_data.get('vehicle_metrics', VehicleMetrics())
            self.counts_labels['cars'].setText(f"🚗 {metrics.cars}")
            self.counts_labels['trucks'].setText(f"🚛 {metrics.trucks}")
            self.counts_labels['buses'].setText(f"🚌 {metrics.buses}")
            self.counts_labels['motorcycles'].setText(f"🏍️ {metrics.motorcycles}")
            self.counts_labels['bicycles'].setText(f"🚴 {metrics.bicycles}")
            
            # Update performance metrics
            fps = detection_data.get('fps', 0.0)
            self.fps_label.setText(f"FPS: {fps:.1f}")
            
            weight = metrics.weighted_score()
            self.weight_label.setText(f"Weight: {weight:.1f}")
            
            priority = detection_data.get('priority', Priority.LOW_TRAFFIC)
            priority_color = {
                Priority.EMERGENCY: "#dc3545",
                Priority.HIGH_TRAFFIC: "#fd7e14", 
                Priority.NORMAL: "#28a745",
                Priority.LOW_TRAFFIC: "#6c757d"
            }
            
            self.priority_label.setText(f"Priority: {priority.name.replace('_', ' ')}")
            self.priority_label.setStyleSheet(f"color: {priority_color[priority]}; font-weight: bold;")
            
            # Emergency status
            emergency = detection_data.get('emergency_detected', False)
            siren = detection_data.get('siren_detected', False)
            
            if emergency or siren:
                emergency_text = "🚨 EMERGENCY"
                if siren:
                    emergency_text += " + SIREN"
                self.emergency_label.setText(emergency_text)
                self.emergency_label.setStyleSheet("color: #dc3545; font-weight: bold; font-size: 14px; background-color: #fff5f5; padding: 4px; border-radius: 4px;")
            else:
                self.emergency_label.setText("")
                self.emergency_label.setStyleSheet("")
            
        except Exception as e:
            logger.error(f"Error updating detection data for camera {self.camera_id}: {e}")
    
    def set_active_status(self, is_active: bool):
        """Set active route status with enhanced visuals"""
        self.is_active_route = is_active
        
        if is_active:
            # Green/active styling
            self.video_label.setStyleSheet("""
                QLabel {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                        stop:0 #1a1a1a, stop:1 #2d2d2d);
                    border: 4px solid #28a745;
                    border-radius: 10px;
                    color: #ffffff;
                    font-size: 14px;
                    font-weight: bold;
                }
            """)
            
            self.title_label.setStyleSheet("""
                QLabel {
                    font-weight: bold;
                    font-size: 16px;
                    color: white;
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #28a745, stop:1 #20c997);
                    padding: 8px;
                    border-radius: 6px;
                    border: 2px solid #28a745;
                }
            """)
            
            self.status_indicator.setStyleSheet("color: #28a745; font-size: 16px; font-weight: bold;")
            
        else:
            # Red/inactive styling
            self.video_label.setStyleSheet("""
                QLabel {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                        stop:0 #1a1a1a, stop:1 #2d2d2d);
                    border: 3px solid #dc3545;
                    border-radius: 10px;
                    color: #ffffff;
                    font-size: 14px;
                    font-weight: bold;
                }
            """)
            
            self.title_label.setStyleSheet("""
                QLabel {
                    font-weight: bold;
                    font-size: 16px;
                    color: #2E8B57;
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #F0F8FF, stop:1 #E6F3FF);
                    padding: 8px;
                    border-radius: 6px;
                    border: 2px solid #2E8B57;
                }
            """)
            
            self.status_indicator.setStyleSheet("color: #dc3545; font-size: 16px; font-weight: bold;")

class ShutdownDialog(QDialog):
    """Professional shutdown progress dialog"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("System Shutdown")
        self.setModal(True)
        self.setFixedSize(400, 150)
        self.setup_ui()
        
    def setup_ui(self):
        """Setup shutdown dialog UI"""
        layout = QVBoxLayout(self)
        
        self.status_label = QLabel("Preparing for shutdown...")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        
        self.detail_label = QLabel("")
        self.detail_label.setAlignment(Qt.AlignCenter)
        self.detail_label.setStyleSheet("color: #666; font-size: 10px;")
        layout.addWidget(self.detail_label)
    
    def update_progress(self, progress: int, message: str):
        """Update shutdown progress"""
        self.progress_bar.setValue(progress)
        self.status_label.setText(message)
        self.detail_label.setText(f"Progress: {progress}%")
        QApplication.processEvents()

class EnhancedSetupWindow(QWidget):
    """Enhanced setup window with validation"""
    
    def __init__(self):
        super().__init__()
        self.camera_type = "USB"
        self.camera_urls = []
        self.settings = {
            'enable_audio': True,
            'enable_ocr': True,
            'confidence_threshold': 0.5,
            'detection_interval': 1.0
        }
        self.setup_ui()
        
    def setup_ui(self):
        """Setup enhanced UI"""
        self.setWindowTitle("Smart Traffic Management System - Setup")
        self.setMinimumSize(1000, 700)
        self.resize(1600, 1000)
        
        # Main layout for QWidget (not QMainWindow)
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(20)
        
        # Header
        self.setup_header(main_layout)
        
        # --- Add this block ---
        self.system_status_label = QLabel("INITIALIZING")
        self.system_status_label.setAlignment(Qt.AlignCenter)
        self.system_status_label.setStyleSheet("""
            QLabel {
                color: #ffc107;
                font-size: 18px;
                font-weight: bold;
                background: transparent;
            }
        """)
        main_layout.addWidget(self.system_status_label)
        # --- End block ---

        # Camera configuration
        self.setup_camera_config(main_layout)
        
        # System settings
        self.setup_system_settings(main_layout)
        
        # Action buttons
        self.setup_action_buttons(main_layout)
        
        # Developer credit
        credit_label = QLabel("Developed by Er_NEERAJ VERMA")
        credit_label.setAlignment(Qt.AlignRight)
        credit_label.setStyleSheet("color: #764ba2; font-size: 13px; font-weight: bold; margin-top: 10px;")
        main_layout.addWidget(credit_label)
    
    def setup_header(self, layout):
        """Setup header section with READY indicator on right"""
        header_frame = QFrame()
        header_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #667eea, stop:1 #764ba2);
                border-radius: 15px;
                padding: 20px;
            }
        """)
        header_layout = QHBoxLayout(header_frame)

        # Title and subtitle (left side)
        title_sub_layout = QVBoxLayout()
        title = QLabel("Smart Traffic Management System")
        title.setAlignment(Qt.AlignLeft)
        title.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 26px;
                font-weight: bold;
                background: transparent;
            }
        """)
        subtitle = QLabel("Professional AI-Driven Traffic Control")
        subtitle.setAlignment(Qt.AlignLeft)
        subtitle.setStyleSheet("""
            QLabel {
                color: rgba(255, 255, 255, 0.8);
                font-size: 14px;
                background: transparent;
            }
        """)
        title_sub_layout.addWidget(title)
        title_sub_layout.addWidget(subtitle)
        header_layout.addLayout(title_sub_layout)

        # READY indicator (right side)
        self.system_status_label = QLabel("INITIALIZING")
        self.system_status_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.system_status_label.setMinimumWidth(180)
        self.system_status_label.setStyleSheet("""
            QLabel {
                color: #ffc107;
                font-size: 18px;
                font-weight: bold;
                background: transparent;
            }
        """)
        header_layout.addWidget(self.system_status_label)

        layout.addWidget(header_frame)
    
    def setup_camera_config(self, layout):
        """Setup camera configuration section"""
        camera_group = QGroupBox("Camera Configuration")
        camera_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 14px; }")
        camera_layout = QVBoxLayout(camera_group)
        
        # Camera type selection
        type_layout = QHBoxLayout()
        self.usb_radio = QRadioButton("USB Cameras (Recommended)")
        self.usb_radio.setChecked(True)
        self.usb_radio.toggled.connect(self.on_camera_type_changed)
        
        self.ip_radio = QRadioButton("IP/RTSP Cameras")
        self.ip_radio.toggled.connect(self.on_camera_type_changed)
        
        type_layout.addWidget(self.usb_radio)
        type_layout.addWidget(self.ip_radio)
        camera_layout.addLayout(type_layout)
        
        # Number of cameras
        count_layout = QHBoxLayout()
        count_layout.addWidget(QLabel("Number of Traffic Routes (1-4):"))
        
        self.camera_count_spin = QSpinBox()
        self.camera_count_spin.setRange(1, 4)
        self.camera_count_spin.setValue(2)
        self.camera_count_spin.valueChanged.connect(self.on_camera_count_changed)
        count_layout.addWidget(self.camera_count_spin)
        count_layout.addStretch()
        
        camera_layout.addLayout(count_layout)
        
        # IP camera URLs (hidden by default)
        self.ip_group = QGroupBox("IP Camera URLs (RTSP Format)")
        self.ip_group.setVisible(False)
        self.ip_layout = QFormLayout(self.ip_group)
        camera_layout.addWidget(self.ip_group)
        
        layout.addWidget(camera_group)
    
    def setup_system_settings(self, layout):
        """Setup system settings section"""
        settings_group = QGroupBox("System Settings")
        settings_layout = QVBoxLayout(settings_group)

        # Detection settings
        detection_frame = QFrame()
        detection_layout = QVBoxLayout(detection_frame)

        # Siren detection ON/OFF
        self.enable_audio = QCheckBox("Enable Siren Detection (Requires Microphone)")
        self.enable_audio.setChecked(self.settings.get('enable_audio', True))
        detection_layout.addWidget(self.enable_audio)

        # Siren detection threshold slider
        siren_layout = QHBoxLayout()
        siren_layout.addWidget(QLabel("Siren Detection Threshold:"))
        self.siren_slider = QSlider(Qt.Horizontal)
        self.siren_slider.setRange(1, 100)
        self.siren_slider.setValue(int(self.settings.get('siren_threshold', 8)))
        self.siren_slider.valueChanged.connect(self.on_siren_threshold_changed)
        siren_layout.addWidget(self.siren_slider)
        self.siren_label = QLabel(f"{self.siren_slider.value()/100:.2f}")
        siren_layout.addWidget(self.siren_label)
        detection_layout.addLayout(siren_layout)

        # OCR detection ON/OFF
        self.enable_ocr = QCheckBox("Enable Emergency Vehicle Text Recognition (OCR)")
        self.enable_ocr.setChecked(self.settings.get('enable_ocr', True))
        detection_layout.addWidget(self.enable_ocr)

        # OCR detection threshold slider
        ocr_layout = QHBoxLayout()
        ocr_layout.addWidget(QLabel("OCR Detection Threshold:"))
        self.ocr_slider = QSlider(Qt.Horizontal)
        self.ocr_slider.setRange(1, 100)
        self.ocr_slider.setValue(int(self.settings.get('ocr_threshold', 50)))
        self.ocr_slider.valueChanged.connect(self.on_ocr_threshold_changed)
        ocr_layout.addWidget(self.ocr_slider)
        self.ocr_label = QLabel(f"{self.ocr_slider.value()/100:.2f}")
        ocr_layout.addWidget(self.ocr_label)
        detection_layout.addLayout(ocr_layout)

        # Confidence threshold (existing)
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("Detection Confidence Threshold:"))
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setRange(30, 90)
        self.confidence_slider.setValue(int(self.settings['confidence_threshold'] * 100))
        self.confidence_slider.valueChanged.connect(self.on_confidence_changed)
        conf_layout.addWidget(self.confidence_slider)
        self.confidence_label = QLabel("50%")
        conf_layout.addWidget(self.confidence_label)
        detection_layout.addLayout(conf_layout)

        settings_layout.addWidget(detection_frame)
        layout.addWidget(settings_group)

    def on_siren_threshold_changed(self, value):
        """Handle siren threshold change"""
        threshold = value / 100.0
        self.settings['siren_threshold'] = threshold
        self.siren_label.setText(f"{threshold:.2f}")

    def on_ocr_threshold_changed(self, value):
        """Handle OCR threshold change"""
        threshold = value / 100.0
        self.settings['ocr_threshold'] = threshold
        self.ocr_label.setText(f"{threshold:.2f}")

    def setup_action_buttons(self, layout):
        """Setup action buttons"""
        button_layout = QHBoxLayout()
        
        self.validate_btn = QPushButton("Validate Configuration")
        self.validate_btn.clicked.connect(self.validate_configuration)
        self.validate_btn.setStyleSheet(self.get_button_style("#17a2b8"))
        
        self.test_btn = QPushButton("Test Cameras")
        self.test_btn.clicked.connect(self.test_cameras)
        self.test_btn.setStyleSheet(self.get_button_style("#28a745"))
        
        self.start_btn = QPushButton("Start System")
        self.start_btn.clicked.connect(self.start_system)
        self.start_btn.setStyleSheet(self.get_button_style("#007bff"))
        
        button_layout.addWidget(self.validate_btn)
        button_layout.addWidget(self.test_btn)
        button_layout.addStretch()
        button_layout.addWidget(self.start_btn)
        
        layout.addLayout(button_layout)
    
    def get_button_style(self, color: str) -> str:
        """Get button stylesheet"""
        return f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 14px;
                min-width: 120px;
            }}
            QPushButton:hover {{
                background-color: {color}dd;
            }}
            QPushButton:pressed {{
                background-color: {color}bb;
            }}
        """
    
    def on_camera_type_changed(self):
        """Handle camera type change"""
        self.camera_type = "USB" if self.usb_radio.isChecked() else "IP"
        self.ip_group.setVisible(self.camera_type == "IP")
        if self.camera_type == "IP":
            self.on_camera_count_changed()
    
    def on_camera_count_changed(self):
        """Handle camera count change"""
        # Clear existing IP fields
        while self.ip_layout.count():
            self.ip_layout.removeRow(0)
        
        if self.camera_type == "IP":
            count = self.camera_count_spin.value()
            self.camera_urls = [""] * count
            
            for i in range(count):
                button = QPushButton(f"Configure Camera {i + 1}")
                button.clicked.connect(lambda checked, idx=i: self.configure_ip_camera(idx))
                self.ip_layout.addRow(f"Route {i + 1}:", button)
    
    def on_confidence_changed(self, value):
        """Handle confidence threshold change"""
        confidence = value / 100.0
        self.settings['confidence_threshold'] = confidence
        self.confidence_label.setText(f"{value}%")
    
    def configure_ip_camera(self, index: int):
        """Configure IP camera URL"""
        current_url = self.camera_urls[index] if index < len(self.camera_urls) else ""
        
        url, ok = QInputDialog.getText(
            self,
            f"Configure Camera {index + 1}",
            "Enter RTSP URL:\n(Example: rtsp://192.168.1.100:554/stream1)",
            text=current_url
        )
        
        if ok and url.strip():
            # Basic URL validation
            if not (url.startswith('rtsp://') or url.startswith('http://')):
                QMessageBox.warning(self, "Invalid URL", "URL must start with rtsp:// or http://")
                return
            
            self.camera_urls[index] = url.strip()
            button = self.ip_layout.itemAt(index, QFormLayout.FieldRole).widget()
            display_url = url.strip()
            if len(display_url) > 40:
                display_url = display_url[:37] + "..."
            button.setText(display_url)
    
    def validate_configuration(self):
        """Validate system configuration"""
        try:
            issues = []
            
            # Check camera configuration
            if self.camera_type == "IP":
                count = self.camera_count_spin.value()
                if len(self.camera_urls) < count or any(not url for url in self.camera_urls[:count]):
                    issues.append("Not all IP camera URLs are configured")
            
            # Check AI models
            if not os.path.exists("yolov8s.pt"):
                issues.append("YOLO model file (yolov8s.pt) not found")
            
            # Check audio settings
            if self.enable_audio.isChecked():
                try:
                    devices = sd.query_devices()
                    if not any(d['max_input_channels'] > 0 for d in devices):
                        issues.append("No audio input devices found for siren detection")
                except:
                    issues.append("Cannot access audio system")
            
            if issues:
                QMessageBox.warning(self, "Configuration Issues", 
                                  "Issues found:\n" + "\n".join(f"• {issue}" for issue in issues))
            else:
                QMessageBox.information(self, "Validation Successful", 
                                      "Configuration is valid and ready to start!")
                
        except Exception as e:
            QMessageBox.critical(self, "Validation Error", f"Error during validation: {e}")
    
    def test_cameras(self):
        """Test camera connections"""
        try:
            sources = self.get_camera_sources()
            failed_cameras = []
            
            for i, source in enumerate(sources):
                try:
                    cap = cv2.VideoCapture(source)
                    if not cap.isOpened():
                        failed_cameras.append(f"Camera {i+1}: Cannot open source")
                    else:
                        ret, frame = cap.read()
                        if not ret:
                            failed_cameras.append(f"Camera {i+1}: Cannot read frames")
                    cap.release()
                except Exception as e:
                    failed_cameras.append(f"Camera {i+1}: {str(e)}")
            
            if failed_cameras:
                QMessageBox.warning(self, "Camera Test Results", 
                                  "Camera issues found:\n" + "\n".join(failed_cameras))
            else:
                QMessageBox.information(self, "Camera Test Results", 
                                      f"All {len(sources)} cameras tested successfully!")
                
        except Exception as e:
            QMessageBox.critical(self, "Test Error", f"Error testing cameras: {e}")
    
    def get_camera_sources(self) -> List:
        """Get camera sources based on configuration"""
        if self.camera_type == "USB":
            return list(range(self.camera_count_spin.value()))
        else:
            count = self.camera_count_spin.value()
            if len(self.camera_urls) < count or any(not url for url in self.camera_urls[:count]):
                raise ValueError("Please configure all camera URLs")
            return self.camera_urls[:count]
    
    def start_system(self):
        """Start traffic management system"""
        try:
            sources = self.get_camera_sources()
            # Update settings with new options
            self.settings.update({
                'enable_audio': self.enable_audio.isChecked(),
                'enable_ocr': self.enable_ocr.isChecked(),
                'siren_threshold': self.siren_slider.value() / 100.0,
                'ocr_threshold': self.ocr_slider.value() / 100.0,
                'camera_type': self.camera_type,
                'confidence_threshold': self.confidence_slider.value() / 100.0
            })
            logger.info(f"Starting system with {len(sources)} cameras")
            self.hide()
            self.dashboard = EnhancedTrafficDashboard(sources, self.settings)
            self.dashboard.show()
        except Exception as e:
            logger.error(f"System startup error: {e}")
            QMessageBox.critical(self, "Startup Error", f"Cannot start system: {e}")

class EnhancedTrafficDashboard(QMainWindow):
    """Enhanced traffic dashboard with smooth shutdown and fixed priority logic"""
    
    def __init__(self, camera_sources: List, settings: Dict):
        super().__init__()
        
        # Core configuration
        self.camera_sources = camera_sources
        self.settings = settings
        self.num_cameras = len(camera_sources)
        
        # System components
        self.arduino = ArduinoController()
        self.ai_models = AIModels()
        self.camera_streams = []
        self.detection_engines = []
        self.video_widgets = []
        
        # Traffic control state
        self.current_route = 0
        self.manual_override = False
        self.manual_route = None
        self.last_switch_time = time.time()
        self.switch_interval = 60.0  # seconds
        self.emergency_active = False
        self.system_state = SystemState.INITIALIZING
        
        # Performance tracking
        self.start_time = time.time()
        self.total_vehicles_detected = 0
        self.route_metrics = [VehicleMetrics() for _ in range(self.num_cameras)]
        self.route_priorities = [Priority.LOW_TRAFFIC for _ in range(self.num_cameras)]
        
        # Shutdown handling
        self.shutdown_dialog = None
        shutdown_manager.set_progress_callback(self.on_shutdown_progress)
        
        # Initialize UI and system
        self.setup_ui()
        self.setup_timers()
        self.initialize_system()
        
        logger.info("Enhanced Traffic Dashboard initialized")
    
    def setup_ui(self):
        """Setup enhanced UI"""
        self.setWindowTitle("Smart Traffic Management Dashboard v3.0")
        self.setMinimumSize(1000, 700)
        self.resize(1600, 1000)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Header
        self.setup_header(main_layout)
        
        # --- REMOVE this block ---
        # self.system_status_label = QLabel("INITIALIZING")
        # self.system_status_label.setAlignment(Qt.AlignCenter)
        # self.system_status_label.setStyleSheet("""
        #     QLabel {
        #         color: #ffc107;
        #         font-size: 18px;
        #         font-weight: bold;
        #         background: transparent;
        #     }
        # """)
        # main_layout.addWidget(self.system_status_label)
        # --- END REMOVE ---

        # Main content with splitter
        splitter = QSplitter(Qt.Horizontal)
        
        # Video area
        self.setup_video_area(splitter)
        
        # Control panel
        self.setup_control_panel(splitter)
        
        splitter.setSizes([1200, 400])
        main_layout.addWidget(splitter)
        
        # Enhanced status bar
        self.setup_enhanced_status_bar()
        
        # Apply theme
        self.apply_modern_theme()
    
    def setup_header(self, layout):
        """Setup header section with READY indicator on right"""
        header_frame = QFrame()
        header_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #667eea, stop:1 #764ba2);
                border-radius: 15px;
                padding: 20px;
            }
        """)
        header_layout = QHBoxLayout(header_frame)

        # Title and subtitle (left side)
        title_sub_layout = QVBoxLayout()
        title = QLabel("Smart Traffic Management System")
        title.setAlignment(Qt.AlignLeft)
        title.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 26px;
                font-weight: bold;
                background: transparent;
            }
        """)
        subtitle = QLabel("Professional AI-Driven Traffic Control")
        subtitle.setAlignment(Qt.AlignLeft)
        subtitle.setStyleSheet("""
            QLabel {
                color: rgba(255, 255, 255, 0.8);
                font-size: 14px;
                background: transparent;
            }
        """)
        title_sub_layout.addWidget(title)
        title_sub_layout.addWidget(subtitle)
        header_layout.addLayout(title_sub_layout)

        # READY indicator (right side)
        self.system_status_label = QLabel("INITIALIZING")
        self.system_status_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.system_status_label.setMinimumWidth(180)
        self.system_status_label.setStyleSheet("""
            QLabel {
                color: #ffc107;
                font-size: 18px;
                font-weight: bold;
                background: transparent;
            }
        """)
        header_layout.addWidget(self.system_status_label)

        layout.addWidget(header_frame)
    
    def setup_video_area(self, splitter):
        """Setup enhanced video area"""
        video_widget = QWidget()
        video_layout = QVBoxLayout(video_widget)
        
        # Control buttons
        self.setup_control_buttons(video_layout)
        
        # Video grid with scroll area
        self.video_scroll = QScrollArea()
        self.video_scroll.setWidgetResizable(True)
        self.video_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.video_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        self.video_grid_widget = QWidget()
        self.video_grid = QGridLayout(self.video_grid_widget)
        self.video_grid.setSpacing(15)
        
        self.setup_video_widgets()
        
        self.video_scroll.setWidget(self.video_grid_widget)
        video_layout.addWidget(self.video_scroll)
        
        splitter.addWidget(video_widget)
    
    def setup_control_buttons(self, layout):
        """Setup enhanced control buttons"""
        button_frame = QFrame()
        button_frame.setFrameStyle(QFrame.StyledPanel)
        button_layout = QHBoxLayout(button_frame)
        
        self.start_btn = QPushButton("🎬 Start Detection")
        self.start_btn.clicked.connect(self.start_detection)
        self.start_btn.setStyleSheet(self.get_control_button_style("#28a745"))
        
        self.pause_btn = QPushButton("⏸️ Pause")
        self.pause_btn.clicked.connect(self.pause_detection)
        self.pause_btn.setEnabled(False)
        self.pause_btn.setStyleSheet(self.get_control_button_style("#ffc107"))
        
        self.stop_btn = QPushButton("⏹️ Stop")
        self.stop_btn.clicked.connect(self.stop_detection)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet(self.get_control_button_style("#dc3545"))
        
        self.emergency_btn = QPushButton("🚨 Emergency Override")
        self.emergency_btn.clicked.connect(self.emergency_override)
        self.emergency_btn.setStyleSheet(self.get_control_button_style("#fd7e14"))
        
        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.pause_btn)
        button_layout.addWidget(self.stop_btn)
        button_layout.addStretch()
        button_layout.addWidget(self.emergency_btn)
        
        layout.addWidget(button_frame)
    
    def setup_video_widgets(self):
        """Setup enhanced video widgets"""
        for i in range(self.num_cameras):
            widget = EnhancedVideoWidget(i, f"Route {i + 1}")
            self.video_widgets.append(widget)
            
            # Position widgets in grid
            row, col = divmod(i, 2)
            self.video_grid.addWidget(widget, row, col)
    
    def setup_control_panel(self, splitter):
        """Setup enhanced control panel"""
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)

        # Manual traffic control
        self.setup_manual_control(control_layout)

        # Pass time slider (NEW)
        self.setup_pass_time_slider(control_layout)

        # System information
        self.setup_system_info(control_layout)

        # Performance monitoring
        self.setup_performance_monitor(control_layout)

        # Priority analyzer
        self.setup_priority_analyzer(control_layout)

        # System log
        self.setup_system_log(control_layout)

        control_layout.addStretch()
        splitter.addWidget(control_widget)
    
    def setup_pass_time_slider(self, layout):
        """Add manual slider to manage pass time (switch interval)"""
        pass_time_group = QGroupBox("⏱️ Route Pass Time (Switch Interval)")
        pass_time_layout = QHBoxLayout(pass_time_group)

        pass_time_label = QLabel("Pass Time (seconds):")
        self.pass_time_slider = QSlider(Qt.Horizontal)
        self.pass_time_slider.setRange(2, 300)  # 2 to 300 seconds
        self.pass_time_slider.setValue(int(self.switch_interval))
        self.pass_time_slider.setTickInterval(10)
        self.pass_time_slider.setTickPosition(QSlider.TicksBelow)
        self.pass_time_slider.valueChanged.connect(self.on_pass_time_changed)

        self.pass_time_value_label = QLabel(f"{self.switch_interval:.0f}s")
        self.pass_time_value_label.setStyleSheet("font-weight: bold; color: #007bff;")

        pass_time_layout.addWidget(pass_time_label)
        pass_time_layout.addWidget(self.pass_time_slider)
        pass_time_layout.addWidget(self.pass_time_value_label)
        layout.addWidget(pass_time_group)

    def on_pass_time_changed(self, value):
        """Update switch interval when slider changes"""
        self.switch_interval = value
        self.pass_time_value_label.setText(f"{value}s")
        self.log_message(f"⏱️ Route pass time updated to {value} seconds")

    def setup_manual_control(self, layout):
        """Setup manual traffic control panel"""
        manual_group = QGroupBox("🎮 Manual Traffic Control")
        manual_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 14px; }")
        manual_layout = QVBoxLayout(manual_group)
        
        # Manual control buttons
        for i in range(self.num_cameras):
            btn = QPushButton(f"Route {i + 1} GREEN")
            btn.clicked.connect(lambda checked, route=i: self.set_manual_route(route))
            btn.setStyleSheet(self.get_manual_button_style())
            manual_layout.addWidget(btn)
        
        # Auto mode button
        self.auto_btn = QPushButton("🤖 AUTO MODE")
        self.auto_btn.clicked.connect(self.set_auto_mode)
        self.auto_btn.setStyleSheet(self.get_control_button_style("#6f42c1"))
        manual_layout.addWidget(self.auto_btn)
        
        # Current mode indicator
        self.mode_indicator = QLabel("Mode: Automatic")
        self.mode_indicator.setStyleSheet("font-weight: bold; color: #28a745; padding: 8px; background-color: #f8f9fa; border-radius: 4px;")
        manual_layout.addWidget(self.mode_indicator)
        
        layout.addWidget(manual_group)
    
    def setup_system_info(self, layout):
        """Setup system information panel with minimize/maximize"""
        info_group = QGroupBox("ℹ️ System Information")
        info_layout = QVBoxLayout(info_group)

        # Toggle button
        self.info_toggle_btn = QPushButton("−")
        self.info_toggle_btn.setFixedWidth(30)
        self.info_toggle_btn.setStyleSheet("font-weight: bold; font-size: 16px;")
        self.info_toggle_btn.clicked.connect(self.toggle_system_info)

        # Header layout
        header_layout = QHBoxLayout()
        header_layout.addWidget(QLabel("System Information"))
        header_layout.addStretch()
        header_layout.addWidget(self.info_toggle_btn)
        info_layout.addLayout(header_layout)

        # Collapsible content
        self.info_content_widget = QWidget()
        info_content_layout = QVBoxLayout(self.info_content_widget)
        self.uptime_label = QLabel("Uptime: 00:00:00")
        self.total_vehicles_label = QLabel("Total Vehicles: 0")
        self.switch_count_label = QLabel("Route Switches: 0")
        self.emergency_count_label = QLabel("Emergencies: 0")
        info_labels = [
            self.uptime_label,
            self.total_vehicles_label, 
            self.switch_count_label,
            self.emergency_count_label
        ]
        for label in info_labels:
            label.setStyleSheet("font-weight: bold; padding: 4px; color: #495057;")
            info_content_layout.addWidget(label)
        info_layout.addWidget(self.info_content_widget)
        layout.addWidget(info_group)

    def toggle_system_info(self):
        """Toggle system info panel"""
        visible = self.info_content_widget.isVisible()
        self.info_content_widget.setVisible(not visible)
        self.info_toggle_btn.setText("+" if visible else "−")

    def setup_performance_monitor(self, layout):
        """Setup performance monitoring panel with minimize/maximize"""
        perf_group = QGroupBox("📊 Performance Monitor")
        perf_layout = QVBoxLayout(perf_group)

        # Toggle button
        self.perf_toggle_btn = QPushButton("−")
        self.perf_toggle_btn.setFixedWidth(30)
        self.perf_toggle_btn.setStyleSheet("font-weight: bold; font-size: 16px;")
        self.perf_toggle_btn.clicked.connect(self.toggle_performance_monitor)

        # Header layout
        header_layout = QHBoxLayout()
        header_layout.addWidget(QLabel("Performance Monitor"))
        header_layout.addStretch()
        header_layout.addWidget(self.perf_toggle_btn)
        perf_layout.addLayout(header_layout)

        # Collapsible content
        self.perf_content_widget = QWidget()
        perf_content_layout = QVBoxLayout(self.perf_content_widget)

        cpu_layout = QHBoxLayout()
        self.cpu_label = QLabel("CPU: 0%")
        self.cpu_progress = QProgressBar()
        self.cpu_progress.setMaximum(100)
        cpu_layout.addWidget(self.cpu_label)
        cpu_layout.addWidget(self.cpu_progress)
        perf_content_layout.addLayout(cpu_layout)

        mem_layout = QHBoxLayout()
        self.memory_label = QLabel("Memory: 0%")
        self.memory_progress = QProgressBar()
        self.memory_progress.setMaximum(100)
        mem_layout.addWidget(self.memory_label)
        mem_layout.addWidget(self.memory_progress)
        perf_content_layout.addLayout(mem_layout)

        self.arduino_status = QLabel("Arduino: Disconnected")
        self.arduino_status.setStyleSheet("font-weight: bold; color: #dc3545; padding: 4px;")
        perf_content_layout.addWidget(self.arduino_status)

        self.total_fps_label = QLabel("Total FPS: 0.0")
        self.total_fps_label.setStyleSheet("font-weight: bold; color: #17a2b8; padding: 4px;")
        perf_content_layout.addWidget(self.total_fps_label)

        perf_layout.addWidget(self.perf_content_widget)
        layout.addWidget(perf_group)

    def toggle_performance_monitor(self):
        """Toggle performance monitor panel"""
        visible = self.perf_content_widget.isVisible()
        self.perf_content_widget.setVisible(not visible)
        self.perf_toggle_btn.setText("+" if visible else "−")
    
    def setup_priority_analyzer(self, layout):
        """Setup traffic priority analyzer"""
        priority_group = QGroupBox("🎯 Traffic Priority Analysis")
        priority_layout = QVBoxLayout(priority_group)
        
        self.priority_labels = []
        for i in range(self.num_cameras):
            label = QLabel(f"Route {i + 1}: LOW (Weight: 0)")
            label.setStyleSheet("font-size: 12px; padding: 2px; color: #6c757d;")
            self.priority_labels.append(label)
            priority_layout.addWidget(label)
        
        layout.addWidget(priority_group)
    
    def setup_system_log(self, layout):
        """Setup system log viewer with minimize/maximize toggle"""
        log_group = QGroupBox("📝 System Log")
        log_layout = QVBoxLayout(log_group)

        # Toggle button
        self.log_toggle_btn = QPushButton("−")
        self.log_toggle_btn.setFixedWidth(30)
        self.log_toggle_btn.setStyleSheet("font-weight: bold; font-size: 16px;")
        self.log_toggle_btn.clicked.connect(self.toggle_system_log)

        # Header layout
        header_layout = QHBoxLayout()
        header_layout.addWidget(QLabel("System Log"))
        header_layout.addStretch()
        header_layout.addWidget(self.log_toggle_btn)
        log_layout.addLayout(header_layout)

        # Collapsible content
        self.log_content_widget = QWidget()
        log_content_layout = QVBoxLayout(self.log_content_widget)
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(200)
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #00ff00;
                font-family: 'Courier New', monospace;
                font-size: 10px;
                border: 1px solid #444;
                border-radius: 4px;
            }
        """)
        log_content_layout.addWidget(self.log_text)
        log_layout.addWidget(self.log_content_widget)
        layout.addWidget(log_group)

    def toggle_system_log(self):
        """Toggle system log panel"""
        visible = self.log_content_widget.isVisible()
        self.log_content_widget.setVisible(not visible)
        self.log_toggle_btn.setText("+" if visible else "−")
    
    def setup_enhanced_status_bar(self):
        """Setup enhanced status bar"""
        self.status_bar = self.statusBar()
        self.status_bar.setStyleSheet("""
            QStatusBar {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f8f9fa, stop:1 #e9ecef);
                border-top: 1px solid #dee2e6;
                padding: 5px;
            }
        """)
        self.status_bar.showMessage("System initialized - Ready to start detection")
    
    def get_control_button_style(self, color: str) -> str:
        """Get control button stylesheet"""
        return f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border: none;
                padding: 12px 20px;
                border-radius: 8px;
                font-weight: bold;
                font-size: 14px;
                min-width: 140px;
            }}
            QPushButton:hover {{
                background-color: {color}dd;
                transform: translateY(-1px);
            }}
            QPushButton:pressed {{
                background-color: {color}bb;
            }}
            QPushButton:disabled {{
                background-color: #6c757d;
                color: #adb5bd;
            }}
        """
    
    def get_manual_button_style(self) -> str:
        """Get manual control button stylesheet"""
        return """
            QPushButton {
                background-color: #007bff;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                font-weight: bold;
                margin: 2px;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            QPushButton:pressed {
                background-color: #004085;
            }
        """
    
    def apply_modern_theme(self):
        """Apply modern theme to the application"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f8f9fa;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #dee2e6;
                border-radius: 8px;
                margin-top: 1ex;
                padding-top: 10px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px 0 8px;
                color: #495057;
            }
            QLabel {
                color: #495057;
            }
            QProgressBar {
                border: 1px solid #dee2e6;
                border-radius: 4px;
                text-align: center;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #007bff;
                border-radius: 3px;
            }
        """)
    
    def setup_timers(self):
        """Setup system timers"""
        # Main status update timer
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_system_status)
        self.status_timer.start(1000)
        
        # Traffic priority logic timer
        self.priority_timer = QTimer()
        self.priority_timer.timeout.connect(self.update_traffic_priority)
        self.priority_timer.start(2000)
        
        # Performance monitoring timer
        self.perf_timer = QTimer()
        self.perf_timer.timeout.connect(self.update_performance_stats)
        self.perf_timer.start(3000)
        
        # UI refresh timer
        self.ui_timer = QTimer()
        self.ui_timer.timeout.connect(self.refresh_ui_elements)
        self.ui_timer.start(500)
    
    def initialize_system(self):
        """Initialize all system components"""
        try:
            self.log_message("🚀 Initializing Smart Traffic Management System...")
            self.update_system_state(SystemState.INITIALIZING)
            
            # Initialize Arduino
            if self.arduino.connect():
                self.arduino_status.setText("Arduino: Connected ✅")
                self.arduino_status.setStyleSheet("font-weight: bold; color: #28a745; padding: 4px;")
                self.log_message("✅ Arduino connected and ready")
                
                # Register for shutdown
                shutdown_manager.register_component(self.arduino, "Arduino Controller")
            else:
                self.arduino_status.setText("Arduino: Disconnected ❌")
                self.arduino_status.setStyleSheet("font-weight: bold; color: #dc3545; padding: 4px;")
                self.log_message("❌ Arduino connection failed - hardware control disabled")
            
            # Initialize AI models
            if self.ai_models.load_models():
                self.log_message("✅ AI models loaded successfully")
            else:
                self.log_message("❌ Failed to load AI models - detection will be limited")
                QMessageBox.warning(self, "AI Models Warning", 
                                  "Failed to load AI models. Detection functionality will be limited.")
            
            # Initialize camera streams and detection engines
            self.initialize_cameras_and_detection()
            
            self.update_system_state(SystemState.READY)
            self.log_message("✅ System initialization completed successfully")
            
        except Exception as e:
            logger.error(f"System initialization error: {e}")
            self.log_message(f"❌ System initialization error: {e}")
            self.update_system_state(SystemState.ERROR)
            QMessageBox.critical(self, "Initialization Error", 
                               f"Failed to initialize system: {e}")
    
    def initialize_cameras_and_detection(self):
        """Initialize camera streams and detection engines"""
        for i, source in enumerate(self.camera_sources):
            try:
                # Create camera stream
                camera_name = f"Camera {i + 1} ({'USB' if isinstance(source, int) else 'IP'})"
                camera = EnhancedCameraStream(i, source, camera_name)
                camera.frame_ready.connect(self.on_frame_received)
                camera.error_occurred.connect(self.on_camera_error)
                camera.status_changed.connect(self.on_camera_status_changed)
                self.camera_streams.append(camera)
                
                # Create detection engine
                detector = EnhancedDetectionEngine(i, self.ai_models, self.settings)
                detector.detection_result.connect(self.on_detection_result)
                detector.error_occurred.connect(self.on_detection_error)
                self.detection_engines.append(detector)
                
                # Setup siren detection if enabled
                if self.settings.get('enable_audio', False):
                    camera.setup_siren_detection(detector)
                
                # Register for shutdown
                shutdown_manager.register_component(camera, f"Camera Stream {i}")
                shutdown_manager.register_component(detector, f"Detection Engine {i}")
                
                self.log_message(f"✅ Initialized {camera_name}")
                
            except Exception as e:
                logger.error(f"Failed to initialize camera {i}: {e}")
                self.log_message(f"❌ Failed to initialize camera {i}: {e}")
    
    @pyqtSlot(int, np.ndarray)
    def on_frame_received(self, camera_id: int, frame: np.ndarray):
        """Handle received camera frame"""
        try:
            if camera_id < len(self.video_widgets):
                self.video_widgets[camera_id].update_frame(frame)
            
            if camera_id < len(self.detection_engines):
                self.detection_engines[camera_id].process_frame(frame)
                
        except Exception as e:
            logger.error(f"Error processing frame from camera {camera_id}: {e}")
    
    @pyqtSlot(int, dict)
    def on_detection_result(self, camera_id: int, result: Dict[str, Any]):
        """Handle detection results with enhanced priority logic"""
        try:
            if camera_id < len(self.video_widgets):
                self.video_widgets[camera_id].update_detection_data(result)
            
            # Update route metrics and priorities
            if camera_id < len(self.route_metrics):
                metrics = result.get('vehicle_metrics', VehicleMetrics())
                priority = result.get('priority', Priority.LOW_TRAFFIC)
                
                self.route_metrics[camera_id] = metrics
                self.route_priorities[camera_id] = priority
                
                # Update total vehicle count
                total_current = metrics.total_vehicles()
                self.total_vehicles_detected += total_current
            
            # Handle emergency detection
            if result.get('emergency_detected', False):
                self.handle_emergency_detection(camera_id, result)
                
        except Exception as e:
            logger.error(f"Error handling detection result from camera {camera_id}: {e}")
    
    @pyqtSlot(int, str)
    def on_camera_error(self, camera_id: int, error_message: str):
        """Handle camera errors"""
        self.log_message(f"📹 Camera {camera_id + 1} error: {error_message}")
        logger.error(f"Camera {camera_id} error: {error_message}")
    
    @pyqtSlot(int, str)
    def on_camera_status_changed(self, camera_id: int, status: str):
        """Handle camera status changes"""
        self.log_message(f"📹 Camera {camera_id + 1} status: {status}")
    
    @pyqtSlot(int, str)
    def on_detection_error(self, camera_id: int, error_message: str):
        """Handle detection errors"""
        self.log_message(f"🔍 Detection error on camera {camera_id + 1}: {error_message}")
        logger.error(f"Detection error on camera {camera_id}: {error_message}")
    
    def handle_emergency_detection(self, camera_id: int, detection_result: Dict):
        """Handle emergency vehicle detection with priority override"""
        if self.manual_override:
            return  # Don't override manual control
        
        emergency_type = "Emergency Vehicle"
        if detection_result.get('siren_detected', False):
            emergency_type += " + Siren"
        
        if camera_id != self.current_route:
            self.log_message(f"🚨 {emergency_type} detected on Route {camera_id + 1} - Priority Override!")
            self.switch_to_route(camera_id, f"Emergency: {emergency_type}")
            self.emergency_active = True
            
            # Reset emergency after 45 seconds
            QTimer.singleShot(45000, lambda: setattr(self, 'emergency_active', False))
    
    def start_detection(self):
        """Start the detection system"""
        try:
            if not self.ai_models.is_ready():
                QMessageBox.warning(self, "AI Models Not Ready", 
                                  "AI models are not loaded. Detection will be limited.")
            
            self.log_message("🎬 Starting detection system...")
            self.update_system_state(SystemState.RUNNING)
            
            # Start camera streams
            for camera in self.camera_streams:
                camera.start()
            
            # Start detection engines
            for detector in self.detection_engines:
                detector.start()
            
            # Update UI
            self.start_btn.setEnabled(False)
            self.pause_btn.setEnabled(True)
            self.stop_btn.setEnabled(True)
            
            # Initialize traffic control
            self.switch_to_route(0, "System Startup")
            
            self.log_message("✅ Detection system started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start detection: {e}")
            self.log_message(f"❌ Failed to start detection: {e}")
            QMessageBox.critical(self, "Start Error", f"Failed to start detection system: {e}")
    
    def pause_detection(self):
        """Pause the detection system"""
        try:
            if self.system_state == SystemState.RUNNING:
                self.update_system_state(SystemState.PAUSED)
                
                # Disable detection but keep cameras running
                for detector in self.detection_engines:
                    detector.enable_detection(False)
                
                self.log_message("⏸️ Detection system paused")
                
            elif self.system_state == SystemState.PAUSED:
                self.update_system_state(SystemState.RUNNING)
                
                # Re-enable detection
                for detector in self.detection_engines:
                    detector.enable_detection(True)
                
                self.log_message("▶️ Detection system resumed")
                
        except Exception as e:
            logger.error(f"Error pausing/resuming detection: {e}")
            self.log_message(f"❌ Error pausing/resuming detection: {e}")
    
    def stop_detection(self):
        """Stop the detection system"""
        try:
            self.log_message("⏹️ Stopping detection system...")
            self.update_system_state(SystemState.READY)
            
            # Stop detection engines
            for detector in self.detection_engines:
                detector.stop()
            
            # Stop camera streams
            for camera in self.camera_streams:
                camera.stop()
            
            # Emergency shutdown of traffic lights
            if self.arduino.connected:
                self.arduino.emergency_shutdown()
            
            # Update UI
            self.start_btn.setEnabled(True)
            self.pause_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)
            
            self.log_message("✅ Detection system stopped")
            
        except Exception as e:
            logger.error(f"Error stopping detection: {e}")
            self.log_message(f"❌ Error stopping detection: {e}")
    
    def emergency_override(self):
        """Emergency override - all routes to RED"""
        try:
            reply = QMessageBox.question(
                self, "Emergency Override",
                "This will set ALL traffic lights to RED immediately.\n\nProceed?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                if self.arduino.connected:
                    self.arduino.emergency_shutdown()
                    self.log_message("🚨 EMERGENCY OVERRIDE - All routes set to RED")
                    QMessageBox.information(self, "Emergency Override Complete", 
                                          "All traffic lights have been set to RED for emergency.")
                else:
                    QMessageBox.warning(self, "Arduino Not Connected", 
                                      "Cannot control traffic lights - Arduino not connected.")
                    
        except Exception as e:
            logger.error(f"Emergency override error: {e}")
            self.log_message(f"❌ Emergency override error: {e}")
    
    def set_manual_route(self, route: int):
        """Set manual route override"""
        if route >= self.num_cameras:
            return
        
        self.manual_override = True
        self.manual_route = route
        self.switch_to_route(route, f"Manual Override - Route {route + 1}")
        
        # Update UI
        self.mode_indicator.setText(f"Mode: Manual (Route {route + 1})")
        self.mode_indicator.setStyleSheet("font-weight: bold; color: #dc3545; padding: 8px; background-color: #f8d7da; border-radius: 4px;")
        
        self.log_message(f"🎮 Manual override: Route {route + 1} set to GREEN")
    
    def set_auto_mode(self):
        """Return to automatic mode"""
        self.manual_override = False
        self.manual_route = None
        self.last_switch_time = time.time() - self.switch_interval  # Force immediate evaluation
        
        # Update UI
        self.mode_indicator.setText("Mode: Automatic")
        self.mode_indicator.setStyleSheet("font-weight: bold; color: #28a745; padding: 8px; background-color: #d4edda; border-radius: 4px;")
        
        self.log_message("🤖 Returned to automatic traffic control")
    
    def switch_to_route(self, route: int, reason: str):
        """Switch active traffic route with enhanced logging"""
        if route == self.current_route:
            return
        
        old_route = self.current_route
        self.current_route = route
        self.last_switch_time = time.time()
        
        # Update Arduino traffic lights
        success = False
        if self.arduino.connected:
            success = self.arduino.set_traffic_lights(route, self.num_cameras)
        
        # Update video widget visuals
        for i, widget in enumerate(self.video_widgets):
            widget.set_active_status(i == route)
        
        # Update header
        self.active_route_label.setText(f"Active Route: {route + 1}")
        
        # Log the switch
        status = "✅" if success else "❌"
        self.log_message(f"{status} Route switch: {old_route + 1} → {route + 1} ({reason})")
        
        if not success and self.arduino.connected:
            logger.error("Failed to update Arduino traffic lights")
    
    def update_traffic_priority(self):
        """Enhanced traffic priority logic with proper weight calculation"""
        if self.manual_override or self.system_state != SystemState.RUNNING:
            return
        
        try:
            current_time = time.time()
            
            # Emergency vehicles have absolute priority
            emergency_routes = []
            for i, priority in enumerate(self.route_priorities):
                if priority == Priority.EMERGENCY:
                    emergency_routes.append(i)
            
            if emergency_routes:
                # Switch to first emergency route if not already active
                emergency_route = emergency_routes[0]
                if emergency_route != self.current_route:
                    self.switch_to_route(emergency_route, "Emergency Vehicle Priority")
                return
            
            # Normal priority switching (every 60 seconds)
            if current_time - self.last_switch_time >= self.switch_interval:
                # Calculate weighted scores for each route
                route_scores = []
                for i, metrics in enumerate(self.route_metrics):
                    score = metrics.weighted_score()
                    route_scores.append((i, score, metrics))
                
                # Sort by score (highest first)
                route_scores.sort(key=lambda x: x[1], reverse=True)
                
                if route_scores:
                    next_route, max_score, best_metrics = route_scores[0]
                    
                    # Only switch if there's significant traffic or round-robin
                    if max_score > 0:
                        if next_route != self.current_route:
                            self.switch_to_route(
                                next_route, 
                                f"Traffic Priority (Weight: {max_score:.1f}, "
                                f"Vehicles: {best_metrics.total_vehicles()})"
                            )
                    else:
                        # Round-robin when no significant traffic
                        next_route = (self.current_route + 1) % self.num_cameras
                        self.switch_to_route(next_route, "Round-robin (No Traffic)")
                
        except Exception as e:
            logger.error(f"Traffic priority update error: {e}")
    
    def update_system_status(self):
        """Update system status information"""
        try:
            # Calculate uptime
            uptime = int(time.time() - self.start_time)
            hours = uptime // 3600
            minutes = (uptime % 3600) // 60
            seconds = uptime % 60
            self.uptime_label.setText(f"Uptime: {hours:02d}:{minutes:02d}:{seconds:02d}")
            
            # Update total vehicles (use a running count instead of current count)
            self.total_vehicles_label.setText(f"Total Vehicles: {self.total_vehicles_detected}")
            
            # Update route switch count (placeholder - you'd track this)
            # self.switch_count_label.setText(f"Route Switches: {self.switch_count}")
            
            # Update emergency count (placeholder - you'd track this)
            # self.emergency_count_label.setText(f"Emergencies: {self.emergency_count}")
            
        except Exception as e:
            logger.error(f"Status update error: {e}")
    
    def update_performance_stats(self):
        """Update performance statistics"""
        try:
            # CPU and memory usage
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            
            self.cpu_label.setText(f"CPU: {cpu_percent:.1f}%")
            self.cpu_progress.setValue(int(cpu_percent))
            
            self.memory_label.setText(f"Memory: {memory_percent:.1f}%")
            self.memory_progress.setValue(int(memory_percent))
            
            # Total FPS from all cameras
            total_fps = 0
            for i, detector in enumerate(self.detection_engines):
                if hasattr(detector, 'current_fps'):
                    total_fps += detector.current_fps
            
            self.total_fps_label.setText(f"Total FPS: {total_fps:.1f}")
            
            # Update status bar
            self.status_bar.showMessage(
                f"System: {self.system_state.value} | "
                f"Active Route: {self.current_route + 1} | "
                f"Total FPS: {total_fps:.1f} | "
                f"CPU: {cpu_percent:.1f}% | "
                f"Memory: {memory_percent:.1f}%"
            )
            
        except Exception as e:
            logger.error(f"Performance stats error: {e}")
    
    def refresh_ui_elements(self):
        """Refresh UI elements regularly"""
        try:
            # Update priority analyzer
            for i, (metrics, priority) in enumerate(zip(self.route_metrics, self.route_priorities)):
                if i < len(self.priority_labels):
                    weight = metrics.weighted_score()
                    vehicles = metrics.total_vehicles()
                    
                    priority_colors = {
                        Priority.EMERGENCY: "#dc3545",
                        Priority.HIGH_TRAFFIC: "#fd7e14",
                        Priority.NORMAL: "#28a745", 
                        Priority.LOW_TRAFFIC: "#6c757d"
                    }
                    
                    color = priority_colors.get(priority, "#6c757d")
                    priority_name = priority.name.replace('_', ' ')
                    
                    self.priority_labels[i].setText(f"Route {i + 1}: {priority_name} (Weight: {weight:.1f}, Vehicles: {vehicles})")
                    self.priority_labels[i].setStyleSheet(f"font-size: 12px; padding: 2px; color: {color}; font-weight: bold;")
            
        except Exception as e:
            logger.error(f"UI refresh error: {e}")
    
    def update_system_state(self, state: SystemState):
        """Update system state with visual feedback"""
        self.system_state = state
        
        state_colors = {
            SystemState.INITIALIZING: "#ffc107",
            SystemState.READY: "#17a2b8",
            SystemState.RUNNING: "#28a745",
            SystemState.PAUSED: "#fd7e14",
            SystemState.SHUTTING_DOWN: "#6c757d",
            SystemState.ERROR: "#dc3545",
            SystemState.OFFLINE: "#343a40"
        }
        
        color = state_colors.get(state, "#6c757d")
        self.system_status_label.setText(state.value.upper())
        self.system_status_label.setStyleSheet(f"""
            QLabel {{
                color: {color};
                font-size: 18px;
                font-weight: bold;
                background: transparent;
            }}
        """)
    
    def log_message(self, message: str):
        """Add message to system log with timestamp"""
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            formatted_message = f"[{timestamp}] {message}"
            
            self.log_text.append(formatted_message)
            
            # Auto-scroll to bottom
            scrollbar = self.log_text.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
            
            # Limit log size for performance
            if self.log_text.document().blockCount() > 200:
                cursor = self.log_text.textCursor()
                cursor.movePosition(cursor.Start)
                cursor.select(cursor.BlockUnderCursor)
                cursor.removeSelectedText()
                
        except Exception as e:
            logger.error(f"Log message error: {e}")
    
    def on_shutdown_progress(self, progress: int, message: str):
        """Handle shutdown progress updates"""
        if not self.shutdown_dialog:
            self.shutdown_dialog = ShutdownDialog(self)
            self.shutdown_dialog.show()
        
        self.shutdown_dialog.update_progress(progress, message)
        
        if progress >= 100:
            if self.shutdown_dialog:
                self.shutdown_dialog.accept()
                self.shutdown_dialog = None
    
    def closeEvent(self, event: QCloseEvent):
        """Handle application close with graceful shutdown"""
        try:
            self.log_message("🔄 Initiating graceful system shutdown...")
            self.update_system_state(SystemState.SHUTTING_DOWN)
            
            # Show shutdown dialog
            self.shutdown_dialog = ShutdownDialog(self)
            self.shutdown_dialog.show()
            
            # Start graceful shutdown
            shutdown_manager.shutdown()
            
            event.accept()
            
        except Exception as e:
            logger.error(f"Shutdown error: {e}")
            event.accept()

def main():
    """Enhanced main function with proper setup"""
    try:
        # Set Qt application attributes
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
        
        # Create application
        app = QApplication(sys.argv)
        app.setApplicationName("Smart Traffic Management System")
        app.setApplicationVersion("3.0")
        app.setOrganizationName("Smart Traffic Systems")
        
        # Set modern style
        app.setStyle('Fusion')
        
        # Apply dark palette if preferred
        # app.setPalette(get_dark_palette())  # Optional
        
        # Create and show setup window
        setup_window = EnhancedSetupWindow()
        setup_window.show()
        
        logger.info("🚀 Smart Traffic Management System v3.0 started successfully")
        
        return app.exec_()
        
    except Exception as e:
        logger.error(f"Application startup error: {e}")
        print(f"Fatal error during startup: {e}")
        return 1
    finally:
        logger.info("Application terminated")

if __name__ == "__main__":
    sys.exit(main())
