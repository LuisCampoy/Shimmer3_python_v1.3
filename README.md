# Shimmer3 IMU Data Streaming, Logging, and Export Tool

A comprehensive Python application for streaming, logging, and exporting data from Shimmer3 IMU devices via Bluetooth RFCOMM. This application provides real-time data acquisition, multi-format data logging, and advanced data processing capabilities for research and development applications.

## Features

### Data Acquisition
- **Real-time data streaming** from Shimmer3 devices via Bluetooth RFCOMM
- **Multiple sensor support** - accelerometer, gyroscope, magnetometer
- **Configurable sampling rates** - Support for standard Shimmer3 sampling frequencies (51.2 Hz default)
- **22-byte packet parsing** - Robust packet structure with header, sensor data, metadata, and checksum
- **Passive connection mode** - Connect without handshake for streaming-only mode
- **Connection preservation** - Keep RFCOMM binding alive between sessions
- **Manual RFCOMM mode** - Use pre-bound RFCOMM devices without automatic pairing

### Data Logging & Export
- **Multi-format data logging** - CSV format with timestamped files
- **High-performance buffering** - Configurable buffer sizes (default 1000 samples)
- **Automatic file rotation** - New files created at configurable intervals
- **Multiple export formats** - CSV, HDF5 with pandas backend
- **Statistical analysis** - Comprehensive statistics including mean, std, min, max, skewness, kurtosis
- **Session metadata** - JSON metadata files with device and session information

### Technical Features
- **Async/await architecture** - Non-blocking real-time processing
- **Thread-safe operations** - Safe concurrent data handling
- **Comprehensive logging** - Detailed application logs with timestamps (logs/ directory)
- **Flexible configuration** - JSON-based configuration with validation
- **Robust error handling** - Error recovery and user feedback
- **Diagnostic tools** - Built-in stream diagnostics and packet analysis

## Installation

### Prerequisites
- Python 3.8 or higher
- Shimmer3 device with serial/Bluetooth connection
- Virtual environment (recommended)

### Quick Setup
```bash
# Navigate to project directory
cd Shimmer3_python_v1.3

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Linux/Mac
# .venv\Scripts\activate   # On Windows

# Install Python dependencies
pip install -r requirements.txt

# Install in development mode (optional)
pip install -e .

# Install Bluetooth tools (Linux only)
sudo apt-get update
sudo apt-get install -y bluez bluez-tools

# Add user to dialout group for serial port access
sudo usermod -a -G dialout $USER
# Logout and login for group changes to take effect
```

## Configuration

The application uses `config/shimmer_config.json` for configuration:

```json
{
  "shimmer": {
    "device_id": "shimmer3_default",
    "mac_address": "00:06:66:B1:4D:A1",
    "port": "/dev/rfcomm0",
    "baudrate": 115200,
    "timeout": 1.0,
    "sampling_rate": 51.2,
    "sensors": ["accelerometer", "gyroscope", "magnetometer"],
    "manual_rfcomm": true,
    "passive_connect": true,
    "keep_open_on_handshake_failure": true,
    "preserve_connection": false,
    "keepalive_enabled": false,
    "keepalive_interval": 30
  },
  "bluetooth": {
    "adapter": "hci0",
    "scan_timeout": 10,
    "connection_timeout": 30
  },
  "data": {
    "raw_directory": "data/raw",
    "processed_directory": "data/processed",
    "format": "csv",
    "max_file_size_mb": 100,
    "buffer_size": 1000,
    "rotation_interval": 1000
  },
  "export": {
    "formats": ["csv", "hdf5"],
    "interval": 1000,
    "enabled": true
  },
  "logging": {
    "level": "INFO",
    "directory": "logs",
    "max_file_size_mb": 10,
    "backup_count": 5
  }
}
```

### Key Configuration Options

#### Shimmer Settings
- **manual_rfcomm**: `true` - Use manually bound RFCOMM device (skip automatic pairing)
- **passive_connect**: `true` - Skip handshake, start streaming immediately
- **preserve_connection**: `false` - Keep connection open after streaming ends
- **keepalive_enabled**: `false` - Send periodic keepalive commands
- **port**: `/dev/rfcomm0` - Serial device path for RFCOMM connection

#### Data Settings
- **buffer_size**: Number of samples to buffer before writing
- **rotation_interval**: Create new file after N samples
- **export.interval**: Trigger export after N samples

## Usage

### Prerequisites - Manual RFCOMM Binding

Before running the application, manually bind the Shimmer3 device to an RFCOMM port:

```bash
# Scan for Bluetooth devices
hcitool scan
# Output: 00:06:66:B1:4D:A1	n/a

# Bind to RFCOMM channel 1
sudo rfcomm bind 0 00:06:66:B1:4D:A1 1

# Verify binding
rfcomm
# Output: rfcomm0: 00:06:66:B1:4D:A1 channel 1 clean

# Check device access
ls -la /dev/rfcomm0
```

### Basic Usage
```bash
# Ensure virtual environment is activated
source .venv/bin/activate

# Basic data streaming (with manual RFCOMM)
python run_shimmer.py

# Force active mode (override passive_connect config)
python run_shimmer.py --force-active

# Preserve connection after streaming
python run_shimmer.py --preserve-connection

# Custom duration (seconds)
python run_shimmer.py --duration 60

# Export existing data only (no streaming)
python run_shimmer.py --export-only

# Diagnostic modes
python run_shimmer.py --diagnostic-sniff  # Sniff packets without parsing
python run_shimmer.py --diagnostic-stream-seconds 10  # Stream raw bytes
python run_shimmer.py --diagnostic-parse-seconds 10  # Stream and parse
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--config PATH` | Configuration file path | `config/shimmer_config.json` |
| `--duration SECONDS` | Recording duration | Continuous |
| `--force-active` | Override passive_connect, use active mode | `false` |
| `--preserve-connection` | Keep RFCOMM open after streaming | `false` |
| `--export-only` | Export mode only (no streaming) | `false` |
| `--diagnostic-sniff` | Passive packet sniffing mode | `false` |
| `--diagnostic-stream-seconds N` | Stream raw bytes for N seconds | - |
| `--diagnostic-parse-seconds N` | Stream and parse for N seconds | - |

## Data Output

### File Organization
```
data/
├── raw/                              # Raw data files
│   ├── shimmer3_data_20251014_172951_001_20251014_173019.csv
│   └── session_metadata_20251014_172951.json
├── processed/                        # Processed/exported data
│   ├── shimmer3_export_20251014_173020.csv
│   ├── shimmer3_export_20251014_173023.h5
│   └── shimmer3_export_20251014_173020_statistics.json
logs/
└── shimmer3_streamer_20251014_172951.log
```

### CSV Data Format
Raw data files contain timestamped sensor readings:
```csv
timestamp,session_id,file_sequence,packet_id,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z,mag_x,mag_y,mag_z
1728931791.234,20251014_172951,1,1,0.98,0.02,9.81,0.1,-0.2,0.0,125.4,-112.3,245.6
1728931791.254,20251014_172951,1,2,0.97,0.03,9.82,0.0,-0.1,0.1,125.2,-112.1,245.8
```

### Packet Structure
The application parses 22-byte packets from the Shimmer3 device:
- **Header**: 2 bytes (starts with 0x00)
- **Accelerometer**: 6 bytes (3 × 16-bit signed integers, LSB first)
- **Gyroscope**: 6 bytes (3 × 16-bit signed integers, LSB first)
- **Magnetometer**: 6 bytes (3 × 16-bit signed integers, LSB first)
- **Metadata**: 2 bytes
- **Checksum**: 2 bytes

## Testing & Validation

### System Requirements Test
```bash
# Test configuration loading
python -c "
import sys
sys.path.insert(0, 'src')
from utils.config import Config
config = Config('config/shimmer_config.json')
print('✓ Configuration loaded successfully')
print('Sampling rate:', config.get('shimmer.sampling_rate'))
print('Port:', config.get('shimmer.port'))
print('Sensors:', config.get('shimmer.sensors'))
"
```

### Application Testing
```bash
# Test application help
python run_shimmer.py --help

# Test export functionality (no hardware required)
python run_shimmer.py --config config/shimmer_config.json --export-only

# Test with device (if available)
python run_shimmer.py -c config/shimmer_config.json -d /dev/ttyUSB0 -t 10
```

## Troubleshooting
hciconfig
hci0:	Type: Primary  Bus: USB
BD Address: 24:0A:64:1D:0A:AC  ACL MTU: 1022:8  SCO MTU: 183:5
UP RUNNING ISCAN 
RX bytes:80224 acl:0 sco:0 events:2218 errors:0
TX bytes:7005 acl:0 sco:0 commands:198 errors:0

hcitool scan
Scanning ...

00:06:66:B1:4D:A1	n/a

sudo rfcomm bind 0 00:06:66:B1:4D:A1

rfcomm
rfcomm0: 00:06:66:B1:4D:A1 channel 1 clean 

Ensure you’ve manually bound the device first:

sudo rfcomm release 0 2>/dev/null || true
sudo rfcomm bind 0 00:06:66:B1:4D:A1 1
rfcomm

check if we can access device:

ls -la /dev/rfcomm0

### Common Issues

#### Import Errors
```bash
# Ensure virtual environment is activated and dependencies installed
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

#### Serial Port Issues
```bash
# Linux: Add user to dialout group
sudo usermod -a -G dialout $USER
# Then logout and login again

# Check available ports
python -c "import serial.tools.list_ports; print([p.device for p in serial.tools.list_ports.comports()])"
```

#### Configuration Issues
```bash
# Validate JSON syntax
python -c "import json; json.load(open('config/shimmer_config.json')); print('Valid JSON')"
```

### Hardware Testing
```bash
# Test serial port availability
ls -la /dev/ttyUSB*  # Linux

# Short device test
python run_shimmer.py -c config/shimmer_config.json -d /dev/ttyUSB0 -t 10
```

## Project Structure
```
Shimmer3_python_v1.3/
├── src/                             # Source code
│   ├── main.py                      # Main application orchestration
│   ├── shimmer_client.py            # Shimmer3 device communication and packet parsing
│   ├── data_logger.py               # Data buffering and CSV logging
│   ├── data_exporter.py             # Multi-format export and statistics
│   ├── bluetooth_manager.py         # Bluetooth connection management
│   └── utils/                       # Utility modules
│       ├── config.py                # Configuration management
│       └── __init__.py              # Helper functions
├── config/
│   └── shimmer_config.json          # Main configuration file
├── data/
│   ├── raw/                         # Raw CSV data and session metadata
│   └── processed/                   # Exported data (CSV, HDF5, statistics)
├── logs/                            # Application log files
├── run_shimmer.py                   # CLI entry point
├── requirements.txt                 # Python dependencies
├── pyproject.toml                   # Package configuration
└── README.md                        # This file
```

## Advanced Features

### Diagnostic Tools
```bash
# Passive packet sniffing (no commands sent to device)
python run_shimmer.py --diagnostic-sniff

# Stream raw bytes for analysis
python run_shimmer.py --diagnostic-stream-seconds 10

# Stream and parse packets with diagnostics
python run_shimmer.py --diagnostic-parse-seconds 10
```

### Connection Management
- **preserve_connection**: Keeps RFCOMM binding and serial port open after streaming ends
- **manual_rfcomm**: Use pre-bound RFCOMM devices (skip automatic Bluetooth pairing)
- **passive_connect**: Skip device handshake, start streaming immediately
- **keep_open_on_handshake_failure**: Don't close connection if handshake fails

### Logging
- Application logs are written to `logs/shimmer3_streamer_TIMESTAMP.log`
- Both console and file logging are enabled
- Configurable log levels (DEBUG, INFO, WARNING, ERROR)

## Known Issues & Solutions

### RFCOMM Channel Closes After Script Exit
This is normal OS behavior. When the serial port closes, the RFCOMM channel is released by the Bluetooth stack. The `/dev/rfcomm0` device file persists, but the channel shows as "closed" in `rfcomm` output. To keep streaming, use `--preserve-connection` to prevent port closure during script execution.

### "No module named 'pandas'" Error
Ensure the virtual environment is activated and dependencies are installed:
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Permission Denied on /dev/rfcomm0
Add your user to the dialout group:
```bash
sudo usermod -a -G dialout $USER
# Logout and login for changes to take effect
```

---

**Author**: Luis Campoy  
**Version**: 1.3.0  
**Last Updated**: October 15, 2025  
**License**: MIT
