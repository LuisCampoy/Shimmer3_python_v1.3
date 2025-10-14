# Shimmer3 IMU Data Streaming, Logging, and Export Tool

A comprehensive Python application for streaming, logging, and exporting data from Shimmer3 IMU devices. This application provides real-time data acquisition, multi-format data logging, and advanced data processing capabilities for research and development applications.

## Features

### Data Acquisition
- **Real-time data streaming** from Shimmer3 devices via serial/Bluetooth
- **Multiple sensor support** - accelerometer, gyroscope, magnetometer, and battery monitoring
- **Configurable sampling rates** - Support for standard Shimmer3 sampling frequencies
- **Sensor range configuration** - Customizable measurement ranges for each sensor type
- **Robust connection handling** - Automatic reconnection and error recovery

### Data Logging & Export
- **Multi-format data logging** - CSV, JSON Lines, and binary (pickle) formats
- **High-performance buffering** - Configurable buffer sizes for optimal performance
- **Multiple export formats** - CSV, HDF5, Excel, MATLAB (.mat), Parquet
- **Statistical analysis** - Comprehensive statistics including mean, std, skewness, kurtosis
- **Data visualization** - Automatic plot generation for all sensor channels
- **Compression support** - Optional gzip compression for all file formats

### Technical Features
- **Async/await architecture** - Non-blocking real-time processing
- **Thread-safe operations** - Safe concurrent data handling
- **Comprehensive logging** - Detailed application and error logging
- **Flexible configuration** - JSON-based configuration with validation
- **Robust error handling** - Error recovery and user feedback

## Installation

### Prerequisites
- Python 3.8 or higher
- Shimmer3 device with serial/Bluetooth connection
- Virtual environment (recommended)

### Quick Setup
```bash
# Clone or download the project
cd Simmer3_1.1

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .

# install bluez tools
sudo apt-get update && sudo apt-get install -y bluez-tools
sudo apt install mono-devel
```

## Configuration

The application uses `config/shimmer_config.json` for configuration:

```json
{
  "shimmer": {
    "port": "/dev/ttyUSB0",
    "baudrate": 115200,
    "sampling_rate": 51.2,
    "sensors": ["accelerometer", "gyroscope", "magnetometer"],
    "sensor_range": {
      "accelerometer": 2,
      "gyroscope": 250, 
      "magnetometer": 1
    }
  },
  "data": {
    "raw_directory": "data/raw",
    "processed_directory": "data/processed",
    "format": "csv",
    "max_file_size_mb": 100,
    "buffer_size": 1000
  },
  "export": {
    "formats": ["csv", "hdf5"],
    "real_time": false,
    "compression": true
  }
}
```

## Usage

### Basic Usage
```bash
# Ensure virtual environment is activated
source .venv/bin/activate

# Basic data streaming
python3 run_shimmer.py -c config/shimmer_config.json -d /dev/ttyUSB0

# Record for specific duration (60 seconds)
python3 run_shimmer.py -c config/shimmer_config.json -d /dev/ttyUSB0 -t 60

# Use custom sampling rate
python3 run_shimmer.py -c config/shimmer_config.json -d /dev/ttyUSB0 -r 102.4

# Export existing data only (no streaming)
python3 run_shimmer.py -c config/shimmer_config.json --export-only
```

### Command Line Options

| Option | Description | Example |
|--------|-------------|---------|
| `-c, --config` | Path to configuration file | `-c config/shimmer_config.json` |
| `-d, --device` | Shimmer3 device port or ID | `-d /dev/ttyUSB0` |
| `-r, --rate` | Sampling rate in Hz | `-r 102.4` |
| `-t, --duration` | Recording duration in seconds | `-t 300` |
| `--export-only` | Export mode only (no streaming) | `--export-only` |

## Data Output

### File Organization
```
data/
├── raw/                              # Raw data files
│   ├── shimmer3_data_20251009_080224_001_20251009_080224.csv
│   └── session_metadata_20251009_080224.json
└── processed/                        # Processed/exported data
    ├── shimmer3_export_20251009_090000.csv
    ├── shimmer3_export_20251009_090000.h5
    └── shimmer3_export_20251009_090000_statistics.json
```

### CSV Data Format
```csv
timestamp,packet_id,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z,mag_x,mag_y,mag_z,battery
1728462124.123,1,0.98,0.02,9.81,0.1,-0.2,0.0,25.4,-12.3,45.6,3.1
1728462124.143,2,0.97,0.03,9.82,0.0,-0.1,0.1,25.2,-12.1,45.8,3.1
```

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
Simmer3_1.1/
├── src/                             # Source code
│   ├── main.py                      # Main application
│   ├── shimmer_client.py            # Shimmer3 communication  
│   ├── data_logger.py               # Data logging functionality
│   ├── data_exporter.py             # Data export and analysis
│   └── utils/
│       └── config.py                # Configuration management
├── config/
│   └── shimmer_config.json          # Configuration file
├── data/                            # Data directories (raw/processed)
├── logs/                            # Application logs
└── run_shimmer.py                   # Entry point script
```

luis@luis-cornell:~$ hciconfig
hci0:	Type: Primary  Bus: USB
BD Address: 24:0A:64:1D:0A:AC  ACL MTU: 1022:8  SCO MTU: 183:5
UP RUNNING ISCAN 
RX bytes:80224 acl:0 sco:0 events:2218 errors:0
TX bytes:7005 acl:0 sco:0 commands:198 errors:0

luis@luis-cornell:~$ hcitool scan
Scanning ...
luis@luis-cornell:~$ hcitool scan
Scanning ...
	00:06:66:B1:4D:A1	n/a
luis@luis-cornell:~$ sudo rfcomm bind 0 00:06:66:B1:4D:A1
[sudo] password for luis: 

luis@luis-cornell:~$ rfcomm
rfcomm0: 00:06:66:B1:4D:A1 channel 1 clean 
luis@luis-cornell:~$ 

The rfcomm0 device exists but it's closed. Let me try to establish the connection properly:

sudo rfcomm release 0

sudo rfcomm bind 0 00:06:66:B1:4D:A1 1

rfcomm show

Ensure you’ve manually bound the device first:
sudo rfcomm release 0 2>/dev/null || true
sudo rfcomm bind 0 00:06:66:B1:4D:A1 1
rfcomm

sudo rfcomm show 0
Good! Now the device is in "clean" state. Let me check if we can access it:


ls -la /dev/rfcomm0



---

**Author**: Luis Campoy  
**Version**: 1.1.0  
**Last Updated**: October 13, 2025
