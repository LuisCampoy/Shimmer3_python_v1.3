"""
Shimmer3 Client Module
Handles communication with Shimmer3 IMU devices via serial/Bluetooth
Based on Shimmer3 API and protocol specifications
Last revision: 10/10/2025
"""

import serial
import asyncio
import logging
import struct
import time
import os
import subprocess
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from utils.config_helpers import safe_config_get

try:
    from src.bluetooth_manager import BluetoothManager
except ImportError:
    from bluetooth_manager import BluetoothManager

class ShimmerState(Enum):
    """Shimmer device states"""
    DISCONNECTED = 0
    CONNECTED = 1
    STREAMING = 2

class SensorType(Enum):
    """Available sensor types"""
    ACCELEROMETER = 0x80
    GYROSCOPE = 0x40  
    MAGNETOMETER = 0x20
    BATTERY = 0x2000
    EXT_ADC_A7 = 0x02
    EXT_ADC_A6 = 0x01
    EXT_ADC_A15 = 0x0800


@dataclass
class SensorData:
    """Data structure for sensor readings"""
    timestamp: float
    packet_id: int
    accelerometer: Optional[Dict[str, float]] = None
    gyroscope: Optional[Dict[str, float]] = None
    magnetometer: Optional[Dict[str, float]] = None
    battery: Optional[float] = None


class ShimmerClient:
    """
    Shimmer3 device client for data streaming and configuration
    """
    
    # Shimmer3 Commands
    COMMANDS = {
        'GET_INQUIRY_COMMAND': 0x01,
        'GET_SAMPLING_RATE_COMMAND': 0x03,
        'SET_SAMPLING_RATE_COMMAND': 0x05,
        'TOGGLE_LED_COMMAND': 0x06,
        'START_STREAMING_COMMAND': 0x07,
        'SET_SENSORS_COMMAND': 0x08,
        'SET_5V_REGULATOR_COMMAND': 0x09,
        'SET_PMUX_COMMAND': 0x0A,
        'SET_CONFIG_BYTE0_COMMAND': 0x0B,
        'GET_CONFIG_BYTE0_COMMAND': 0x0C,
        'GET_CALIB_DUMP_COMMAND': 0x2C,
        'STOP_STREAMING_COMMAND': 0x20,
        'SET_ACCEL_RANGE_COMMAND': 0x21,
        'GET_ACCEL_RANGE_COMMAND': 0x22,
        'SET_GSR_RANGE_COMMAND': 0x23,
        'GET_GSR_RANGE_COMMAND': 0x24,
        'GET_SHIMMER_VERSION_COMMAND': 0x3F,
    }
    
    # Response bytes
    RESPONSES = {
        'ACK_COMMAND_PROCESSED': 0xFF,
        'INQUIRY_RESPONSE': 0x02,
        'SAMPLING_RATE_RESPONSE': 0x04,
        'CONFIG_BYTE0_RESPONSE': 0x0D,
        'CALIB_DUMP_RESPONSE': 0x2D,
        'ACCEL_RANGE_RESPONSE': 0x25,
        'GSR_RANGE_RESPONSE': 0x26,
        'SHIMMER_VERSION_RESPONSE': 0x40,
    }
    
    def __init__(self, config, device_id=None):  # Add device_id parameter with default
        """Initialize Shimmer3 client with configuration"""
        self.config = config
        
        # Use helper for consistent config access
        self.device_id = device_id or safe_config_get(config, 'device_id', 'shimmer3_default')
        self.port = safe_config_get(config, 'port', '/dev/rfcomm0')
        self.baud_rate = safe_config_get(config, 'baud_rate', 115200)
        self.timeout = safe_config_get(config, 'timeout', 5)
        self.sampling_rate = safe_config_get(config, 'sampling_rate', 51.2)
        self.sensors = safe_config_get(config, 'sensors', ['accelerometer', 'gyroscope', 'magnetometer'])
        self.device_address = safe_config_get(config, 'device_address', None)
        # New connection behavior flags
        self.manual_rfcomm = safe_config_get(config, 'manual_rfcomm', False)
        self.keep_open_on_handshake_failure = safe_config_get(config, 'keep_open_on_handshake_failure', True)
        self.passive_connect = safe_config_get(config, 'passive_connect', False)  # if True, open port and do not send handshake
        self.verbose_serial = safe_config_get(config, 'verbose_serial', False)    # if True, log raw incoming bytes during connect
        # Connection preservation & keepalive
        self.preserve_connection = safe_config_get(config, 'preserve_connection', False)  # if True, skip automatic disconnect/serial close
        self.keepalive_enabled = safe_config_get(config, 'keepalive_enabled', False)  # if True, send periodic lightweight commands to keep BT link active
        self.keepalive_interval = safe_config_get(config, 'keepalive_interval_seconds', 10.0)
        self._keepalive_task = None
        
        # Initialize Bluetooth manager
        self.bluetooth_manager = BluetoothManager()
        
        # Connection state
        self.serial_conn = None  # unified serial connection handle
        self.connected = False
        self.streaming = False
        self.state = ShimmerState.DISCONNECTED
        
        # Data buffer
        self.data_buffer = []
        self.enabled_sensors = []  # initialize list to avoid attribute errors
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"ShimmerClient initialized for device {self.device_id}")

    async def connect(self):
        """Connect to Shimmer3 device with optional manual RFCOMM management.

        Behaviors:
        - If manual_rfcomm=True: assumes /dev/rfcommX already bound; skips bluetooth_manager.
        - If handshake (inquiry/version) fails and keep_open_on_handshake_failure=True, leaves port open for manual interaction.
        """
        try:
            self.logger.info("Attempting to connect to Shimmer3 device")

            # Determine RFCOMM device path
            if self.manual_rfcomm:
                self.logger.info("Manual RFCOMM mode enabled (manual_rfcomm=True); skipping bluetooth preparation")
                if not os.path.exists(self.port):
                    raise ConnectionError(f"RFCOMM device {self.port} does not exist. Bind it manually then retry.")
            else:
                device_address = self.config.get('device_address', '00:06:66:B1:4D:A1')
                self.logger.info(f"Preparing RFCOMM connection for device {device_address}")
                rfcomm_device = self.bluetooth_manager.prepare_shimmer3_connection(device_address)
                if not rfcomm_device:
                    raise ConnectionError(f"Failed to prepare RFCOMM connection for device {device_address}")
                self.port = rfcomm_device

            # Log current rfcomm show output (best effort)
            try:
                show_out = subprocess.run(['rfcomm', 'show', self.port.replace('/dev/rfcomm','')], capture_output=True, text=True, timeout=3)
                if show_out.returncode == 0:
                    self.logger.debug(f"rfcomm show output before opening: {show_out.stdout.strip()}")
            except Exception as sub_e:
                self.logger.debug(f"Could not get rfcomm show output: {sub_e}")

            # Open serial
            self.serial_conn = serial.Serial(self.port, self.baud_rate, timeout=self.timeout)
            if not self.serial_conn.is_open:
                self.serial_conn.open()
            self.logger.info(f"Serial connection established to {self.port}")

            # Allow device settle
            await asyncio.sleep(1.0)
            try:
                self.serial_conn.reset_input_buffer(); self.serial_conn.reset_output_buffer()
            except Exception:
                pass

            if self.passive_connect:
                self.logger.warning("Passive connect enabled: skipping Shimmer inquiry/version handshake.")
                # Optionally read a small amount for diagnostics
                start_time = time.time()
                raw_bytes = bytearray()
                while time.time() - start_time < 2.0:
                    if self.serial_conn.in_waiting:
                        raw_bytes.extend(self.serial_conn.read(self.serial_conn.in_waiting))
                        if self.verbose_serial:
                            self.logger.debug(f"[passive] Read {len(raw_bytes)} bytes so far: {raw_bytes[:64].hex()}...")
                    await asyncio.sleep(0.05)
                if raw_bytes:
                    self.logger.info(f"Passive connect captured {len(raw_bytes)} bytes initial data")
                self.connected = True
                self.state = ShimmerState.CONNECTED
                # Start keepalive if enabled even in passive mode (will send version command silently)
                if self.keepalive_enabled:
                    self._start_keepalive_loop()
                return True
            else:
                # Attempt inquiry + version handshake only if not disabled
                inquiry = await self._send_and_expect(
                    self.COMMANDS['GET_INQUIRY_COMMAND'],
                    self.RESPONSES['INQUIRY_RESPONSE'],
                    response_len=0,
                    timeout=3.0,
                )
                if inquiry is not None:
                    self.logger.info("Handshake success: inquiry response received")
                    self.connected = True
                    self.state = ShimmerState.CONNECTED
                    if self.keepalive_enabled:
                        self._start_keepalive_loop()
                    return True

                version = await self._send_and_expect(
                    self.COMMANDS['GET_SHIMMER_VERSION_COMMAND'],
                    self.RESPONSES['SHIMMER_VERSION_RESPONSE'],
                    response_len=1,
                    timeout=3.0,
                )
                if version is not None:
                    self.logger.info(f"Handshake partial success: version response {version[0]}")
                    self.connected = True
                    self.state = ShimmerState.CONNECTED
                    if self.keepalive_enabled:
                        self._start_keepalive_loop()
                    return True

                msg = "No inquiry/version response received"
                if self.keep_open_on_handshake_failure:
                    self.logger.warning(msg + "; leaving port open (keep_open_on_handshake_failure=True)")
                    self.connected = True  # treat as connected so user can attempt manual commands
                    self.state = ShimmerState.CONNECTED
                    if self.keepalive_enabled:
                        self._start_keepalive_loop()
                    return True
                else:
                    raise ConnectionError(msg)

        except Exception as e:
            self.logger.error(f"Connection failed: {e}")
            # Only close if we are not explicitly keeping it open for diagnostics
            if (not self.keep_open_on_handshake_failure) and self.serial_conn and self.serial_conn.is_open:
                try:
                    self.serial_conn.close()
                except Exception:
                    pass
            return False
    
    async def disconnect(self):
        """Disconnect from Shimmer3 device and cleanup RFCOMM"""
        try:
            if self.preserve_connection:
                self.logger.info("preserve_connection=True; skipping disconnect to keep RFCOMM binding and serial open")
                return
            if self.connected and self.serial_conn:
                # Stop streaming if active
                if hasattr(self, 'streaming') and self.streaming:
                    await self.stop_streaming()
                
                # Close serial connection
                self.serial_conn.close()
                self.logger.info("Serial connection closed")
            
            # Cleanup RFCOMM bindings
            self.bluetooth_manager.cleanup_connections()
            
            self.connected = False
            self.state = ShimmerState.DISCONNECTED
            self.logger.info("Shimmer3 device disconnected and cleaned up")
            
        except Exception as e:
            self.logger.error(f"Error during disconnection: {e}")

    def _start_keepalive_loop(self):
        """Start background keepalive loop if not already running."""
        if self._keepalive_task and not self._keepalive_task.done():
            return
        self.logger.info(f"Starting keepalive loop every {self.keepalive_interval}s (enabled={self.keepalive_enabled})")
        self._keepalive_task = asyncio.create_task(self._keepalive_loop())

    async def _keepalive_loop(self):
        consecutive_failures = 0
        while self.keepalive_enabled and self.connected and self.serial_conn and self.serial_conn.is_open:
            try:
                # Send a lightweight version command to nudge link
                resp = await self._send_and_expect(
                    self.COMMANDS['GET_SHIMMER_VERSION_COMMAND'],
                    self.RESPONSES['SHIMMER_VERSION_RESPONSE'],
                    response_len=1,
                    timeout=2.0
                )
                if resp is not None:
                    consecutive_failures = 0
                    self.logger.debug(f"Keepalive success (version={resp[0]})")
                else:
                    consecutive_failures += 1
                    if consecutive_failures == 1:
                        self.logger.warning("Keepalive failed (no version response)")
                    elif consecutive_failures % 3 == 0:
                        self.logger.warning(f"Keepalive failing repeatedly (failures={consecutive_failures})")
                await asyncio.sleep(self.keepalive_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Keepalive loop error: {e}")
                await asyncio.sleep(self.keepalive_interval)
        self.logger.info("Keepalive loop ending")

    def enable_preserve(self):
        """Public method to enable connection preservation at runtime."""
        self.preserve_connection = True
        self.logger.info("Connection preservation enabled (will skip disconnect)")
    
    async def configure_sensors(self, sensors: List[str], sampling_rate: float = 51.2) -> bool:
        """
        Configure enabled sensors and sampling rate
        
        Args:
            sensors: List of sensor names to enable
            sampling_rate: Sampling rate in Hz
            
        Returns:
            bool: True if configuration successful
        """
        if self.state != ShimmerState.CONNECTED:
            self.logger.error("Device not connected")
            return False
        
        try:
            # Convert sensor names to sensor types
            sensor_map = {
                'accelerometer': SensorType.ACCELEROMETER,
                'gyroscope': SensorType.GYROSCOPE, 
                'magnetometer': SensorType.MAGNETOMETER,
                'battery': SensorType.BATTERY
            }
            
            self.enabled_sensors = []
            sensor_bitmap = 0
            
            for sensor_name in sensors:
                if sensor_name in sensor_map:
                    sensor_type = sensor_map[sensor_name]
                    self.enabled_sensors.append(sensor_type)
                    sensor_bitmap |= sensor_type.value
                else:
                    self.logger.warning(f"Unknown sensor: {sensor_name}")
            
            # Set sampling rate
            await self._set_sampling_rate(sampling_rate)
            
            # Set enabled sensors
            await self._set_sensors(sensor_bitmap)
            
            # Calculate packet size
            self._calculate_packet_size()
            
            self.logger.info(f"Configured sensors: {sensors}, sampling rate: {sampling_rate} Hz")
            return True
            
        except Exception as e:
            self.logger.error(f"Sensor configuration failed: {e}")
            return False

    async def passive_sniff(self, seconds: float = 5.0, max_bytes: int = 2048) -> Dict[str, Any]:
        """Sniff raw bytes from the serial port without sending any commands.
        Args:
            seconds: duration to listen
            max_bytes: cap on bytes to collect
        Returns: dict with count and a hex preview
        """
        result = {"bytes_captured": 0, "hex_preview": "", "truncated": False}
        if not self.serial_conn or not self.serial_conn.is_open:
            self.logger.error("passive_sniff called but serial not open")
            return result
        start = time.time()
        buf = bytearray()
        while time.time() - start < seconds:
            try:
                if self.serial_conn.in_waiting:
                    chunk = self.serial_conn.read(self.serial_conn.in_waiting)
                    if chunk:
                        remaining = max_bytes - len(buf)
                        if remaining <= 0:
                            result["truncated"] = True
                            break
                        if len(chunk) > remaining:
                            buf.extend(chunk[:remaining])
                            result["truncated"] = True
                            break
                        buf.extend(chunk)
                await asyncio.sleep(0.02)
            except Exception as e:
                self.logger.error(f"Error during passive_sniff: {e}")
                break
        result["bytes_captured"] = len(buf)
        result["hex_preview"] = buf[:64].hex()
        if result["bytes_captured"]:
            self.logger.info(f"Passive sniff captured {result['bytes_captured']} bytes (preview {result['hex_preview']})")
        else:
            self.logger.info("Passive sniff captured 0 bytes")
        return result
    
    async def start_streaming(self) -> bool:
        """
        Start data streaming
        
        Returns:
            bool: True if streaming started successfully
        """
        if self.state != ShimmerState.CONNECTED:
            self.logger.error("Device not connected")
            return False
        
        try:
            # Send start streaming command
            if await self._send_command(self.COMMANDS['START_STREAMING_COMMAND']):
                if await self._wait_for_ack():
                    self.state = ShimmerState.STREAMING
                    self.packet_counter = 0
                    self.data_buffer.clear()
                    self.logger.info("Started streaming")
                    return True
            
            self.logger.error("Failed to start streaming")
            return False
            
        except Exception as e:
            self.logger.error(f"Error starting streaming: {e}")
            return False
    
    async def stop_streaming(self) -> bool:
        """
        Stop data streaming
        
        Returns:
            bool: True if streaming stopped successfully
        """
        if self.state != ShimmerState.STREAMING:
            return True
        
        try:
            # Send stop streaming command
            if await self._send_command(self.COMMANDS['STOP_STREAMING_COMMAND']):
                if await self._wait_for_ack():
                    self.state = ShimmerState.CONNECTED
                    self.logger.info("Stopped streaming")
                    return True
            
            self.logger.error("Failed to stop streaming")
            return False
            
        except Exception as e:
            self.logger.error(f"Error stopping streaming: {e}")
            return False
    
    async def get_data(self) -> Optional[Dict[str, Any]]:
        """
        Get next data packet from stream
        
        Returns:
            Dict containing sensor data or None if no data available
        """
        if self.state != ShimmerState.STREAMING:
            return None
        
        try:
            # Read data from serial port
            if self.serial_conn and self.serial_conn.in_waiting:
                data = self.serial_conn.read(self.serial_conn.in_waiting)
                self.data_buffer.extend(data)
            
            # Parse complete packets from buffer
            return await self._parse_data_packet()
            
        except Exception as e:
            self.logger.error(f"Error reading data: {e}")
            return None
    
    async def _get_device_info(self) -> None:
        """Get device information"""
        try:
            # Get Shimmer version
            if await self._send_command(self.COMMANDS['GET_SHIMMER_VERSION_COMMAND']):
                response = await self._read_response(self.RESPONSES['SHIMMER_VERSION_RESPONSE'], 1)
                if response:
                    version = response[0]
                    self.logger.info(f"Shimmer version: {version}")
            
            # Get calibration data
            if await self._send_command(self.COMMANDS['GET_CALIB_DUMP_COMMAND']):
                response = await self._read_response(self.RESPONSES['CALIB_DUMP_RESPONSE'], 21)
                if response:
                    await self._parse_calibration_data(response)
                    
        except Exception as e:
            self.logger.warning(f"Could not retrieve device info: {e}")
    
    async def _set_sampling_rate(self, rate: float) -> bool:
        """Set sampling rate"""
        try:
            # Convert rate to Shimmer format (clock ticks)
            # Shimmer uses 32.768 kHz clock
            clock_freq = 32768.0
            rate_val = int(clock_freq / rate)
            
            #command_data = struct.pack('<BH', self.COMMANDS['SET_SAMPLING_RATE_COMMAND'], rate_val)

            if await self._send_raw_command(self.COMMANDS['SET_SAMPLING_RATE_COMMAND'], struct.pack('<H', rate_val)):
                if await self._wait_for_ack():
                    self.sampling_rate = rate
                    return True
            return False
            
        except Exception as e:
            self.logger.error(f"Error setting sampling rate: {e}")
            return False
    
    async def _set_sensors(self, sensor_bitmap: int) -> bool:
        """Set enabled sensors"""
        try:
            #command_data = struct.pack('<BI', self.COMMANDS['SET_SENSORS_COMMAND'], sensor_bitmap)
            
            if await self._send_raw_command(self.COMMANDS['SET_SENSORS_COMMAND'], struct.pack('<I', sensor_bitmap)):
                return await self._wait_for_ack()
            return False
            
        except Exception as e:
            self.logger.error(f"Error setting sensors: {e}")
            return False
    
    def _calculate_packet_size(self) -> None:
        """Calculate expected packet size based on enabled sensors"""
        size = 1  # Packet type byte
        
        for sensor in self.enabled_sensors:
            if sensor == SensorType.ACCELEROMETER:
                size += 6  # 3 axes * 2 bytes
            elif sensor == SensorType.GYROSCOPE:
                size += 6  # 3 axes * 2 bytes
            elif sensor == SensorType.MAGNETOMETER:
                size += 6  # 3 axes * 2 bytes
            elif sensor == SensorType.BATTERY:
                size += 2  # 1 value * 2 bytes
        
        self.packet_size = size
        self.logger.debug(f"Expected packet size: {size} bytes")
    
    async def _parse_data_packet(self) -> Optional[Dict[str, Any]]:
        """Parse data packet from buffer"""
        if len(self.data_buffer) < self.packet_size:
            return None
        
        try:
            # Look for packet start (0x00 for data packet)
            while len(self.data_buffer) >= self.packet_size:
                if self.data_buffer[0] == 0x00:  # Data packet identifier
                    # Extract packet
                    packet_data = bytes(self.data_buffer[1:self.packet_size])
                    self.data_buffer = self.data_buffer[self.packet_size:]
                    
                    # Parse sensor data
                    return await self._parse_sensor_data(packet_data)
                else:
                    # Remove invalid byte
                    self.data_buffer.pop(0)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error parsing data packet: {e}")
            return None
    
    async def _parse_sensor_data(self, data: bytes) -> Dict[str, Any]:
        """Parse sensor data from packet"""
        result = {
            'timestamp': time.time(),
            'packet_id': self.packet_counter
        }
        
        self.packet_counter += 1
        offset = 0
        
        try:
            for sensor in self.enabled_sensors:
                if sensor == SensorType.ACCELEROMETER:
                    if offset + 6 <= len(data):
                        x, y, z = struct.unpack('<HHH', data[offset:offset+6])
                        result['accelerometer'] = {
                            'x': self._convert_accel(x),
                            'y': self._convert_accel(y), 
                            'z': self._convert_accel(z)
                        }
                        offset += 6
                
                elif sensor == SensorType.GYROSCOPE:
                    if offset + 6 <= len(data):
                        x, y, z = struct.unpack('<HHH', data[offset:offset+6])
                        result['gyroscope'] = {
                            'x': self._convert_gyro(x),
                            'y': self._convert_gyro(y),
                            'z': self._convert_gyro(z)
                        }
                        offset += 6
                
                elif sensor == SensorType.MAGNETOMETER:
                    if offset + 6 <= len(data):
                        x, y, z = struct.unpack('<HHH', data[offset:offset+6])
                        result['magnetometer'] = {
                            'x': self._convert_mag(x),
                            'y': self._convert_mag(y),
                            'z': self._convert_mag(z)
                        }
                        offset += 6
                
                elif sensor == SensorType.BATTERY:
                    if offset + 2 <= len(data):
                        battery_raw = struct.unpack('<H', data[offset:offset+2])[0]
                        result['battery'] = self._convert_battery(battery_raw)
                        offset += 2
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error parsing sensor data: {e}")
            return result
    
    def _convert_accel(self, raw_value: int) -> float:
        """Convert raw accelerometer value to m/s²"""
        # Convert from unsigned to signed
        if raw_value > 32767:
            raw_value -= 65536
        
        # Convert to g-force then to m/s²
        sensitivity = 1365.0  # LSB/g for ±2g range
        g_force = raw_value / sensitivity
        return g_force * 9.81  # Convert to m/s²
    
    def _convert_gyro(self, raw_value: int) -> float:
        """Convert raw gyroscope value to degrees/second"""
        if raw_value > 32767:
            raw_value -= 65536
        
        sensitivity = 131.0  # LSB/(°/s) for ±250°/s range
        return raw_value / sensitivity
    
    def _convert_mag(self, raw_value: int) -> float:
        """Convert raw magnetometer value to µT"""
        if raw_value > 32767:
            raw_value -= 65536
        
        sensitivity = 1090.0  # LSB/Ga for ±1.3 Ga range  
        gauss = raw_value / sensitivity
        return gauss * 100.0  # Convert Gauss to µT
    
    def _convert_battery(self, raw_value: int) -> float:
        """Convert raw battery value to voltage"""
        return (raw_value / 4095.0) * 3.0  # 12-bit ADC, 3V reference
    
    async def _parse_calibration_data(self, data: bytes) -> None:
        """Parse calibration data"""
        try:
            # This is a simplified version - actual calibration is more complex
            self.calibration_data = {
                'accelerometer': {
                    'offset': [0, 0, 0],
                    'sensitivity': [1, 1, 1]
                },
                'gyroscope': {
                    'offset': [0, 0, 0], 
                    'sensitivity': [1, 1, 1]
                },
                'magnetometer': {
                    'offset': [0, 0, 0],
                    'sensitivity': [1, 1, 1]
                }
            }
            self.logger.info("Calibration data parsed")
            
        except Exception as e:
            self.logger.warning(f"Could not parse calibration data: {e}")
    
    async def _send_command(self, command: int) -> bool:
        """Send command to Shimmer device"""
        if not self.serial_conn or not self.serial_conn.is_open:
            return False
        
        try:
            self.serial_conn.write(bytes([command]))
            self.serial_conn.flush()
            return await self._wait_for_ack()
        except Exception as e:
            self.logger.error(f"Error sending command: {e}")
            return False

    async def _send_raw_command(self, command: int, data: bytes = b'') -> bool:
        """Send raw command with optional data to Shimmer device"""
        if not self.serial_conn or not self.serial_conn.is_open:
            return False
        
        try:
            # Send command
            self.serial_conn.write(bytes([command]))
            if data:
                self.serial_conn.write(data)
            self.serial_conn.flush()
            return await self._wait_for_ack()
        except Exception as e:
            self.logger.error(f"Error sending raw command: {e}")
            return False

    async def _wait_for_ack(self, timeout: float = 2.0) -> bool:
        """Wait for ACK response"""
        if not self.serial_conn:
            return False
        
        start_time = time.time()
        while (time.time() - start_time) < timeout:
            if self.serial_conn.in_waiting > 0:
                response = self.serial_conn.read(1)
                if response and response[0] == self.RESPONSES['ACK_COMMAND_PROCESSED']:
                    return True
            await asyncio.sleep(0.01)
        
        return False
    
    async def _read_response(self, expected_response: int, length: int, timeout: float = 2.0) -> Optional[bytes]:
        """Read response with expected format"""
        start_time = time.time()
        buffer = bytearray()
        
        while time.time() - start_time < timeout:
            if self.serial_conn and self.serial_conn.in_waiting:
                data = self.serial_conn.read(self.serial_conn.in_waiting)
                buffer.extend(data)
                
                if len(buffer) >= length + 1:  # +1 for response byte
                    if buffer[0] == expected_response:
                        return bytes(buffer[1:length+1])
                    else:
                        # Remove invalid byte and continue
                        buffer.pop(0)
            
            await asyncio.sleep(0.01)
        
        return None

    async def _send_and_expect(self, command: int, expected_resp: int, response_len: int = 0, timeout: float = 2.0) -> Optional[bytes]:
        """Send a command and wait for a specific response header (and payload length)."""
        try:
            # Write command
            if not self.serial_conn or not self.serial_conn.is_open:
                return None
            self.serial_conn.write(bytes([command]))
            self.serial_conn.flush()

            # Wait for response with header matching expected_resp
            resp = await self._read_response(expected_resp, response_len, timeout=timeout)
            return resp
        except Exception as e:
            self.logger.error(f"_send_and_expect error: {e}")
            return None