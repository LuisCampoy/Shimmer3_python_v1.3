
"""
Shimmer3 Client Module
Handles communication with Shimmer3 IMU devices via serial/Bluetooth
Based on Shimmer3 API and protocol specifications
Last revision: 10/14/2025
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
                # Only stop streaming, do not close serial port or cleanup RFCOMM
                if self.connected and self.serial_conn:
                    if hasattr(self, 'streaming') and self.streaming:
                        await self.stop_streaming()
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
            print(f"Setting sampling rate to {sampling_rate} Hz...")
            rate_ok = await self._set_sampling_rate(sampling_rate)
            print(f"Sampling rate set: {rate_ok}")
            
            # Set enabled sensors
            print(f"Setting sensors bitmap: 0x{sensor_bitmap:X}...")
            sensors_ok = await self._set_sensors(sensor_bitmap)
            print(f"Sensors set: {sensors_ok}")
            
            # Calculate packet size
            self._calculate_packet_size()
            
            print(f"Configuration complete - sensors: {sensors}, rate: {sampling_rate} Hz, packet_size: {self.packet_size}")
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
        print(f"start_streaming called, current state: {self.state}")
        if self.state != ShimmerState.CONNECTED:
            print(f"ERROR: Device not in CONNECTED state (state={self.state})")
            self.logger.error(f"Device not connected (state={self.state})")
            return False
        
        try:
            # Send start streaming command (raw send then single ACK wait)
            print(f"Sending START_STREAMING command (0x{self.COMMANDS['START_STREAMING_COMMAND']:02X})...")
            self.logger.info(f"Sending START_STREAMING command (0x{self.COMMANDS['START_STREAMING_COMMAND']:02X})...")
            ack = await self._send_raw_command(self.COMMANDS['START_STREAMING_COMMAND'])
            print(f"ACK received: {ack}")
            if ack:
                self.state = ShimmerState.STREAMING
                self.packet_counter = 0
                self.data_buffer.clear()
                print("Started streaming (ACK received)")
                self.logger.info("Started streaming (ACK received)")
                return True
            
            print("Failed to start streaming (no ACK received)")
            self.logger.error("Failed to start streaming (no ACK received)")
            return False
            
        except Exception as e:
            self.logger.error(f"Error starting streaming: {e}")
            return False

    async def _wait_for_ack(self, timeout: float = 2.0) -> bool:
        """Wait for ACK byte (0xFF)."""
        if not self.serial_conn or not self.serial_conn.is_open:
            return False
        start = time.time()
        while time.time() - start < timeout:
            if self.serial_conn.in_waiting:
                resp = self.serial_conn.read(1)
                if resp and resp[0] == self.RESPONSES['ACK_COMMAND_PROCESSED']:
                    return True
            await asyncio.sleep(0.01)
        return False

    async def _send_command(self, command: int) -> bool:
        """Send single-byte command and wait for ACK."""
        if not self.serial_conn or not self.serial_conn.is_open:
            return False
        try:
            self.serial_conn.write(bytes([command]))
            self.serial_conn.flush()
            return await self._wait_for_ack()
        except Exception as e:
            self.logger.error(f"_send_command error: {e}")
            return False

    async def _send_raw_command(self, command: int, data: bytes = b'') -> bool:
        """Send command followed by raw payload then wait for ACK."""
        if not self.serial_conn or not self.serial_conn.is_open:
            return False
        try:
            self.serial_conn.write(bytes([command]))
            if data:
                self.serial_conn.write(data)
            self.serial_conn.flush()
            return await self._wait_for_ack()
        except Exception as e:
            self.logger.error(f"_send_raw_command error: {e}")
            return False

    async def diagnostic_stream_dump(self, seconds: float = 5.0, max_bytes: int = 8192) -> Dict[str, Any]:
        """Capture raw bytes after issuing START_STREAMING for diagnostics without parsing.
        Args:
            seconds: duration to capture after start
            max_bytes: cap on data collected
        Returns: dict with byte count and hex preview
        """
        result = {"bytes": 0, "hex_preview": "", "truncated": False, "started": False, "buffer": b""}
        if self.state != ShimmerState.CONNECTED:
            self.logger.error("Cannot diagnostic stream dump: device not CONNECTED")
            return result
        # Attempt to start streaming
        started = await self.start_streaming()
        result["started"] = started
        if not started:
            self.logger.error("diagnostic_stream_dump could not start streaming")
            return result
        start_time = time.time()
        buf = bytearray()
        while time.time() - start_time < seconds:
            try:
                if self.serial_conn and self.serial_conn.in_waiting:
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
                await asyncio.sleep(0.01)
            except Exception as e:
                self.logger.error(f"Error during diagnostic_stream_dump: {e}")
                break
        result["bytes"] = len(buf)
        result["hex_preview"] = buf[:64].hex()
        result["buffer"] = bytes(buf)
        self.logger.info(f"Diagnostic stream dump complete: bytes={result['bytes']}, truncated={result['truncated']}, started={started}")
        # Attempt to stop streaming gracefully
        await self.stop_streaming()
        return result

    async def diagnostic_stream_and_parse(self, seconds: float = 5.0, max_bytes: int = 8192,
                                          candidate_sizes: Optional[List[int]] = None) -> Dict[str, Any]:
        """Capture raw bytes then attempt to infer packet size and parse sensor values.
        Heuristic: look for 0x00 header occurrences and test candidate packet sizes.
        Returns dict with chosen_size, packets_decoded, sample_packets (up to 5), and remainder bytes.
        """
        if candidate_sizes is None:
            # plausible sizes: header + (accel+gyro+mag=18) + optional battery(2) + maybe packet ID(1) => range 19-24
            candidate_sizes = list(range(18, 26))
        dump = await self.diagnostic_stream_dump(seconds=seconds, max_bytes=max_bytes)
        raw_len = dump.get('bytes', 0)
        analysis = {
            'raw_bytes': raw_len,
            'candidate_sizes': candidate_sizes,
            'size_scores': {},
            'chosen_size': None,
            'packets_decoded': 0,
            'sample_packets': [],
            'remainder': 0
        }
        # No data case
        if raw_len == 0:
            self.logger.warning("diagnostic_stream_and_parse: no raw bytes captured")
            return analysis
        raw = dump.get('buffer', b'')
        if not raw:
            analysis['raw_bytes'] = 0
            return analysis
        analysis['raw_bytes'] = len(raw)

        # Score candidate sizes
        for size in candidate_sizes:
            if size <= 0:
                continue
            packets = 0
            idx = 0
            while idx + size <= len(raw):
                if raw[idx] == 0x00:  # header match
                    packets += 1
                    idx += size
                else:
                    # shift until next possible header
                    idx += 1
            remainder = len(raw) - (packets * size)
            # heuristic score: prefer many packets & small remainder
            score = packets - (remainder / max(1, size))
            analysis['size_scores'][size] = {
                'packets': packets,
                'remainder': remainder,
                'score': score
            }
        # Choose size with highest score
        chosen = max(analysis['size_scores'], key=lambda s: analysis['size_scores'][s]['score']) if analysis['size_scores'] else None
        analysis['chosen_size'] = chosen
        if chosen:
            packets = []
            idx = 0
            while idx + chosen <= len(raw) and len(packets) < 5:
                if raw[idx] == 0x00:
                    pkt = raw[idx:idx+chosen]
                    packets.append(pkt)
                    idx += chosen
                else:
                    idx += 1
            analysis['packets_decoded'] = analysis['size_scores'][chosen]['packets']
            analysis['remainder'] = analysis['size_scores'][chosen]['remainder']
            # Parse sample packets into 16-bit words after header
            parsed_samples = []
            for pkt in packets:
                words = []
                payload = pkt[1:]
                for i in range(0, len(payload), 2):
                    if i + 2 <= len(payload):
                        val = struct.unpack('<H', payload[i:i+2])[0]
                        words.append(val)
                parsed_samples.append({'hex': pkt.hex(), 'words': words})
            analysis['sample_packets'] = parsed_samples
        self.logger.info(f"diagnostic_stream_and_parse: chosen_size={analysis['chosen_size']} packets={analysis['packets_decoded']} remainder={analysis['remainder']}")
        return analysis
    
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
        """Set packet size using diagnostic-derived constant for current sensor set."""
        # If accel+gyro+mag all enabled we observed 22-byte packets with leading 0x00 and 10 words (20 bytes) following 2-byte header.
        enabled = set(self.enabled_sensors)
        if {SensorType.ACCELEROMETER, SensorType.GYROSCOPE, SensorType.MAGNETOMETER}.issubset(enabled):
            self.packet_size = 22
        else:
            # Fallback estimation: 2-byte header + per-sensor payloads (6 bytes each axis sensor, 2 battery) + optional meta/checksum (4)
            size = 2
            for s in enabled:
                if s in (SensorType.ACCELEROMETER, SensorType.GYROSCOPE, SensorType.MAGNETOMETER):
                    size += 6
                elif s == SensorType.BATTERY:
                    size += 2
            # Add meta+checksum if at least two axis groups (heuristic)
            axis_groups = len([s for s in enabled if s in (SensorType.ACCELEROMETER, SensorType.GYROSCOPE, SensorType.MAGNETOMETER)])
            if axis_groups >= 2:
                size += 4
            self.packet_size = size
        self.logger.debug(f"Packet size set to {self.packet_size}")
    
    async def _parse_data_packet(self) -> Optional[Dict[str, Any]]:
        """Parse data packet from buffer"""
        if len(self.data_buffer) < self.packet_size:
            return None
        try:
            # Scan for header (first byte 0x00). We treat first two bytes as header word.
            buf = self.data_buffer
            max_start = len(buf) - self.packet_size
            for start in range(max_start + 1):
                if buf[start] == 0x00:
                    # Candidate packet slice
                    packet = bytes(buf[start:start + self.packet_size])
                    # Consume buffer up to end of packet
                    self.data_buffer = buf[start + self.packet_size:]
                    header_word = struct.unpack('<H', packet[:2])[0]
                    payload = packet[2:]
                    parsed = await self._parse_sensor_data(payload)
                    parsed['header_word'] = header_word
                    return parsed
            # If no header yet, discard leading noise bytes before first 0x00 to avoid growth
            first_zero = None
            for i, b in enumerate(buf):
                if b == 0x00:
                    first_zero = i; break
            if first_zero is not None and first_zero > 0:
                del self.data_buffer[:first_zero]
            elif first_zero is None:
                # If no zero byte at all and buffer large, trim some noise
                if len(self.data_buffer) > self.packet_size * 2:
                    self.data_buffer = self.data_buffer[-self.packet_size:]
            return None
        except Exception as e:
            self.logger.error(f"Error parsing data packet: {e}")
            return None
    

    async def _parse_sensor_data(self, data: bytes) -> dict:
        """Parse 20-byte payload: 3x3 axes (accel, gyro, mag), meta, checksum."""
        result = {'timestamp': time.time(), 'packet_id': getattr(self, 'packet_counter', 0)}
        offset = 0
        try:
            # Each axes group: 3x 16-bit little-endian
            def read_axes():
                nonlocal offset
                x, y, z = struct.unpack('<HHH', data[offset:offset+6])
                offset += 6
                return x, y, z
            # Parse in order: accel, gyro, mag
            if offset + 6 <= len(data):
                ax, ay, az = read_axes()
                result['accelerometer'] = {
                    'x_raw': ax, 'y_raw': ay, 'z_raw': az,
                    'x': self._convert_accel(ax),
                    'y': self._convert_accel(ay),
                    'z': self._convert_accel(az)
                }
            if offset + 6 <= len(data):
                gx, gy, gz = read_axes()
                result['gyroscope'] = {
                    'x_raw': gx, 'y_raw': gy, 'z_raw': gz,
                    'x': self._convert_gyro(gx),
                    'y': self._convert_gyro(gy),
                    'z': self._convert_gyro(gz)
                }
            if offset + 6 <= len(data):
                mx, my, mz = read_axes()
                result['magnetometer'] = {
                    'x_raw': mx, 'y_raw': my, 'z_raw': mz,
                    'x': self._convert_mag(mx),
                    'y': self._convert_mag(my),
                    'z': self._convert_mag(mz)
                }
            # Meta word
            if offset + 2 <= len(data):
                meta = struct.unpack('<H', data[offset:offset+2])[0]
                result['meta_word'] = meta
                offset += 2
            # Checksum word
            if offset + 2 <= len(data):
                checksum = struct.unpack('<H', data[offset:offset+2])[0]
                result['checksum_word'] = checksum
                offset += 2
        except Exception as e:
            self.logger.error(f"Error parsing sensor data: {e}")
        return result
    
    def _convert_gyro(self, raw_value: int) -> float:
        """Convert raw gyroscope value to degrees/second"""
        if raw_value > 32767:
            raw_value -= 65536
        sensitivity = 131.0  # LSB/(°/s) for ±250°/s range
        return raw_value / sensitivity

    def _convert_accel(self, raw_value: int) -> float:
        """Convert raw accelerometer value to g units (assuming ±2g, 16-bit, 16384 LSB/g)."""
        if raw_value > 32767:
            raw_value -= 65536
        sensitivity = 16384.0  # LSB/g for ±2g
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