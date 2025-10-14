# Data Logger Module
# Handles logging of Shimmer3 IMU data to various file formats
# Supports CSV, JSON, and binary formats with buffering and rotation
# Last revision: 10/12/2025

import asyncio
import csv
import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
import threading
from queue import Queue, Empty
import gzip
import pickle
import pandas as pd
from utils.config_helpers import safe_config_get

@dataclass
class LogEntry:
    """Data structure for a single log entry"""
    timestamp: float
    packet_id: int
    accel_x: Optional[float] = None
    accel_y: Optional[float] = None
    accel_z: Optional[float] = None
    gyro_x: Optional[float] = None
    gyro_y: Optional[float] = None
    gyro_z: Optional[float] = None
    mag_x: Optional[float] = None
    mag_y: Optional[float] = None
    mag_z: Optional[float] = None
    battery: Optional[float] = None
    temperature: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    def to_csv_row(self) -> List[Any]:
        """Convert to CSV row"""
        return [
            self.timestamp, self.packet_id, self.accel_x, self.accel_y, self.accel_z,
            self.gyro_x, self.gyro_y, self.gyro_z, self.mag_x, self.mag_y, self.mag_z,
            self.battery, self.temperature
        ]

class DataLogger:
    # CSV column headers
    CSV_HEADERS = [
        'timestamp', 'packet_id', 'accel_x', 'accel_y', 'accel_z',
        'gyro_x', 'gyro_y', 'gyro_z', 'mag_x', 'mag_y', 'mag_z',
        'battery', 'temperature'
    ]
    
    def __init__(self, output_dir: str, file_format: str = 'csv', 
                 buffer_size: int = 1000, max_file_size_mb: int = 100, config: Any = None):
        """Initialize DataLogger with configuration"""
        self.output_dir = Path(output_dir)
        self.file_format = file_format
        self.buffer_size = buffer_size
        self.max_file_size_mb = max_file_size_mb
        self.max_file_size = max_file_size_mb * 1024 * 1024  # Convert to bytes
        self.config = config
        
        # Use consistent config access
        self.auto_flush_interval = safe_config_get(config, 'data.auto_flush_interval', 30)
        self.compression_enabled = safe_config_get(config, 'data.compression', False)
        self.compression = self.compression_enabled  # Alias for compatibility
        self.backup_enabled = safe_config_get(config, 'data.backup_enabled', True)
        
        # Initialize core attributes
        self.buffer = []  # Main data buffer
        self.data_buffer = []  # Alias for compatibility
        self.buffer_lock = threading.Lock()
        self.current_file = None
        self.current_writer = None
        self.current_file_handle = None  # Ensure attribute exists before use
        self.file_counter = 0
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.session_start_time = datetime.now()
        self.logging_active = False
        self.last_flush_time = time.time()  # Initialize flush timer
        self.file_entries_written = 0
        
        # Statistics tracking
        self.stats = {
            'total_entries': 0,
            'entries_since_last_export': 0,
            'total_exports': 0,
            'files_created': 0,
            'bytes_written': 0,
            'session_start': self.session_start_time,
            'last_entry_time': None,
            'last_export_time': None
        }
        
        # Queue for async logging
        self.log_queue = Queue()
        
        self.logger = logging.getLogger(__name__)
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    async def log_data(self, data_packet: Dict[str, Any]) -> None:
        """
        Log a data packet (async interface)
        
        Args:
            data_packet: Data packet from ShimmerClient
        """
        try:
            self.logger.debug(f"log_data called with packet: {data_packet}")
            # Convert data packet to LogEntry
            entry = self._create_log_entry(data_packet)

            # Add to async queue
            self.log_queue.put(entry)

            # Start logging thread if not active
            if not self.logging_active:
                self._start_logging_thread()

        except Exception as e:
            self.logger.error(f"Error logging data: {e}")
    
    def log_data_sync(self, data_packet: Dict[str, Any]) -> None:
        """
        Log a data packet (synchronous interface)
        
        Args:
            data_packet: Data packet from ShimmerClient
        """
        try:
            entry = self._create_log_entry(data_packet)
            self._add_to_buffer(entry)
            
        except Exception as e:
            self.logger.error(f"Error logging data: {e}")
    
    def _create_log_entry(self, data_packet: Dict[str, Any]) -> LogEntry:
        """Create LogEntry from data packet"""
        entry = LogEntry(
            timestamp=data_packet.get('timestamp', time.time()),
            packet_id=data_packet.get('packet_id', 0)
        )
        
        # Extract accelerometer data
        accel = data_packet.get('accelerometer', {})
        if accel:
            entry.accel_x = accel.get('x')
            entry.accel_y = accel.get('y')
            entry.accel_z = accel.get('z')

        # Extract gyroscope data
        gyro = data_packet.get('gyroscope', {})
        if gyro:
            entry.gyro_x = gyro.get('x')
            entry.gyro_y = gyro.get('y')
            entry.gyro_z = gyro.get('z')

        # Extract magnetometer data
        mag = data_packet.get('magnetometer', {})
        if mag:
            entry.mag_x = mag.get('x')
            entry.mag_y = mag.get('y')
            entry.mag_z = mag.get('z')

        # Extract battery data
        entry.battery = data_packet.get('battery')
        
        return entry
    
    def _start_logging_thread(self) -> None:
        """Start background logging thread"""
        if not self.logging_active:
            self.logging_active = True
            self.logging_thread = threading.Thread(target=self._logging_worker, daemon=True)
            self.logging_thread.start()
            self.logger.debug("Logging thread started")
    
    def _logging_worker(self) -> None:
        """Background worker for processing log queue"""
        while self.logging_active:
            try:
                # Get entry from queue with timeout
                entry = self.log_queue.get(timeout=1.0)
                self._add_to_buffer(entry)
                self.log_queue.task_done()
                
            except Empty:
                # Check for auto-flush
                if time.time() - self.last_flush_time > self.auto_flush_interval:
                    self._flush_buffer()
                continue
            except Exception as e:
                self.logger.error(f"Error in logging worker: {e}")
    
    def _add_to_buffer(self, entry: LogEntry) -> None:
        """Add entry to buffer and flush if needed"""
        with self.buffer_lock:
            self.buffer.append(entry)
            
            # Update stats
            self.stats['total_entries'] += 1
            self.stats['entries_since_last_export'] += 1
            self.stats['last_entry_time'] = datetime.now()
            
            # Check if buffer is full or auto-flush needed
            if (len(self.buffer) >= self.buffer_size or 
                time.time() - self.last_flush_time > self.auto_flush_interval):
                self._flush_buffer()
    
    def _flush_buffer(self) -> None:
        """Flush buffer to file"""
        if not self.buffer:
            return
        
        buffer_copy = []
        try:
            # Create a copy of buffer for processing
            buffer_copy = self.buffer.copy()
            buffer_length = len(buffer_copy)
            
            # Clear buffer immediately to avoid blocking
            self.buffer.clear()
            
            # Check if file rotation needed
            if self._needs_file_rotation():
                self._rotate_file()
            
            # Write entries based on format
            if self.file_format == 'csv':
                self._write_csv_entries(buffer_copy)
            elif self.file_format == 'json':
                self._write_json_entries(buffer_copy)
            elif self.file_format == 'binary':
                self._write_binary_entries(buffer_copy)
            else:
                raise ValueError(f"Unsupported file format: {self.file_format}")
            
            # Update stats
            self.stats['bytes_written'] += self._estimate_buffer_size_for_entries(buffer_copy)
            self.last_flush_time = time.time()
            
            self.logger.debug(f"Flushed {buffer_length} entries to {self.current_file}")
            
        except Exception as e:
            self.logger.error(f"Error flushing buffer: {e}")
            # Re-add entries to buffer if write failed
            self.buffer.extend(buffer_copy)

    def _estimate_buffer_size_for_entries(self, entries: List[LogEntry]) -> int:
        """Estimate size for specific entries"""
        if not entries:
            return 0
        
        entry_size = 100  # Base estimate
        if self.file_format == 'csv':
            entry_size = 80
        elif self.file_format == 'json':
            entry_size = 150
        elif self.file_format == 'binary':
            entry_size = 120
        
        return len(entries) * entry_size
    
    def _needs_file_rotation(self) -> bool:
        """Check if file rotation is needed"""
        if not self.current_file or not self.current_file.exists():
            return True
        
        try:
            file_size = self.current_file.stat().st_size
            return file_size >= self.max_file_size
        except Exception:
            return True
    
    def _rotate_file(self) -> None:
        """Rotate to a new file"""
        try:
            # Close current file
            self._close_current_file()
            
            # Create new file
            self._create_new_file()
            
            self.logger.info(f"Rotated to new file: {self.current_file}")
            
        except Exception as e:
            self.logger.error(f"Error rotating file: {e}")
    
    def _create_new_file(self) -> None:
        """Create a new log file"""
        try:
            self.file_counter += 1
            
            # Generate filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_name = f"shimmer3_data_{self.session_id}_{self.file_counter:03d}_{timestamp}"
            
            if self.file_format == 'csv':
                filename = f"{base_name}.csv"
            elif self.file_format == 'json':
                filename = f"{base_name}.jsonl"  # JSON Lines format
            elif self.file_format == 'binary':
                filename = f"{base_name}.pkl"
            else:
                filename = f"{base_name}.dat"
            
            if self.compression:
                filename += '.gz'
            
            self.current_file = self.output_dir / filename
            
            # Open file and create writer
            if self.file_format == 'binary':
                # Always open in binary mode
                if self.compression:
                    self.current_file_handle = gzip.open(self.current_file, 'ab')
                else:
                    self.current_file_handle = open(self.current_file, 'ab')
                self.current_writer = None
            elif self.file_format == 'json':
                # Always open in text mode
                if self.compression:
                    import io
                    self.current_file_handle = io.TextIOWrapper(gzip.open(self.current_file, 'wb'), encoding='utf-8')
                else:
                    self.current_file_handle = open(self.current_file, 'w', encoding='utf-8')
                self.current_writer = None
            elif self.file_format == 'csv':
                # Always open in text mode
                if self.compression:
                    self.current_file_handle = gzip.open(self.current_file, 'wt', encoding='utf-8', newline='')
                else:
                    self.current_file_handle = open(self.current_file, 'w', encoding='utf-8', newline='')
                self.current_writer = csv.writer(self.current_file_handle)
                self.current_writer.writerow(self.CSV_HEADERS)
                if self.current_file_handle is not None:
                    self.current_file_handle.flush()
                try:
                    if hasattr(self.current_file_handle, "fileno"):
                        os.fsync(self.current_file_handle.fileno())
                except Exception:
                    pass
            else:
                # Default to text mode
                if self.compression:
                    self.current_file_handle = gzip.open(self.current_file, 'wt', encoding='utf-8')
                else:
                    self.current_file_handle = open(self.current_file, 'w', encoding='utf-8')
                self.current_writer = None
            # Reset per-file counter
            self.file_entries_written = 0
            # Update stats
            self.stats['files_created'] += 1
            # Write metadata
            self._write_metadata()
            self.logger.info(f"Created new log file: {self.current_file}")
        except Exception as e:
            self.logger.error(f"Error creating new file: {e}")
            raise
    
    def _close_current_file(self) -> None:
        """Close current file"""
        try:
            if self.current_file_handle:
                self.current_file_handle.close()
                self.current_file_handle = None
                self.current_writer = None
                
        except Exception as e:
            self.logger.error(f"Error closing file: {e}")
    
    def _write_csv_entries(self, entries: List[LogEntry]) -> None:
        """Write entries in CSV format"""
        if self.current_writer:
            for entry in entries:
                self.current_writer.writerow(entry.to_csv_row())
            if self.current_file_handle is not None:
                self.current_file_handle.flush()
            self.file_entries_written += len(entries)
    
    def _write_json_entries(self, entries: List[LogEntry]) -> None:
        """Write entries in JSON Lines format (always text mode)"""
        if self.current_file_handle:
            # Ensure file handle is text mode
            import io
            if not isinstance(self.current_file_handle, (io.TextIOBase,)) and not (hasattr(self.current_file_handle, 'encoding') and hasattr(self.current_file_handle, 'write')):
                raise RuntimeError("JSON logging requires a text file handle (not binary)")
            for entry in entries:
                json_line = json.dumps(entry.to_dict()) + '\n'
                self.current_file_handle.write(json_line) # type: ignore
            self.current_file_handle.flush()
            self.file_entries_written += len(entries)

    def _write_binary_entries(self, entries: List[LogEntry]) -> None:
        """Write entries in binary format (always binary mode)"""
        if self.current_file_handle:
            import io
            if not isinstance(self.current_file_handle, (io.BufferedIOBase,)) and not (hasattr(self.current_file_handle, 'write') and not hasattr(self.current_file_handle, 'encoding')):
                raise RuntimeError("Binary logging requires a binary file handle (not text)")
            for entry in entries:
                pickle.dump(entry.to_dict(), self.current_file_handle) # type: ignore
            self.current_file_handle.flush()
            self.file_entries_written += len(entries)
    
    def _write_metadata(self) -> None:
        """Write session metadata"""
        try:
            metadata_file = self.output_dir / f"session_metadata_{self.session_id}.json"
            
            metadata = {
                'session_id': self.session_id,
                'session_start': self.session_start_time.isoformat(),
                'file_format': self.file_format,
                'compression': self.compression,
                'buffer_size': self.buffer_size,
                'max_file_size_mb': self.max_file_size / (1024 * 1024),
                'output_directory': str(self.output_dir),
                'stats': self.stats.copy()
            }
            
            # Convert any datetime objects within stats to ISO strings
            for key, value in list(metadata['stats'].items()):
                if isinstance(value, datetime):
                    metadata['stats'][key] = value.isoformat()
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            self.logger.warning(f"Could not write metadata: {e}")
    
    def should_export(self) -> bool:
        """Return True when we should trigger an export (entry count or time-based)."""
        entries = self.stats.get('entries_since_last_export', 0)
        last_export_time = self.stats.get('last_export_time') or datetime.now()
        # thresholds
        min_records = getattr(self, 'min_records_threshold', 1000)
        interval_seconds = getattr(self, 'export_interval_seconds', 60)
        return entries >= min_records or (datetime.now() - last_export_time) >= timedelta(seconds=interval_seconds)

    def mark_export_completed(self) -> None:
        """Reset counters after a successful export."""
        self.stats['last_export_time'] = datetime.now()
        self.stats['entries_since_last_export'] = 0
        self.stats['total_exports'] = self.stats.get('total_exports', 0) + 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get logging statistics"""
        stats = self.stats.copy()
        stats['buffer_size_current'] = len(self.buffer)
        stats['current_file'] = str(self.current_file) if self.current_file else None
        stats['logging_active'] = self.logging_active
        stats['file_entries_written'] = self.file_entries_written
        return stats
    
    async def close(self) -> None:
        """Close logger and cleanup resources"""
        try:
            self.logger.info("Closing DataLogger...")
            
            # Stop logging thread
            self.logging_active = False
            
            if self.logging_thread and self.logging_thread.is_alive():
                # Wait for queue to empty
                self.log_queue.join()
                self.logging_thread.join(timeout=5.0)
            
            # Flush remaining buffer
            with self.buffer_lock:
                if self.buffer:
                    self._flush_buffer()
            
            # Close current file
            self._close_current_file()
            
            # Write final metadata
            self._write_metadata()
            
            self.logger.info(f"DataLogger closed. Total entries logged: {self.stats['total_entries']}")
            
        except Exception as e:
            self.logger.error(f"Error closing DataLogger: {e}")
    
    def force_flush(self) -> None:
        """Force flush buffer to disk"""
        with self.buffer_lock:
            self._flush_buffer()
    
    def get_current_file_path(self) -> Optional[Path]:
        """Get current file path"""
        return self.current_file
    
    def get_session_files(self) -> List[Path]:
        """Get all files for current session"""
        pattern = f"shimmer3_data_{self.session_id}_*"
        return list(self.output_dir.glob(pattern))

    # REMOVE misplaced exporter methods from DataLogger
    async def export_data(self) -> None:
        # This method does not belong here; implemented in main/data_exporter
        raise NotImplementedError("export_data is not a DataLogger responsibility")

    async def _load_single_file(self, file_path: Path) -> Optional[pd.DataFrame]:
        # This method does not belong here; implemented in DataExporter
        raise NotImplementedError("_load_single_file is not a DataLogger responsibility")