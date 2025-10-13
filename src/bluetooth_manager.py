# Bluetooth manager
# Created 10/13/2025
# Last update 10/13/2025

import subprocess
import re
import logging
import time
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)

class BluetoothManager:
    """Manages Bluetooth operations including pairing and connection status"""
    
    def __init__(self):
        self.logger = logger
    
    def check_bluetooth_service(self) -> bool:
        """Check if Bluetooth service is running"""
        try:
            result = subprocess.run(['systemctl', 'is-active', 'bluetooth'], 
                                  capture_output=True, text=True, timeout=10)
            is_active = result.stdout.strip() == 'active'
            
            if not is_active:
                self.logger.warning("Bluetooth service is not active")
                # Try to start bluetooth service
                subprocess.run(['sudo', 'systemctl', 'start', 'bluetooth'], 
                             capture_output=True, text=True, timeout=15)
                time.sleep(2)  # Wait for service to start
                
            return is_active
        except Exception as e:
            self.logger.error(f"Error checking Bluetooth service: {e}")
            return False
    
    def is_device_paired(self, device_address: str) -> bool:
        """Check if a Bluetooth device is already paired"""
        try:
            # Use bluetoothctl to list paired devices
            result = subprocess.run(['bluetoothctl', 'paired-devices'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                self.logger.error(f"Failed to list paired devices: {result.stderr}")
                return False
            
            # Look for the device address in paired devices
            paired_devices = result.stdout
            is_paired = device_address.upper() in paired_devices.upper()
            
            self.logger.info(f"Device {device_address} paired status: {is_paired}")
            return is_paired
            
        except Exception as e:
            self.logger.error(f"Error checking pairing status: {e}")
            return False
    
    def is_device_connected(self, device_address: str) -> bool:
        """Check if a Bluetooth device is currently connected"""
        try:
            result = subprocess.run(['bluetoothctl', 'info', device_address], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                return False
            
            # Check for "Connected: yes" in the output
            return "Connected: yes" in result.stdout
            
        except Exception as e:
            self.logger.error(f"Error checking connection status: {e}")
            return False
    
    def scan_for_device(self, device_address: str, timeout: int = 30) -> bool:
        """Scan for a specific Bluetooth device"""
        try:
            self.logger.info(f"Scanning for device {device_address}...")
            
            # Start scanning
            subprocess.run(['bluetoothctl', 'scan', 'on'], 
                          capture_output=True, text=True, timeout=5)
            
            # Wait and check if device is discovered
            start_time = time.time()
            while time.time() - start_time < timeout:
                result = subprocess.run(['bluetoothctl', 'devices'], 
                                      capture_output=True, text=True, timeout=5)
                
                if device_address.upper() in result.stdout.upper():
                    self.logger.info(f"Device {device_address} found")
                    subprocess.run(['bluetoothctl', 'scan', 'off'], 
                                  capture_output=True, text=True, timeout=5)
                    return True
                
                time.sleep(2)
            
            # Stop scanning
            subprocess.run(['bluetoothctl', 'scan', 'off'], 
                          capture_output=True, text=True, timeout=5)
            
            self.logger.warning(f"Device {device_address} not found during scan")
            return False
            
        except Exception as e:
            self.logger.error(f"Error during device scan: {e}")
            return False
    
    def pair_device(self, device_address: str) -> bool:
        """Pair with a Bluetooth device"""
        try:
            self.logger.info(f"Attempting to pair with device {device_address}")
            
            # First, ensure device is discoverable
            if not self.scan_for_device(device_address, timeout=20):
                self.logger.error(f"Cannot find device {device_address} for pairing")
                return False
            
            # Trust the device first
            trust_result = subprocess.run(['bluetoothctl', 'trust', device_address], 
                                        capture_output=True, text=True, timeout=15)
            
            if trust_result.returncode != 0:
                self.logger.warning(f"Failed to trust device: {trust_result.stderr}")
            
            # Attempt pairing
            pair_result = subprocess.run(['bluetoothctl', 'pair', device_address], 
                                       capture_output=True, text=True, timeout=30)
            
            if pair_result.returncode == 0:
                self.logger.info(f"Successfully paired with device {device_address}")
                
                # Also connect after successful pairing
                connect_result = subprocess.run(['bluetoothctl', 'connect', device_address], 
                                             capture_output=True, text=True, timeout=15)
                
                if connect_result.returncode == 0:
                    self.logger.info(f"Successfully connected to device {device_address}")
                
                return True
            else:
                self.logger.error(f"Failed to pair with device: {pair_result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error during pairing process: {e}")
            return False
    
    def unpair_device(self, device_address: str) -> bool:
        """Unpair a Bluetooth device"""
        try:
            result = subprocess.run(['bluetoothctl', 'remove', device_address], 
                                  capture_output=True, text=True, timeout=15)
            
            success = result.returncode == 0
            if success:
                self.logger.info(f"Successfully unpaired device {device_address}")
            else:
                self.logger.error(f"Failed to unpair device: {result.stderr}")
                
            return success
            
        except Exception as e:
            self.logger.error(f"Error during unpairing: {e}")
            return False
    
    def get_device_info(self, device_address: str) -> Optional[dict]:
        """Get detailed information about a Bluetooth device"""
        try:
            result = subprocess.run(['bluetoothctl', 'info', device_address], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                return None
            
            info = {}
            for line in result.stdout.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    info[key.strip()] = value.strip()
            
            return info
            
        except Exception as e:
            self.logger.error(f"Error getting device info: {e}")
            return None
    
    def ensure_device_ready(self, device_address: str, max_attempts: int = 3) -> bool:
        """Ensure device is paired and ready for connection"""
        try:
            self.logger.info(f"Ensuring device {device_address} is ready for connection")
            
            # Check Bluetooth service
            if not self.check_bluetooth_service():
                self.logger.error("Bluetooth service is not available")
                return False
            
            # Check if already paired and connected
            if self.is_device_paired(device_address):
                self.logger.info(f"Device {device_address} is already paired")
                
                if self.is_device_connected(device_address):
                    self.logger.info(f"Device {device_address} is already connected")
                    return True
                else:
                    # Try to connect to paired device
                    self.logger.info(f"Attempting to connect to paired device {device_address}")
                    connect_result = subprocess.run(['bluetoothctl', 'connect', device_address], 
                                                  capture_output=True, text=True, timeout=15)
                    
                    if connect_result.returncode == 0:
                        self.logger.info(f"Successfully connected to device {device_address}")
                        return True
                    else:
                        self.logger.warning(f"Failed to connect to paired device: {connect_result.stderr}")
            
            # Device is not paired or connection failed, attempt pairing
            for attempt in range(max_attempts):
                self.logger.info(f"Pairing attempt {attempt + 1}/{max_attempts}")
                
                if self.pair_device(device_address):
                    # Verify pairing was successful
                    if self.is_device_paired(device_address):
                        self.logger.info(f"Device {device_address} successfully paired and ready")
                        return True
                
                if attempt < max_attempts - 1:
                    self.logger.info(f"Pairing attempt failed, retrying in 5 seconds...")
                    time.sleep(5)
            
            self.logger.error(f"Failed to pair device {device_address} after {max_attempts} attempts")
            return False
            
        except Exception as e:
            self.logger.error(f"Error ensuring device readiness: {e}")
            return False