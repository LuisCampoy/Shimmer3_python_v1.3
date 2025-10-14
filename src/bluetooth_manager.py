# Bluetooth manager
# Created 10/13/2025
# Last update 10/13/2025

import subprocess
import re
import logging
import time
import os
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)

class BluetoothManager:
    """Manages Bluetooth operations including pairing and RFCOMM connection for Shimmer3"""
    
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
            result = subprocess.run(['bluetoothctl', 'devices', 'Paired'], 
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
        """Scan for a specific Bluetooth device using hcitool"""
        try:
            self.logger.info(f"Scanning for device {device_address}...")
            
            # Use hcitool scan to find devices
            result = subprocess.run(['hcitool', 'scan'], 
                                  capture_output=True, text=True, timeout=timeout)
            
            if result.returncode == 0:
                if device_address.upper() in result.stdout.upper():
                    self.logger.info(f"Device {device_address} found during scan")
                    return True
            
            self.logger.warning(f"Device {device_address} not found during scan")
            return False
            
        except Exception as e:
            self.logger.error(f"Error during device scan: {e}")
            return False
    
    def pair_device_legacy(self, device_address: str) -> bool:
        """Pair with Bluetooth device using legacy method for older devices like Shimmer3"""
        try:
            self.logger.info(f"Attempting to pair with Shimmer3 device {device_address}")
            
            # First scan for the device
            if not self.scan_for_device(device_address, timeout=20):
                self.logger.warning(f"Device {device_address} not found during scan, proceeding anyway...")
            
            # Use bluetoothctl for pairing (more reliable than hcitool for pairing)
            pair_commands = [
                f"agent on",
                f"default-agent", 
                f"scan on",
                f"pair {device_address}",
                f"trust {device_address}",
                f"scan off"
            ]
            
            for cmd in pair_commands:
                self.logger.debug(f"Executing bluetoothctl command: {cmd}")
                result = subprocess.run(['bluetoothctl'], 
                                      input=cmd + '\n', 
                                      capture_output=True, 
                                      text=True, 
                                      timeout=15)
                
                if "Failed" in result.stdout or result.returncode != 0:
                    self.logger.warning(f"Command '{cmd}' had issues: {result.stdout}")
                
                time.sleep(1)  # Small delay between commands
            
            # Verify pairing
            if self.is_device_paired(device_address):
                self.logger.info(f"Successfully paired with device {device_address}")
                return True
            else:
                self.logger.error(f"Pairing verification failed for device {device_address}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error during pairing process: {e}")
            return False
    
    def create_rfcomm_binding(self, device_address: str, channel: int = 1, rfcomm_port: int = 0, allow_rebind: bool = True) -> Optional[str]:
        """Create RFCOMM binding for serial communication.
        If the rfcomm device already exists and is bound to the requested device, it will be reused without release.
        If allow_rebind is False and the device is bound to something else, we return None instead of forcibly releasing.
        """
        try:
            rfcomm_device = f"/dev/rfcomm{rfcomm_port}"
            
            # Check if RFCOMM device already exists and is bound
            if os.path.exists(rfcomm_device):
                self.logger.info(f"RFCOMM device {rfcomm_device} already exists")
                result = subprocess.run(['rfcomm', 'show', str(rfcomm_port)],
                                        capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    stdout_upper = result.stdout.upper()
                    if device_address.upper() in stdout_upper:
                        self.logger.info(f"RFCOMM device {rfcomm_device} already bound to target {device_address}; reusing (state: {result.stdout.strip()})")
                        return rfcomm_device
                    else:
                        if not allow_rebind:
                            self.logger.warning(f"RFCOMM device {rfcomm_device} bound to different device and allow_rebind=False; not modifying.")
                            return None
                        self.logger.info(f"Releasing existing RFCOMM binding on {rfcomm_device} (different device)")
                        subprocess.run(['sudo', 'rfcomm', 'release', str(rfcomm_port)],
                                       capture_output=True, text=True, timeout=10)
            
            # Create new RFCOMM binding
            self.logger.info(f"Creating RFCOMM binding: {rfcomm_device} -> {device_address}:{channel}")
            result = subprocess.run(['sudo', 'rfcomm', 'bind', str(rfcomm_port), device_address, str(channel)], 
                                  capture_output=True, text=True, timeout=15)
            
            if result.returncode == 0:
                self.logger.info(f"Successfully created RFCOMM binding {rfcomm_device}")
                
                # Wait for device to be ready
                time.sleep(2)
                
                # Verify device exists
                if os.path.exists(rfcomm_device):
                    return rfcomm_device
                else:
                    self.logger.error(f"RFCOMM device {rfcomm_device} was not created")
                    return None
            else:
                self.logger.error(f"Failed to create RFCOMM binding: {result.stderr}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error creating RFCOMM binding: {e}")
            return None
    
    def release_rfcomm_binding(self, rfcomm_port: int = 0) -> bool:
        """Release RFCOMM binding"""
        try:
            result = subprocess.run(['sudo', 'rfcomm', 'release', str(rfcomm_port)], 
                                  capture_output=True, text=True, timeout=10)
            
            success = result.returncode == 0
            if success:
                self.logger.info(f"Successfully released RFCOMM binding /dev/rfcomm{rfcomm_port}")
            else:
                self.logger.warning(f"Failed to release RFCOMM binding: {result.stderr}")
                
            return success
            
        except Exception as e:
            self.logger.error(f"Error releasing RFCOMM binding: {e}")
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
                if ':' in line and 'Device' not in line:
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = parts[1].strip()
                        info[key] = value
            
            return info
            
        except Exception as e:
            self.logger.error(f"Error getting device info: {e}")
            return None
    
    def prepare_shimmer3_connection(self, device_address: str, max_attempts: int = 3) -> Optional[str]:
        """Prepare Shimmer3 device for RFCOMM connection"""
        try:
            self.logger.info(f"Preparing Shimmer3 device {device_address} for RFCOMM connection")
            
            # Check Bluetooth service
            if not self.check_bluetooth_service():
                self.logger.error("Bluetooth service is not available")
                return None
            
            # Check if device is paired
            if not self.is_device_paired(device_address):
                self.logger.warning(f"Device {device_address} is not paired. Please pair manually before running this application.")
                return None
            else:
                self.logger.info(f"Device {device_address} is already paired")
            
            # Create RFCOMM binding
            rfcomm_device = self.create_rfcomm_binding(device_address, channel=1, rfcomm_port=0)
            
            if rfcomm_device:
                self.logger.info(f"Shimmer3 device ready for connection at {rfcomm_device}")
                return rfcomm_device
            else:
                self.logger.error("Failed to create RFCOMM binding for Shimmer3 device")
                return None
                
        except Exception as e:
            self.logger.error(f"Error preparing Shimmer3 connection: {e}")
            return None
    
    def cleanup_connections(self):
        """Cleanup all RFCOMM connections"""
        try:
            self.logger.info("Cleaning up RFCOMM connections...")
            # Skip cleanup to avoid sudo prompts during normal shutdown
            self.logger.info("Skipping RFCOMM cleanup to avoid sudo prompts")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")