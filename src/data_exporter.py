#Data Exporter Module
#Handles processing and exporting of Shimmer3 IMU data to various formats
#Supports data filtering, analysis, and multiple export formats
#Last revision: 10/13/2025

import asyncio
import csv
import json
import logging
import time
import gzip
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Union, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from datetime import datetime
try:
    from .utils.path_utils import normalize_path, ensure_dir
except ImportError:
    from utils.path_utils import normalize_path, ensure_dir

@dataclass
class ExportConfig:
    """Configuration for data export"""
    format_type: str = 'csv'
    include_filtered: bool = True
    include_statistics: bool = True
    include_plots: bool = False
    filter_frequency: Optional[float] = None
    downsample_factor: Optional[int] = None
    time_range: Optional[Tuple[datetime, datetime]] = None
    sensor_selection: Optional[List[str]] = None
    compression: bool = False

class DataProcessor:
    """Data processing utilities"""
    
    @staticmethod
    def apply_filters(data: pd.DataFrame, filter_config: Dict[str, Any]) -> pd.DataFrame:
        """Apply various filters to the data"""
        filtered_data = data.copy()
        
        try:
            # Low-pass filter
            if filter_config.get('lowpass_freq'):
                fs = filter_config.get('sampling_rate', 51.2)
                nyquist = fs / 2
                cutoff = filter_config['lowpass_freq']
                
                if cutoff < nyquist:
                    # Design Butterworth filter
                    sos = signal.butter(4, cutoff/nyquist, btype='low', output='sos')
                    
                    # Apply to sensor columns
                    sensor_cols = [col for col in data.columns if any(sensor in col 
                                  for sensor in ['accel', 'gyro', 'mag'])]
                    
                    for col in sensor_cols:
                        if col in filtered_data.columns:
                            filtered_data[col] = signal.sosfilt(sos, filtered_data[col])
            
            # High-pass filter  
            if filter_config.get('highpass_freq'):
                fs = filter_config.get('sampling_rate', 51.2)
                nyquist = fs / 2
                cutoff = filter_config['highpass_freq']
                
                if cutoff < nyquist:
                    sos = signal.butter(4, cutoff/nyquist, btype='high', output='sos')
                    
                    sensor_cols = [col for col in data.columns if any(sensor in col 
                                  for sensor in ['accel', 'gyro', 'mag'])]
                    
                    for col in sensor_cols:
                        if col in filtered_data.columns:
                            filtered_data[col] = signal.sosfilt(sos, filtered_data[col])
            
            # Moving average
            if filter_config.get('moving_average_window'):
                window = filter_config['moving_average_window']
                sensor_cols = [col for col in data.columns if any(sensor in col 
                              for sensor in ['accel', 'gyro', 'mag'])]
                
                for col in sensor_cols:
                    if col in filtered_data.columns:
                        filtered_data[col] = filtered_data[col].rolling(
                            window=window, center=True).mean()
            
            return filtered_data
            
        except Exception as e:
            logging.error(f"Error applying filters: {e}")
            return data
    
    @staticmethod
    def calculate_statistics(data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive statistics (optimized)"""
        stats_dict = {}
        try:
            # Use pandas describe for all numeric columns except timestamp/packet_id
            numeric_cols = data.select_dtypes(include=[np.number]).columns.difference(['timestamp', 'packet_id'])
            desc = data[numeric_cols].describe(percentiles=[0.25, 0.5, 0.75]).to_dict()
            for col in numeric_cols:
                col_data = data[col].dropna()
                if len(col_data) > 0:
                    stats_dict[col] = {
                        'count': int(desc[col]['count']),
                        'mean': float(desc[col]['mean']),
                        'std': float(desc[col]['std']),
                        'min': float(desc[col]['min']),
                        'max': float(desc[col]['max']),
                        'median': float(desc[col]['50%']),
                        'q25': float(desc[col]['25%']),
                        'q75': float(desc[col]['75%']),
                        'skewness': float(stats.skew(col_data)),
                        'kurtosis': float(stats.kurtosis(col_data)),
                        'rms': float(np.sqrt(np.mean(np.square(col_data))))
                    }

            # Vectorized magnitude calculation for 3-axis sensors
            for sensor in ['accel', 'gyro', 'mag']:
                cols = [f'{sensor}_x', f'{sensor}_y', f'{sensor}_z']
                if all(c in data.columns for c in cols):
                    arr = data[cols].dropna().values
                    if arr.shape[0] > 0:
                        magnitude = np.linalg.norm(arr, axis=1)
                        stats_dict[f'{sensor}_magnitude'] = {
                            'mean': float(np.mean(magnitude)),
                            'std': float(np.std(magnitude)),
                            'min': float(np.min(magnitude)),
                            'max': float(np.max(magnitude)),
                            'rms': float(np.sqrt(np.mean(np.square(magnitude))))
                        }

            # Data quality metrics (vectorized)
            total_samples = len(data)
            missing = int(data.isnull().sum().sum())
            completeness = float((total_samples * len(data.columns) - missing) / (total_samples * len(data.columns)) * 100) if total_samples > 0 else 0
            sampling_rate_actual = float(len(data) / (data['timestamp'].max() - data['timestamp'].min())) if len(data) > 1 and 'timestamp' in data.columns else 0
            stats_dict['data_quality'] = {
                'total_samples': total_samples,
                'missing_samples': missing,
                'data_completeness': completeness,
                'sampling_rate_actual': sampling_rate_actual
            }
            return stats_dict
        except Exception as e:
            logging.error(f"Error calculating statistics: {e}")
            return {}
    
    @staticmethod
    def perform_fft_analysis(data: pd.DataFrame, sampling_rate: float = 51.2) -> Dict[str, Any]:
        """Perform FFT analysis on sensor data"""
        fft_results = {}
        try:
            sensor_cols = [col for col in data.columns if any(sensor in col 
                          for sensor in ['accel', 'gyro', 'mag'])]
            for col in sensor_cols:
                if col in data.columns:
                    col_data = data[col].dropna()
                    if len(col_data) > 1:
                        fft_values = fft(col_data)
                        freqs = fftfreq(len(col_data), 1 / sampling_rate)
                        positive_freq_idx = freqs > 0
                        freqs_positive = freqs[positive_freq_idx]
                        fft_magnitude = np.abs(np.asarray(fft_values)[positive_freq_idx])
                        dominant_idx = int(np.argmax(fft_magnitude))
                        dominant_freq = float(freqs_positive[dominant_idx])
                        fft_results[col] = {
                            'dominant_frequency': dominant_freq,
                            'dominant_magnitude': float(fft_magnitude[dominant_idx]),
                            'frequency_resolution': float(sampling_rate / len(col_data)),
                            'total_power': float(np.sum(fft_magnitude**2))
                        }
            return fft_results
        except Exception as e:
            logging.error(f"Error in FFT analysis: {e}")
            return {}


class DataExporter:
    """
    Data exporter for Shimmer3 IMU data with processing and analysis capabilities
    """
    
    SUPPORTED_FORMATS = ['csv', 'json', 'hdf5', 'excel', 'matlab', 'parquet']
    
    def __init__(self, input_dir: str = 'data/raw', output_dir: str = 'data/processed'):
        """
        Initialize DataExporter
        
        Args:
            input_dir: Input directory containing raw data files
            output_dir: Output directory for processed data
        """
        # Normalize and ensure directories
        normalized_input = normalize_path(input_dir, 'data/raw', 'export.input_dir')
        normalized_output = normalize_path(output_dir, 'data/processed', 'export.output_dir')
        ensure_dir(normalized_input)
        ensure_dir(normalized_output)
        self.input_dir = Path(normalized_input)
        self.output_dir = Path(normalized_output)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize processor
        self.processor = DataProcessor()
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # Export statistics
        self.export_stats = {
            'files_processed': 0,
            'total_records': 0,
            'last_export': None
        }
        
        self.logger.info(f"DataExporter initialized - Input: {input_dir}, Output: {output_dir}")
        # Prevents processing the same data multiple times
        self.last_processed_files: Set[str] = set()
        self.last_export_time: Optional[datetime] = None

    async def export(self, format_type: str, export_config: Optional[ExportConfig] = None) -> bool:
        """
        Export data in specified format
        
        Args:
            format_type: Export format ('csv', 'json', 'hdf5', etc.)
            export_config: Export configuration options
            
        Returns:
            bool: True if export successful
        """
        if format_type not in self.SUPPORTED_FORMATS:
            self.logger.error(f"Unsupported format: {format_type}")
            return False
        
        if not export_config:
            export_config = ExportConfig(format_type=format_type)
        
        try:
            self.logger.info(f"Starting export to {format_type} format")
            
            # Load and combine data
            combined_data = await self._load_and_combine_data()
            
            if combined_data is None or combined_data.empty:
                self.logger.warning("No data to export")
                return False
            
            # Apply processing
            processed_data = await self._process_data(combined_data, export_config)
            
            # Export based on format
            success = False
            if format_type == 'csv':
                success = await self._export_csv(processed_data, export_config)
            elif format_type == 'json':
                success = await self._export_json(processed_data, export_config)
            elif format_type == 'hdf5':
                success = await self._export_hdf5(processed_data, export_config)
            elif format_type == 'excel':
                success = await self._export_excel(processed_data, export_config)
            elif format_type == 'matlab':
                success = await self._export_matlab(processed_data, export_config)
            elif format_type == 'parquet':
                success = await self._export_parquet(processed_data, export_config)
            
            if success:
                self.export_stats['files_processed'] += 1
                self.export_stats['total_records'] += len(processed_data)
                self.export_stats['last_export'] = datetime.now()
                self.logger.info(f"Export to {format_type} completed successfully")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Export failed: {e}")
            return False
    
    async def _load_and_combine_data(self) -> Optional[pd.DataFrame]:
        """Load and combine data from input files"""
        try:
            data_frames = []
            current_files = set()
            
            # Find all data files
            file_patterns = ['*.csv', '*.json', '*.jsonl', '*.pkl']
            input_files = []
            
            for pattern in file_patterns:
                input_files.extend(self.input_dir.glob(pattern))
                input_files.extend(self.input_dir.glob(pattern + '.gz'))
            
            # Only process new or modified files
            new_files = []
            for file_path in input_files:
                file_id = f"{file_path.name}_{file_path.stat().st_mtime}"
                current_files.add(file_id)
                if file_id not in self.last_processed_files:
                    new_files.append(file_path)
        
            if not new_files:
                self.logger.info("No new data files to process")
                return None
                
            self.logger.info(f"Found {len(new_files)} new/modified data files to process")
            self.last_processed_files = current_files
            
            # Process only new files...
            for file_path in new_files:
                try:
                    # call sync loader (remove await)
                    df = self._load_single_file(file_path)
                    if df is not None and not df.empty:
                        data_frames.append(df)
                except Exception as e:
                    self.logger.warning(f"Failed to load {file_path}: {e}")
            
            if not data_frames:
                self.logger.error("No valid data could be loaded")
                return None
            
            # Combine all data
            combined_df = pd.concat(data_frames, ignore_index=True)
            
            # Sort by timestamp
            if 'timestamp' in combined_df.columns:
                combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
            
            self.logger.info(f"Combined data: {len(combined_df)} records from {len(data_frames)} files")
            return combined_df
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            return None
    
    def _is_recent_or_empty(self, file_path: Path) -> bool:
        try:
            st = file_path.stat()
            if st.st_size == 0:
                self.logger.warning(f"Skipping empty file: {file_path}")
                return True
            # Skip files modified in the last 2 seconds (race with logger)
            if time.time() - st.st_mtime < 2.0:
                self.logger.debug(f"Skipping very recent file: {file_path}")
                return True
        except Exception:
            return False
        return False

    def _load_single_file(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Load a single raw file, tolerating header-only CSVs and skipping empties."""
        try:
            if self._is_recent_or_empty(file_path):
                return None

            is_compressed = file_path.suffix == '.gz'
            suffix = file_path.stem.split('.')[-1] if is_compressed else file_path.suffix[1:]

            if suffix == 'csv':
                # Tolerant read; header-only yields empty DataFrame with columns
                try:
                    df = pd.read_csv(file_path, on_bad_lines='skip')
                except pd.errors.EmptyDataError:
                    self.logger.warning(f"CSV has no columns/rows, skipping: {file_path}")
                    return None
                except Exception as e:
                    self.logger.error(f"Error loading CSV {file_path}: {e}")
                    return None
                # Discard frames without columns
                if df is None or (len(df.columns) == 0):
                    self.logger.warning(f"CSV empty (no columns), skipping: {file_path}")
                    return None
                return df

            elif suffix == 'jsonl' or suffix == 'json':
                try:
                    return pd.read_json(file_path, lines=True)
                except ValueError as e:
                    # Change this from WARNING to DEBUG to reduce log spam
                    self.logger.debug(f"JSON/JSONL parse error, skipping {file_path}: {e}")
                    return None
                except Exception as e:
                    self.logger.error(f"Error loading JSON {file_path}: {e}")
                    return None

            elif suffix == 'pkl':
                rows: List[Any] = []
                opener = gzip.open if is_compressed else open
                mode = 'rb'
                with opener(file_path, mode) as f:
                    while True:
                        try:
                            rows.append(pickle.load(f))
                        except EOFError:
                            break
                        except Exception as e:
                            self.logger.warning(f"Stopping PKL read at {file_path}: {e}")
                            break
                return pd.DataFrame(rows) if rows else None

            else:
                self.logger.warning(f"Unsupported file type for loading: {file_path}")
                return None

        except Exception as e:
            self.logger.error(f"Error loading file {file_path}: {e}")
            return None

    def _infer_sampling_rate(self, df: pd.DataFrame, fallback: float = 51.2) -> float:
        try:
            if 'timestamp' in df.columns and len(df) > 1:
                ts = df['timestamp'].astype(float).to_numpy()
                dt = np.diff(ts)
                dt = dt[dt > 0]
                if len(dt) > 0:
                    return float(1.0 / np.median(dt))
        except Exception:
            pass
        return fallback

    async def _process_data(self, data: pd.DataFrame, config: ExportConfig) -> pd.DataFrame:
        """Process data according to configuration"""
        processed_data = data.copy()
        try:
            # Time range (expects datetime range)
            if config.time_range and 'timestamp' in processed_data.columns:
                start_dt, end_dt = config.time_range
                start_ts = start_dt.timestamp()
                end_ts = end_dt.timestamp()
                processed_data = processed_data[
                    (processed_data['timestamp'] >= start_ts) &
                    (processed_data['timestamp'] <= end_ts)
                ]

            # Sensor selection
            if config.sensor_selection:
                keep = {'timestamp', 'packet_id'}
                for sensor in config.sensor_selection:
                    keep.update([c for c in processed_data.columns if sensor in c])
                cols = [c for c in processed_data.columns if c in keep]
                if cols:
                    processed_data = processed_data[cols]

            # Filtering
            if config.include_filtered and config.filter_frequency:
                fs = self._infer_sampling_rate(processed_data)
                filter_cfg = {
                    'sampling_rate': fs,
                    'lowpass_freq': float(config.filter_frequency)
                }
                processed_data = DataProcessor.apply_filters(processed_data, filter_cfg)

            # Downsample
            if config.downsample_factor and int(config.downsample_factor) > 1:
                processed_data = processed_data.iloc[::int(config.downsample_factor)].reset_index(drop=True)

            return processed_data

        except Exception as e:
            self.logger.error(f"Error processing data: {e}")
            return data
    
    async def _export_csv(self, data: pd.DataFrame, config: ExportConfig) -> bool:
        """Export to CSV format"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"shimmer3_export_{timestamp}.csv"
            
            if config.compression:
                filename += '.gz'
            
            output_path = self.output_dir / filename
            
            if config.compression:
                data.to_csv(output_path, index=False, compression='gzip')
            else:
                data.to_csv(output_path, index=False)
            
            # Export statistics if requested
            if config.include_statistics:
                await self._export_statistics(data, output_path.stem)
            
            # Create plots if requested
            if config.include_plots:
                await self._create_plots(data, output_path.stem)
            
            self.logger.info(f"CSV export completed: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"CSV export failed: {e}")
            return False
    
    async def _export_json(self, data: pd.DataFrame, config: ExportConfig) -> bool:
        """Export to JSON format"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"shimmer3_export_{timestamp}.json"
            
            if config.compression:
                filename += '.gz'
            
            output_path = self.output_dir / filename
            
            # Convert to JSON
            json_data = {
                'metadata': {
                    'export_time': datetime.now().isoformat(),
                    'total_records': len(data),
                    'columns': list(data.columns)
                },
                'data': data.to_dict('records')
            }
            
            if config.compression:
                with gzip.open(output_path, 'wt') as f:
                    json.dump(json_data, f, indent=2)
            else:
                with open(output_path, 'w') as f:
                    json.dump(json_data, f, indent=2)
            
            self.logger.info(f"JSON export completed: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"JSON export failed: {e}")
            return False
    
    async def _export_hdf5(self, data: pd.DataFrame, config: ExportConfig) -> bool:
        """Export to HDF5 format"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"shimmer3_export_{timestamp}.h5"
            output_path = self.output_dir / filename

            str_dtype = h5py.string_dtype(encoding='utf-8')

            with h5py.File(output_path, 'w') as f:
                # Create main datasets per column
                for column in data.columns:
                    series = data[column]
                    if series.dtype == 'object':
                        f.create_dataset(column, data=series.astype(str).values, dtype=str_dtype)
                    else:
                        f.create_dataset(column, data=series.values)

                # Add metadata (serialize complex types)
                f.attrs['export_time'] = datetime.now().isoformat()
                f.attrs['total_records'] = int(len(data))
                f.attrs['columns'] = json.dumps(list(map(str, data.columns)))

            self.logger.info(f"HDF5 export completed: {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"HDF5 export failed: {e}")
            return False
    
    async def _export_excel(self, data: pd.DataFrame, config: ExportConfig) -> bool:
        """Export to Excel format"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"shimmer3_export_{timestamp}.xlsx"
            output_path = self.output_dir / filename
            
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Main data sheet
                data.to_excel(writer, sheet_name='Data', index=False)
                
                # Statistics sheet if requested
                if config.include_statistics:
                    stats = self.processor.calculate_statistics(data)
                    stats_df = pd.DataFrame.from_dict(stats, orient='index')
                    stats_df.to_excel(writer, sheet_name='Statistics')
            
            self.logger.info(f"Excel export completed: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Excel export failed: {e}")
            return False
    
    async def _export_matlab(self, data: pd.DataFrame, config: ExportConfig) -> bool:
        """Export to MATLAB format"""
        try:
            from scipy.io import savemat
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"shimmer3_export_{timestamp}.mat"
            output_path = self.output_dir / filename
            
            # Convert DataFrame to MATLAB-compatible format
            matlab_data = {}
            for column in data.columns:
                # Replace problematic characters in column names
                clean_name = column.replace('-', '_').replace(' ', '_')
                matlab_data[clean_name] = data[column].values
            
            # Add metadata
            matlab_data['metadata'] = {
                'export_time': datetime.now().isoformat(),
                'total_records': len(data),
                'columns': list(data.columns)
            }
            
            savemat(output_path, matlab_data)
            
            self.logger.info(f"MATLAB export completed: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"MATLAB export failed: {e}")
            return False
    
    async def _export_parquet(self, data: pd.DataFrame, config: ExportConfig) -> bool:
        """Export to Parquet format"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"shimmer3_export_{timestamp}.parquet"
            output_path = self.output_dir / filename
            
            # Parquet has built-in compression
            data.to_parquet(output_path, compression='snappy', index=False)
            
            self.logger.info(f"Parquet export completed: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Parquet export failed: {e}")
            return False
    
    async def _export_statistics(self, data: pd.DataFrame, base_filename: str) -> None:
        """Export statistics to separate file"""
        try:
            stats = self.processor.calculate_statistics(data)
            fft_analysis = self.processor.perform_fft_analysis(data)
            
            # Combine statistics
            full_stats = {
                'basic_statistics': stats,
                'fft_analysis': fft_analysis,
                'export_metadata': {
                    'export_time': datetime.now().isoformat(),
                    'total_records': len(data),
                    'data_duration_seconds': float(data['timestamp'].max() - data['timestamp'].min()) if len(data) > 1 else 0
                }
            }
            
            stats_path = self.output_dir / f"{base_filename}_statistics.json"
            
            with open(stats_path, 'w') as f:
                json.dump(full_stats, f, indent=2)
            
            self.logger.info(f"Statistics exported: {stats_path}")
            
        except Exception as e:
            self.logger.warning(f"Could not export statistics: {e}")
    
    async def _create_plots(self, data: pd.DataFrame, base_filename: str) -> None:
        """Create visualization plots"""
        try:
            # Fix the deprecated style
            plt.style.use('seaborn-v0_8-darkgrid')  # Updated style name
            
            # Create subplots for each sensor type
            fig, axes = plt.subplots(3, 1, figsize=(12, 10))
            fig.suptitle('Shimmer3 IMU Data Visualization', fontsize=16)
            
            # Time axis (convert timestamp to datetime for better plotting)
            if 'timestamp' in data.columns:
                time_data = pd.to_datetime(data['timestamp'], unit='s')
            else:
                time_data = range(len(data))
            
            # Plot accelerometer data
            if all(col in data.columns for col in ['accel_x', 'accel_y', 'accel_z']):
                axes[0].plot(time_data, data['accel_x'], label='X', alpha=0.7)
                axes[0].plot(time_data, data['accel_y'], label='Y', alpha=0.7)
                axes[0].plot(time_data, data['accel_z'], label='Z', alpha=0.7)
                axes[0].set_title('Accelerometer Data (m/s²)')
                axes[0].set_ylabel('Acceleration')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
            
            # Plot gyroscope data
            if all(col in data.columns for col in ['gyro_x', 'gyro_y', 'gyro_z']):
                axes[1].plot(time_data, data['gyro_x'], label='X', alpha=0.7)
                axes[1].plot(time_data, data['gyro_y'], label='Y', alpha=0.7)
                axes[1].plot(time_data, data['gyro_z'], label='Z', alpha=0.7)
                axes[1].set_title('Gyroscope Data (°/s)')
                axes[1].set_ylabel('Angular Velocity')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
            
            # Plot magnetometer data
            if all(col in data.columns for col in ['mag_x', 'mag_y', 'mag_z']):
                axes[2].plot(time_data, data['mag_x'], label='X', alpha=0.7)
                axes[2].plot(time_data, data['mag_y'], label='Y', alpha=0.7)
                axes[2].plot(time_data, data['mag_z'], label='Z', alpha=0.7)
                axes[2].set_title('Magnetometer Data (µT)')
                axes[2].set_ylabel('Magnetic Field')
                axes[2].set_xlabel('Time')
                axes[2].legend()
                axes[2].grid(True, alpha=0.3)
            
            # Adjust layout and save
            plt.tight_layout()
            plot_path = self.output_dir / f"{base_filename}_plot.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Plot saved: {plot_path}")
            
        except Exception as e:
            self.logger.warning(f"Could not create plots: {e}")
    
    def get_export_stats(self) -> Dict[str, Any]:
        """Get export statistics"""
        return self.export_stats.copy()
    
    async def export_data(self, data_packet: Dict[str, Any]) -> None:
        """Export single data packet (for real-time export)"""
        try:
            # This method can be used for real-time export
            # Convert single packet to DataFrame and export
            df = pd.DataFrame([data_packet])
            
            # Use simple CSV append for real-time data
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"realtime_export_{timestamp}.csv"
            output_path = self.output_dir / filename
            
            # Append to file or create new one
            if output_path.exists():
                df.to_csv(output_path, mode='a', header=False, index=False)
            else:
                df.to_csv(output_path, index=False)
                
        except Exception as e:
            self.logger.error(f"Real-time export failed: {e}")
    
    def finalize_export(self) -> None:
        """Finalize export process and cleanup"""
        self.logger.info("DataExporter finalized")
    
    def _quarantine_file(self, file_path: Path) -> None:
        """
        Move corrupted files to quarantine folder
        Args:
            file_path: Path to the corrupted file
        Returns: None
        """
        try:
            quarantine_dir = self.input_dir / "quarantine"
            quarantine_dir.mkdir(exist_ok=True)
            
            dest_path = quarantine_dir / f"{file_path.stem}_corrupted_{datetime.now().strftime('%Y%m%d_%H%M%S')}{file_path.suffix}"
            file_path.rename(dest_path)
            self.logger.info(f"Moved corrupted file to quarantine: {dest_path}")
        except Exception as e:
            self.logger.error(f"Failed to quarantine file {file_path}: {e}")