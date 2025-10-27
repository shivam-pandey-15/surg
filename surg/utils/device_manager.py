"""
Device Manager for Automatic CPU/GPU Detection

This module provides utilities for automatically detecting and managing
compute devices (CPU/GPU) for PyTorch models, ensuring optimal performance
across different hardware configurations.

Key Features:
- Automatic GPU detection and configuration
- Fallback to CPU when GPU is unavailable
- Multi-GPU support and device selection
- Device-aware model initialization
- Memory management utilities
"""

import torch
from typing import Optional, Union, List
import logging

logger = logging.getLogger(__name__)


class DeviceManager:
    """
    Manages compute device selection and configuration for PyTorch models.
    
    Automatically detects available GPUs and provides convenient methods
    for device management across the SURG package.
    """
    
    def __init__(self, device: Optional[Union[str, torch.device]] = None):
        """
        Initialize device manager with automatic GPU detection.
        
        Args:
            device: Optional device specification ('cpu', 'cuda', 'cuda:0', etc.)
                   If None, automatically selects best available device
        """
        if device is None:
            self.device = self._get_default_device()
        else:
            self.device = torch.device(device)
            
        logger.info(f"DeviceManager initialized with device: {self.device}")
        self._log_device_info()
    
    @staticmethod
    def _get_default_device() -> torch.device:
        """
        Get the default device based on availability.
        
        Returns:
            torch.device: Best available device (CUDA if available, else CPU)
        """
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    
    def _log_device_info(self) -> None:
        """Log information about the selected device."""
        if self.is_cuda():
            gpu_name = torch.cuda.get_device_name(self.device)
            gpu_memory = torch.cuda.get_device_properties(self.device).total_memory / 1e9
            logger.info(f"Using GPU: {gpu_name}")
            logger.info(f"GPU Memory: {gpu_memory:.2f} GB")
        else:
            logger.info("Using CPU (GPU not available)")
    
    def is_cuda(self) -> bool:
        """
        Check if current device is CUDA GPU.
        
        Returns:
            bool: True if using CUDA GPU, False otherwise
        """
        return self.device.type == 'cuda'
    
    def is_cpu(self) -> bool:
        """
        Check if current device is CPU.
        
        Returns:
            bool: True if using CPU, False otherwise
        """
        return self.device.type == 'cpu'
    
    def get_device(self) -> torch.device:
        """
        Get the current device.
        
        Returns:
            torch.device: Current device
        """
        return self.device
    
    def to_device(self, tensor_or_model: Union[torch.Tensor, torch.nn.Module]) -> Union[torch.Tensor, torch.nn.Module]:
        """
        Move tensor or model to the current device.
        
        Args:
            tensor_or_model: PyTorch tensor or model to move
            
        Returns:
            Tensor or model on the current device
        """
        return tensor_or_model.to(self.device)
    
    @staticmethod
    def get_available_devices() -> List[str]:
        """
        Get list of all available devices.
        
        Returns:
            List of available device names
        """
        devices = ['cpu']
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            devices.extend([f'cuda:{i}' for i in range(num_gpus)])
        return devices
    
    @staticmethod
    def get_gpu_memory_info(device_id: int = 0) -> dict:
        """
        Get GPU memory information.
        
        Args:
            device_id: GPU device ID
            
        Returns:
            Dictionary with memory information (allocated, reserved, total)
        """
        if not torch.cuda.is_available():
            return {
                'allocated': 0,
                'reserved': 0,
                'total': 0,
                'available': False
            }
        
        return {
            'allocated': torch.cuda.memory_allocated(device_id) / 1e9,  # GB
            'reserved': torch.cuda.memory_reserved(device_id) / 1e9,  # GB
            'total': torch.cuda.get_device_properties(device_id).total_memory / 1e9,  # GB
            'available': True,
            'device_name': torch.cuda.get_device_name(device_id)
        }
    
    @staticmethod
    def clear_gpu_memory(device_id: Optional[int] = None) -> None:
        """
        Clear GPU memory cache.
        
        Args:
            device_id: Optional GPU device ID. If None, clears all devices.
        """
        if torch.cuda.is_available():
            if device_id is not None:
                with torch.cuda.device(device_id):
                    torch.cuda.empty_cache()
            else:
                torch.cuda.empty_cache()
            logger.info("GPU memory cache cleared")
    
    def set_device(self, device: Union[str, torch.device]) -> None:
        """
        Change the current device.
        
        Args:
            device: New device specification
        """
        self.device = torch.device(device)
        logger.info(f"Device changed to: {self.device}")
        self._log_device_info()
    
    def __repr__(self) -> str:
        """String representation of DeviceManager."""
        return f"DeviceManager(device={self.device})"


# Global device manager instance
_global_device_manager: Optional[DeviceManager] = None


def get_device_manager() -> DeviceManager:
    """
    Get or create the global device manager instance.
    
    Returns:
        DeviceManager: Global device manager instance
    """
    global _global_device_manager
    if _global_device_manager is None:
        _global_device_manager = DeviceManager()
    return _global_device_manager


def get_device() -> torch.device:
    """
    Get the current device from global device manager.
    
    Returns:
        torch.device: Current device
    """
    return get_device_manager().get_device()


def to_device(tensor_or_model: Union[torch.Tensor, torch.nn.Module]) -> Union[torch.Tensor, torch.nn.Module]:
    """
    Move tensor or model to the current device.
    
    Args:
        tensor_or_model: PyTorch tensor or model to move
        
    Returns:
        Tensor or model on the current device
    """
    return get_device_manager().to_device(tensor_or_model)


def is_gpu_available() -> bool:
    """
    Check if GPU is available.
    
    Returns:
        bool: True if GPU is available, False otherwise
    """
    return torch.cuda.is_available()


def print_device_info() -> None:
    """Print detailed information about available devices."""
    print("=" * 60)
    print("DEVICE INFORMATION")
    print("=" * 60)
    
    # CPU Information
    print(f"\n📍 CPU: Available")
    
    # GPU Information
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"\n🚀 GPU: {num_gpus} device(s) available")
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / 1e9
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"          Memory: {memory_gb:.2f} GB")
            print(f"          Compute Capability: {props.major}.{props.minor}")
    else:
        print(f"\n🚀 GPU: Not available")
    
    # Current Device
    current_device = get_device()
    print(f"\n✅ Current Device: {current_device}")
    print("=" * 60)


# Example usage and testing
if __name__ == "__main__":
    # Print device information
    print_device_info()
    
    # Test device manager
    dm = get_device_manager()
    print(f"\nDevice Manager: {dm}")
    print(f"Using CUDA: {dm.is_cuda()}")
    print(f"Available devices: {dm.get_available_devices()}")
    
    # Test tensor movement
    if torch.cuda.is_available():
        x = torch.randn(3, 3)
        print(f"\nTensor on CPU: {x.device}")
        x = to_device(x)
        print(f"Tensor after to_device(): {x.device}")
        
        # Memory info
        mem_info = DeviceManager.get_gpu_memory_info()
        print(f"\nGPU Memory Info:")
        print(f"  Device: {mem_info['device_name']}")
        print(f"  Allocated: {mem_info['allocated']:.2f} GB")
        print(f"  Total: {mem_info['total']:.2f} GB")
