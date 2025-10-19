"""
Configuration management for OCR project.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import yaml
from loguru import logger


@dataclass
class OCRConfig:
    """Configuration class for OCR settings."""
    
    # Engine settings
    default_engine: str = "tesseract"
    tesseract_language: str = "eng"
    tesseract_config: str = "--psm 6"
    
    # EasyOCR settings
    easyocr_languages: list = None
    easyocr_gpu: bool = False
    
    # PaddleOCR settings
    paddleocr_language: str = "en"
    paddleocr_use_angle_cls: bool = True
    paddleocr_use_gpu: bool = False
    
    # Image processing settings
    image_preprocessing: bool = True
    denoise_kernel_size: int = 3
    adaptive_threshold_block_size: int = 11
    adaptive_threshold_c: int = 2
    
    # Output settings
    save_results: bool = True
    output_format: str = "txt"  # txt, json, csv
    include_confidence: bool = True
    include_bounding_boxes: bool = True
    
    # Logging settings
    log_level: str = "INFO"
    log_file: str = "logs/ocr.log"
    
    def __post_init__(self):
        """Initialize default values after dataclass creation."""
        if self.easyocr_languages is None:
            self.easyocr_languages = ["en"]


class ConfigManager:
    """Manages configuration loading and saving."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or "config/config.yaml"
        self.config = self._load_config()
    
    def _load_config(self) -> OCRConfig:
        """
        Load configuration from file or create default.
        
        Returns:
            OCRConfig object
        """
        config_file = Path(self.config_path)
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config_dict = yaml.safe_load(f)
                logger.info(f"Loaded configuration from {config_file}")
                return OCRConfig(**config_dict)
            except Exception as e:
                logger.error(f"Error loading config: {e}")
                logger.info("Using default configuration")
                return OCRConfig()
        else:
            logger.info("No config file found, using default configuration")
            return OCRConfig()
    
    def save_config(self, config: Optional[OCRConfig] = None) -> None:
        """
        Save configuration to file.
        
        Args:
            config: Configuration to save (uses current config if None)
        """
        if config is None:
            config = self.config
        
        config_file = Path(self.config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(config_file, 'w') as f:
                yaml.dump(asdict(config), f, default_flow_style=False)
            logger.info(f"Configuration saved to {config_file}")
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def update_config(self, **kwargs) -> None:
        """
        Update configuration with new values.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Updated config: {key} = {value}")
            else:
                logger.warning(f"Unknown config parameter: {key}")
    
    def get_config(self) -> OCRConfig:
        """Get current configuration."""
        return self.config
    
    def reset_to_defaults(self) -> None:
        """Reset configuration to defaults."""
        self.config = OCRConfig()
        logger.info("Configuration reset to defaults")


def setup_logging(config: OCRConfig) -> None:
    """
    Setup logging configuration.
    
    Args:
        config: OCR configuration object
    """
    # Remove default logger
    logger.remove()
    
    # Add console logging
    logger.add(
        lambda msg: print(msg, end=""),
        level=config.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    # Add file logging
    log_file = Path(config.log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    logger.add(
        log_file,
        level=config.log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="10 MB",
        retention="7 days",
        compression="zip"
    )
    
    logger.info("Logging configured successfully")


# Global configuration manager instance
config_manager = ConfigManager()
