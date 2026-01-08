"""
Configuration Management for FOLIO Evaluation Framework

Handles loading and validation of configuration files for different evaluation scenarios.
"""

import json
import yaml
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Model configuration"""
    provider: str  # 'ollama', 'openai', 'anthropic'
    model_name: str
    provider_config: Optional[Dict[str, Any]] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 512
    num_ctx: Optional[int] = None


@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    data_path: str
    output_dir: str = "results/evaluation"
    max_examples: Optional[int] = None
    start_idx: int = 0
    end_idx: Optional[int] = None
    include_complexity: bool = True
    prompt_template: str = "json_direct"  # direct, cot, json_direct, json_cot
    use_chat_format: bool = True
    filter_by_complexity: Optional[List[float]] = None  # [min, max]
    filter_by_label: Optional[str] = None


@dataclass
class FullConfig:
    """Complete configuration"""
    model: ModelConfig
    evaluation: EvaluationConfig
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FullConfig':
        """Create from dictionary"""
        model_config = ModelConfig(**data['model'])
        eval_config = EvaluationConfig(**data['evaluation'])
        return cls(model=model_config, evaluation=eval_config)


class ConfigManager:
    """Configuration management system"""
    
    def __init__(self, config_dir: str = "evaluation/configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self._create_default_configs()
    
    def _create_default_configs(self):
        """Create default configuration files"""
        default_configs = {
            "ollama_qwen.yaml": {
                "model": {
                    "provider": "ollama",
                    "model_name": "qwen2.5:32b",
                    "base_url": "http://localhost:11434",
                    "temperature": 0.0,
                    "max_tokens": 512
                },
                "evaluation": {
                    "data_path": "data/folio_v2_validation.jsonl",
                    "output_dir": "results/evaluation",
                    "max_examples": None,
                    "include_complexity": True,
                    "prompt_template": "json_direct",
                    "use_chat_format": True
                }
            },
            "openai_gpt4.yaml": {
                "model": {
                    "provider": "openai",
                    "model_name": "gpt-4",
                    "temperature": 0.0,
                    "max_tokens": 512
                },
                "evaluation": {
                    "data_path": "data/folio_v2_validation.jsonl",
                    "output_dir": "results/evaluation",
                    "max_examples": None,
                    "include_complexity": True,
                    "prompt_template": "json_cot",
                    "use_chat_format": True
                }
            },
            "anthropic_claude.yaml": {
                "model": {
                    "provider": "anthropic",
                    "model_name": "claude-3-sonnet-20240229",
                    "temperature": 0.0,
                    "max_tokens": 512
                },
                "evaluation": {
                    "data_path": "data/folio_v2_validation.jsonl",
                    "output_dir": "results/evaluation",
                    "max_examples": None,
                    "include_complexity": True,
                    "prompt_template": "json_cot",
                    "use_chat_format": True
                }
            }
        }
        
        for filename, config in default_configs.items():
            config_path = self.config_dir / filename
            if not config_path.exists():
                with open(config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, indent=2)
                logger.info(f"Created default config: {config_path}")
    
    def load_config(self, config_path: str) -> FullConfig:
        """Load configuration from file"""
        config_path = Path(config_path)
        
        if not config_path.is_absolute():
            config_path = self.config_dir / config_path
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load based on file extension
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(config_path, 'r') as f:
                data = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'r') as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")
        
        # Validate and create config object
        try:
            config = FullConfig.from_dict(data)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            raise ValueError(f"Invalid configuration format: {e}")
    
    def save_config(self, config: FullConfig, config_path: str):
        """Save configuration to file"""
        config_path = Path(config_path)
        
        if not config_path.is_absolute():
            config_path = self.config_dir / config_path
        
        # Save based on file extension
        data = config.to_dict()
        
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(config_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, indent=2)
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")
        
        logger.info(f"Saved configuration to {config_path}")
    
    def list_configs(self) -> List[str]:
        """List available configuration files"""
        configs = []
        for ext in ['*.yaml', '*.yml', '*.json']:
            configs.extend(self.config_dir.glob(ext))
        return [c.name for c in configs]
    
    def create_config(self, 
                     provider: str,
                     model_name: str,
                     data_path: str,
                     output_name: str,
                     **kwargs) -> FullConfig:
        """Create a new configuration programmatically"""
        
        # Model config
        model_config = ModelConfig(
            provider=provider,
            model_name=model_name,
            **{k: v for k, v in kwargs.items() if k in ModelConfig.__annotations__}
        )
        
        # Evaluation config
        eval_config = EvaluationConfig(
            data_path=data_path,
            **{k: v for k, v in kwargs.items() if k in EvaluationConfig.__annotations__}
        )
        
        config = FullConfig(model=model_config, evaluation=eval_config)
        
        # Save config
        config_filename = f"{output_name}.yaml"
        self.save_config(config, config_filename)
        
        return config
    
    def validate_config(self, config: FullConfig) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Validate model config
        if not config.model.provider:
            issues.append("Model provider is required")
        
        if not config.model.model_name:
            issues.append("Model name is required")
        
        if config.model.provider == 'ollama' and not config.model.base_url:
            issues.append("Ollama requires base_url")
        
        if config.model.provider in ['openai', 'anthropic'] and not config.model.api_key:
            # Check environment variables
            import os
            if config.model.provider == 'openai' and not os.getenv('OPENAI_API_KEY'):
                issues.append("OpenAI requires api_key or OPENAI_API_KEY environment variable")
            elif config.model.provider == 'anthropic' and not os.getenv('ANTHROPIC_API_KEY'):
                issues.append("Anthropic requires api_key or ANTHROPIC_API_KEY environment variable")
        
        # Validate evaluation config
        if not config.evaluation.data_path:
            issues.append("Data path is required")
        
        data_path = Path(config.evaluation.data_path)
        if not data_path.exists():
            issues.append(f"Data file not found: {config.evaluation.data_path}")
        
        valid_templates = ['direct', 'cot', 'json_direct', 'json_cot']
        if config.evaluation.prompt_template not in valid_templates:
            issues.append(f"Invalid prompt template. Must be one of: {valid_templates}")
        
        return issues


def get_preset_configs() -> Dict[str, Dict[str, Any]]:
    """Get preset configurations for common scenarios"""
    return {
        "quick_test": {
            "model": {
                "provider": "ollama",
                "model_name": "qwen2.5:32b",
                "base_url": "http://localhost:11434",
                "temperature": 0.0,
                "max_tokens": 256
            },
            "evaluation": {
                "data_path": "data/folio_v2_validation.jsonl",
                "max_examples": 10,
                "include_complexity": True,
                "prompt_template": "json_direct"
            }
        },
        "full_evaluation": {
            "model": {
                "provider": "ollama",
                "model_name": "qwen2.5:32b",
                "base_url": "http://localhost:11434",
                "temperature": 0.0,
                "max_tokens": 512
            },
            "evaluation": {
                "data_path": "data/folio_v2_validation.jsonl",
                "include_complexity": True,
                "prompt_template": "json_cot"
            }
        },
        "complexity_analysis": {
            "model": {
                "provider": "ollama", 
                "model_name": "qwen2.5:32b",
                "base_url": "http://localhost:11434",
                "temperature": 0.0,
                "max_tokens": 512
            },
            "evaluation": {
                "data_path": "data/folio_v2_validation.jsonl",
                "include_complexity": True,
                "prompt_template": "json_cot",
                "filter_by_complexity": [0, 30]  # Low complexity only
            }
        }
    }


if __name__ == "__main__":
    # Test configuration management
    config_manager = ConfigManager()
    
    # List available configs
    print("Available configurations:")
    for config in config_manager.list_configs():
        print(f"  - {config}")
    
    # Load and validate a config
    if config_manager.list_configs():
        config_name = config_manager.list_configs()[0]
        print(f"\nLoading config: {config_name}")
        
        config = config_manager.load_config(config_name)
        print(f"Model: {config.model.provider} - {config.model.model_name}")
        print(f"Data: {config.evaluation.data_path}")
        
        # Validate
        issues = config_manager.validate_config(config)
        if issues:
            print("Validation issues:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("Configuration is valid!")
    
    # Create a new config
    print("\nCreating new configuration...")
    new_config = config_manager.create_config(
        provider="ollama",
        model_name="qwen2.5:32b",
        data_path="data/folio_v2_validation.jsonl",
        output_name="test_config",
        base_url="http://localhost:11434",
        max_examples=5
    )
    print("New configuration created and saved.")
