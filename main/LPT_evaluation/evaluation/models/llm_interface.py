"""
LLM Interface System for FOLIO Evaluation Framework

Supports multiple model providers:
- Local Ollama models (qwen2.5:32b, etc.)
- OpenAI API (with custom base_url support)
- Anthropic Claude
- Other local/remote models

Based on ProverGen's interface but enhanced for FOLIO evaluation.
"""

import os
import abc
import time
import json
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import requests
from openai import OpenAI
import anthropic
import torch  # 新增

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMResponse:
    """Standardized response from any LLM provider"""
    prompt_text: Union[str, List[Dict[str, str]]]
    response_text: str
    prompt_info: Dict[str, Any]
    model_name: str
    provider: str
    error: Optional[str] = None
    

class BaseLLMProvider(abc.ABC):
    """Base class for all LLM providers"""

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.provider_name = self.__class__.__name__
        
    @abc.abstractmethod
    def completion(self, prompt: Union[str, List[Dict[str, str]]], **kwargs) -> LLMResponse:
        """Generate completion for given prompt"""
        raise NotImplementedError("Override me!")
    
    def _create_response(self, prompt, response_text, **kwargs) -> LLMResponse:
        """Helper to create standardized response"""
        return LLMResponse(
            prompt_text=prompt,
            response_text=response_text,
            prompt_info=kwargs,
            model_name=self.model_name,
            provider=self.provider_name
        )


class OllamaProvider(BaseLLMProvider):
    """Ollama local model provider (qwen2.5:32b, etc.)"""
    
    def __init__(self, model_name: str, base_url: str = "http://localhost:11434", **kwargs):
        super().__init__(model_name, **kwargs)
        self.base_url = base_url.rstrip('/')
        self.temperature = kwargs.get('temperature', 0.0)
        self.max_tokens = kwargs.get('max_tokens', 2048)
        self.num_ctx = kwargs.get('num_ctx', None)  # allow overriding context window
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self):
        """Test if Ollama server is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                logger.info(f"Ollama connected. Available models: {model_names}")
                
                # Check if our model is available
                if self.model_name not in model_names:
                    logger.warning(f"Model {self.model_name} not found in Ollama. Available: {model_names}")
            else:
                logger.error(f"Ollama server responded with status {response.status_code}")
        except Exception as e:
            logger.error(f"Failed to connect to Ollama server: {e}")
            raise ConnectionError(f"Cannot connect to Ollama at {self.base_url}")
    
    def completion(self, prompt: Union[str, List[Dict[str, str]]], **kwargs) -> LLMResponse:
        """Generate completion using Ollama API"""
        try:
            # Convert prompt format if needed
            if isinstance(prompt, list):
                # Convert OpenAI format to Ollama format
                messages = prompt
                formatted_prompt = self._format_messages(messages)
            else:
                formatted_prompt = prompt
            
            # Prepare request
            request_data = {
                "model": self.model_name,
                "prompt": formatted_prompt,
                "stream": False,
                "options": {
                    "temperature": kwargs.get('temperature', self.temperature),
                    "num_predict": kwargs.get('max_tokens', self.max_tokens),
                }
            }
            if self.num_ctx is not None:
                request_data["options"]["num_ctx"] = self.num_ctx
            
            # Make request
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("[OLLAMA][REQUEST] model=%s len(prompt)=%d temperature=%s num_predict=%s", self.model_name, len(formatted_prompt), request_data['options']['temperature'], request_data['options']['num_predict'])
                logger.debug("[OLLAMA][PROMPT PREVIEW]\n%s", formatted_prompt[:500])
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=request_data,
                timeout=kwargs.get('timeout', 300)
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get('response', '')
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("[OLLAMA][RAW RESPONSE META] keys=%s", list(result.keys()))
                    logger.debug("[OLLAMA][RAW RESPONSE TEXT LEN]=%d", len(response_text))
                    logger.debug("[OLLAMA][RAW RESPONSE PREVIEW]\n%s", response_text[:500])
                
                return self._create_response(
                    prompt=prompt,
                    response_text=response_text,
                    temperature=request_data["options"]["temperature"],
                    max_tokens=request_data["options"]["num_predict"],
                    ollama_stats=result.get('stats', {})
                )
            else:
                error_msg = f"Ollama API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return self._create_response(
                    prompt=prompt,
                    response_text="",
                    error=error_msg
                )
                
        except Exception as e:
            error_msg = f"Ollama completion failed: {str(e)}"
            logger.error(error_msg)
            return self._create_response(
                prompt=prompt,
                response_text="",
                error=error_msg
            )
    
    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """Convert OpenAI message format to single prompt string"""
        formatted = ""
        for msg in messages:
            role = msg.get('role', '')
            content = msg.get('content', '')
            
            if role == 'system':
                formatted += f"System: {content}\n\n"
            elif role == 'user':
                formatted += f"User: {content}\n\n"
            elif role == 'assistant':
                formatted += f"Assistant: {content}\n\n"
        return formatted


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider with custom base_url support"""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs):
        super().__init__(model_name, **kwargs)
        
        # Support custom base_url
        self.base_url = base_url or os.getenv('OPENAI_BASE_URL') or "https://api.openai.com/v1"
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        # Initialize OpenAI client with custom base_url
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        self.temperature = kwargs.get('temperature', 0.0)
        self.max_tokens = kwargs.get('max_tokens', 1024)
        self.retry_count = 0
        self.max_retries = 3
        
        logger.info(f"OpenAI provider initialized with base_url: {self.base_url}")
    
    def completion(self, prompt: Union[str, List[Dict[str, str]]], **kwargs) -> LLMResponse:
        """Generate completion using OpenAI API with custom base_url"""
        # Convert string prompt to message format if needed
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt

        # Preprocess multimodal content messages
        processed_messages = []
        for msg in messages:
            content = msg.get('content')
            # Handle list of multimodal content items
            if isinstance(content, list):
                for item in content:
                    base_msg = {'role': msg.get('role')}
                    if item.get('type') == 'text':
                        base_msg['content'] = item.get('text')
                    elif item.get('type') == 'image_url' and item.get('image_url'):
                        base_msg['type'] = 'image_url'
                        base_msg['image_url'] = item.get('image_url')
                    processed_messages.append(base_msg)
            else:
                processed_messages.append(msg)
        messages = processed_messages

        stream = kwargs.get('stream', False)
        for attempt in range(self.max_retries):
            try:
                # 针对 dashscope Qwen3 系列模型自动加 enable_thinking=True（流式）或 False（非流式）
                extra_body = kwargs.get('extra_body')
                if extra_body is None and self.base_url.startswith('https://dashscope.aliyuncs.com'):
                    if self.model_name.lower().startswith('qwen') or 'qwen' in self.model_name.lower():
                        # 为 Qwen3 系列启用思维输出，便于拼接 <think>
                        extra_body = {"enable_thinking": True}
                if stream:
                    # 流式输出，收集所有内容
                    reasoning_chunks = []
                    answer_chunks = []
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=kwargs.get('temperature', self.temperature),
                        max_tokens=kwargs.get('max_tokens', self.max_tokens),
                        extra_body=extra_body,
                        stream=True
                    )
                    for chunk in response:
                        msg = chunk.choices[0].delta
                        if hasattr(msg, 'reasoning_content') and msg.reasoning_content:
                            reasoning_chunks.append(msg.reasoning_content)
                        if hasattr(msg, 'content') and msg.content:
                            answer_chunks.append(msg.content)
                    final_text = ''.join(answer_chunks)
                    reasoning_content = ''.join(reasoning_chunks) if reasoning_chunks else None
                    # 合并思维链与答案，确保结果中包含 <think> ... </think>
                    if reasoning_content:
                        response_text = f"<think>\n{reasoning_content}\n</think>\n{final_text}"
                    else:
                        response_text = final_text
                    return self._create_response(
                        prompt=prompt,
                        response_text=response_text,
                        temperature=kwargs.get('temperature', self.temperature),
                        max_tokens=kwargs.get('max_tokens', self.max_tokens),
                        usage={},
                        base_url=self.base_url,
                        reasoning_content=reasoning_content
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=kwargs.get('temperature', self.temperature),
                        max_tokens=kwargs.get('max_tokens', self.max_tokens),
                        extra_body=extra_body
                    )
                    msg = response.choices[0].message
                    final_text = msg.content
                    reasoning_content = getattr(msg, 'reasoning_content', None)
                    if reasoning_content:
                        response_text = f"<think>\n{reasoning_content}\n</think>\n{final_text}"
                    else:
                        response_text = final_text
                    return self._create_response(
                        prompt=prompt,
                        response_text=response_text,
                        temperature=kwargs.get('temperature', self.temperature),
                        max_tokens=kwargs.get('max_tokens', self.max_tokens),
                        usage=response.usage.dict() if response.usage else {},
                        base_url=self.base_url,
                        reasoning_content=reasoning_content
                    )
                
            except Exception as e:
                self.retry_count += 1
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"OpenAI API error (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    error_msg = f"OpenAI API failed after {self.max_retries} attempts: {str(e)}"
                    logger.error(error_msg)
                    return self._create_response(
                        prompt=prompt,
                        response_text="",
                        error=error_msg
                    )


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider"""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None, **kwargs):
        super().__init__(model_name, **kwargs)
        self.client = anthropic.Anthropic(api_key=api_key or os.getenv('ANTHROPIC_API_KEY'))
        self.temperature = kwargs.get('temperature', 0.0)
        self.max_tokens = kwargs.get('max_tokens', 1024)
        self.retry_count = 0
        self.max_retries = 3
    
    def completion(self, prompt: Union[str, List[Dict[str, str]]], **kwargs) -> LLMResponse:
        """Generate completion using Anthropic API"""
        # Convert to Anthropic format
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
            system_prompt = ""
        else:
            messages = []
            system_prompt = ""
            
            for msg in prompt:
                if msg.get('role') == 'system':
                    system_prompt = msg.get('content', '')
                else:
                    messages.append({
                        "role": msg.get('role', 'user'),
                        "content": msg.get('content', '')
                    })
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=kwargs.get('max_tokens', self.max_tokens),
                    temperature=kwargs.get('temperature', self.temperature),
                    system=system_prompt,
                    messages=messages
                )
                
                response_text = response.content[0].text
                
                return self._create_response(
                    prompt=prompt,
                    response_text=response_text,
                    temperature=kwargs.get('temperature', self.temperature),
                    max_tokens=kwargs.get('max_tokens', self.max_tokens),
                    usage=response.usage.dict() if hasattr(response, 'usage') else {}
                )
                
            except Exception as e:
                self.retry_count += 1
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"Anthropic API error (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    error_msg = f"Anthropic API failed after {self.max_retries} attempts: {str(e)}"
                    logger.error(error_msg)
                    return self._create_response(
                        prompt=prompt,
                        response_text="",
                        error=error_msg
                    )


class LocalTransformersLoraProvider(BaseLLMProvider):
    """Local Transformers + LoRA provider for any checkpoints (Qwen, Llama, etc.)"""

    def __init__(
        self,
        model_name: str,
        checkpoint_path: str,
        base_model: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        load_in_4bit: bool = True,
        **kwargs,
    ):
        super().__init__(model_name, **kwargs)
        self.checkpoint_path = checkpoint_path
        self.base_model = base_model
        self.max_tokens = max_tokens
        self.temperature = temperature
        quant_config = None
        if load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            trust_remote_code=True,
            device_map="auto",
            quantization_config=quant_config,
            local_files_only=True,
        )
        self.model = PeftModel.from_pretrained(self.model, self.checkpoint_path, device_map="auto")
        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def completion(self, prompt: Union[str, List[Dict[str, str]]], **kwargs) -> LLMResponse:
        text_prompt = self._format_prompt(prompt)
        max_new_tokens = kwargs.get("max_tokens", self.max_tokens)
        temperature = kwargs.get("temperature", self.temperature)
        do_sample = temperature > 0
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature if do_sample else None,
        }
        generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}
        with torch.inference_mode():
            inputs = self.tokenizer(text_prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(**inputs, **generation_kwargs)
        response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self._create_response(
            prompt=prompt,
            response_text=response_text,
            temperature=temperature,
            max_tokens=max_new_tokens,
        )

    def _format_prompt(self, prompt: Union[str, List[Dict[str, str]]]) -> str:
        if isinstance(prompt, str):
            return prompt
        parts = []
        for msg in prompt:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"{role.capitalize()}: {content}")
        parts.append("Assistant:")
        return "\n".join(parts)


class LLMInterface:
    """Main interface for all LLM providers"""
    
    def __init__(self, provider: str, model_name: str, **kwargs):
        self.provider_name = provider
        self.model_name = model_name
        
        # Initialize the appropriate provider
        if provider.lower() == 'ollama':
            self.provider = OllamaProvider(model_name, **kwargs)
        elif provider.lower() in ['openai', 'gpt']:
            self.provider = OpenAIProvider(model_name, **kwargs)
        elif provider.lower() in ['anthropic', 'claude']:
            self.provider = AnthropicProvider(model_name, **kwargs)
        elif provider.lower() == 'local_transformers_lora':
            self.provider = LocalTransformersLoraProvider(model_name, **kwargs)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        logger.info(f"Initialized {provider} provider with model {model_name}")
    
    def completion(self, prompt: Union[str, List[Dict[str, str]]], **kwargs) -> LLMResponse:
        """Generate completion using the configured provider"""
        return self.provider.completion(prompt, **kwargs)
    
    @classmethod
    def create_from_config(cls, config: Dict[str, Any]) -> 'LLMInterface':
        """Create LLM interface from configuration dictionary"""
        provider = config.get('provider', 'ollama')
        model_name = config.get('model_name', 'qwen2.5:32b')
        
        # Extract provider-specific config
        provider_config = config.get('provider_config', {})
        
        return cls(provider, model_name, **provider_config)


# Convenience functions for common models
def create_ollama_qwen(base_url: str = "http://localhost:11434", **kwargs) -> LLMInterface:
    """Create Qwen2.5:32b via Ollama"""
    return LLMInterface('ollama', 'qwen2.5:32b', base_url=base_url, **kwargs)

def create_openai_gpt4(api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs) -> LLMInterface:
    """Create GPT-4 via OpenAI API with custom base_url"""
    return LLMInterface('openai', 'gpt-4', api_key=api_key, base_url=base_url, **kwargs)

def create_openai_gpt4o(api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs) -> LLMInterface:
    """Create GPT-4o via OpenAI API with custom base_url"""
    return LLMInterface('openai', 'gpt-4o', api_key=api_key, base_url=base_url, **kwargs)

def create_claude(api_key: Optional[str] = None, **kwargs) -> LLMInterface:
    """Create Claude via Anthropic API"""
    return LLMInterface('anthropic', 'claude-3-sonnet-20240229', api_key=api_key, **kwargs)

def create_local_transformers_lora(checkpoint_path: str, base_model: str, **kwargs) -> LLMInterface:
    """Create local Transformers LoRA provider (supports Qwen, Llama, etc.)"""
    return LLMInterface(
        'local_transformers_lora',
        model_name=kwargs.get('model_name', base_model),
        checkpoint_path=checkpoint_path,
        base_model=base_model,
        **kwargs,
    )


if __name__ == "__main__":
    # Test the interfaces
    print("Testing LLM interfaces...")
    
    # Test OpenAI with custom base_url
    try:
        openai = create_openai_gpt4o(
            api_key="your-api-key",
            base_url="https://openapi.monica.im/v1"
        )
        response = openai.completion("Hello, how are you?")
        print(f"OpenAI Response: {response.response_text[:100]}...")
    except Exception as e:
        print(f"OpenAI test failed: {e}")
    
    # Test Ollama (if available)
    try:
        ollama = create_ollama_qwen()
        response = ollama.completion("Hello, how are you?")
        print(f"Ollama Response: {response.response_text[:100]}...")
    except Exception as e:
        print(f"Ollama test failed: {e}")
    
    print("LLM interface test completed.")
