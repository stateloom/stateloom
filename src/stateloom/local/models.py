"""Curated catalog of recommended Ollama models by hardware tier."""

from __future__ import annotations

MODEL_CATALOG: list[dict] = [
    # Ultra-light (<2GB)
    {
        "model": "qwen2.5:0.5b",
        "size_gb": 0.4,
        "description": "Tiny but capable, good for simple classification/extraction",
        "tier": "ultra-light",
        "parameters": "0.5B",
    },
    {
        "model": "llama3.2:1b",
        "size_gb": 1.3,
        "description": "Smallest Llama, fast for basic tasks",
        "tier": "ultra-light",
        "parameters": "1B",
    },
    # Light (2-4GB)
    {
        "model": "gemma2:2b",
        "size_gb": 1.6,
        "description": "Google's compact model, strong for its size",
        "tier": "light",
        "parameters": "2B",
    },
    {
        "model": "llama3.2:3b",
        "size_gb": 2.0,
        "description": "Good balance of speed and quality",
        "tier": "light",
        "parameters": "3B",
    },
    {
        "model": "phi3:3.8b",
        "size_gb": 2.3,
        "description": "Microsoft's small model, strong reasoning",
        "tier": "light",
        "parameters": "3.8B",
    },
    # Medium (4-8GB)
    {
        "model": "llama3.1:8b",
        "size_gb": 4.7,
        "description": "Excellent general-purpose model",
        "tier": "medium",
        "parameters": "8B",
    },
    {
        "model": "mistral:7b",
        "size_gb": 4.1,
        "description": "Strong all-rounder, good code and reasoning",
        "tier": "medium",
        "parameters": "7B",
    },
    {
        "model": "qwen2.5:7b",
        "size_gb": 4.7,
        "description": "Alibaba's model, strong multilingual support",
        "tier": "medium",
        "parameters": "7B",
    },
    {
        "model": "deepseek-coder-v2:16b",
        "size_gb": 8.9,
        "description": "Specialized for code generation and understanding",
        "tier": "medium",
        "parameters": "16B",
    },
    {
        "model": "phi3.5:3.8b",
        "size_gb": 2.2,
        "description": "Microsoft Phi-3.5, improved reasoning over Phi-3",
        "tier": "medium",
        "parameters": "3.8B",
    },
    {
        "model": "phi4:14b",
        "size_gb": 9.1,
        "description": "Microsoft Phi-4, strong reasoning and code",
        "tier": "medium",
        "parameters": "14B",
    },
    {
        "model": "gemma2:9b",
        "size_gb": 5.4,
        "description": "Google Gemma 2 9B, strong general-purpose model",
        "tier": "medium",
        "parameters": "9B",
    },
    {
        "model": "mistral-nemo:12b",
        "size_gb": 7.1,
        "description": "Mistral Nemo, strong multilingual and code",
        "tier": "medium",
        "parameters": "12B",
    },
    {
        "model": "mistral-small:22b",
        "size_gb": 13.0,
        "description": "Mistral Small, enterprise-grade local model",
        "tier": "medium",
        "parameters": "22B",
    },
    {
        "model": "deepseek-r1:8b",
        "size_gb": 4.9,
        "description": "DeepSeek R1 distilled, strong chain-of-thought reasoning",
        "tier": "medium",
        "parameters": "8B",
    },
    {
        "model": "qwen2.5-coder:7b",
        "size_gb": 4.7,
        "description": "Alibaba's code-specialized model",
        "tier": "medium",
        "parameters": "7B",
    },
    # Heavy (16GB+)
    {
        "model": "llama3.3:70b",
        "size_gb": 40.0,
        "description": "Near cloud-quality, requires significant RAM",
        "tier": "heavy",
        "parameters": "70B",
    },
    {
        "model": "qwen2.5:72b",
        "size_gb": 41.0,
        "description": "Top-tier local model, competes with cloud models",
        "tier": "heavy",
        "parameters": "72B",
    },
    {
        "model": "gemma2:27b",
        "size_gb": 16.0,
        "description": "Google Gemma 2 27B, strong performance for its size",
        "tier": "heavy",
        "parameters": "27B",
    },
    {
        "model": "llama3.2-vision:11b",
        "size_gb": 7.9,
        "description": "Llama 3.2 with vision capabilities",
        "tier": "heavy",
        "parameters": "11B",
    },
    {
        "model": "deepseek-r1:70b",
        "size_gb": 42.0,
        "description": "DeepSeek R1 70B, top-tier reasoning model",
        "tier": "heavy",
        "parameters": "70B",
    },
]
