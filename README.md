SLM-Pipeline: Local Fine-Tuning Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Model: Qwen 2.5](https://img.shields.io/badge/Base_Model-Qwen_2.5_1.5B-violet)](https://huggingface.co/Qwen)
[![Technique: QLoRA](https://img.shields.io/badge/Technique-QLoRA_4bit-blue)](https://arxiv.org/abs/2305.14314)
[![Accelerated By: Unsloth](https://img.shields.io/badge/Library-Unsloth_AI-green)](https://github.com/unslothai/unsloth)

A modular, end-to-end pipeline for fine-tuning **Small Language Models (SLMs)** on consumer hardware. This framework is designed to build "Vertical AI" agents—highly specialized models that run locally without cloud dependencies.

## 🎯 The Architecture
This project implements a **Local-First AI** strategy, optimizing for privacy, low latency, and zero inference costs. It moves beyond generic chatbots by "baking" domain-specific knowledge directly into the model weights using Low-Rank Adaptation (LoRA).

**Why this approach?**
* **No RAG Overhead:** For static domains (e.g., hospitality, device manuals), fine-tuning removes the need for complex Vector Databases.
* **Privacy:** All data generation, training, and inference happen within your control.
* **Efficiency:** Runs on standard gaming laptops (RTX 3060/4090/5000 Ada) or edge devices.

## 🚀 Capabilities
* **Synthetic Data Engine:** A Python-based generator that converts raw "Fact Sheets" into high-quality JSONL instruction datasets using LLM APIs.
* **Efficient Training:** Implements QLoRA (4-bit quantization) to fine-tune 1.5B - 7B parameter models in minutes, not hours.
* **Persona Injection:** System-prompt engineering to enforce brand voice (e.g., "Professional Concierge" or "Technical Support Engineer").
* **Local Inference:** A lightweight CLI/API for testing the specialized model immediately after training.

## 🛠️ Tech Stack
* **Base Model:** [Qwen 2.5-1.5B-Instruct](https://huggingface.co/Qwen) (Chosen for its reasoning-to-size ratio).
* **Orchestration:** [Unsloth](https://github.com/unslothai/unsloth) for optimized backward passes.
* **Hardware Support:** NVIDIA GPUs (6GB+ VRAM recommended).
* **Environment:** WSL2 / Linux.