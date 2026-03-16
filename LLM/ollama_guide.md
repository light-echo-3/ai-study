# Ollama：本地运行大语言模型的完全指南

Ollama 是目前最流行的本地大模型运行工具，它把 LLM 的部署体验简化到了类似 Docker 的水平——一行命令即可在自己电脑上跑起一个大模型，无需云服务、无需 API Key、数据完全不出本地。

> GitHub: https://github.com/ollama/ollama | Stars: 380k+

---

## 目录

1. [Ollama 是什么？解决了什么问题？](#1-ollama-是什么解决了什么问题)
2. [架构与技术原理](#2-架构与技术原理)
3. [安装部署](#3-安装部署)
4. [基础使用](#4-基础使用)
5. [支持的模型](#5-支持的模型)
6. [REST API 详解](#6-rest-api-详解)
7. [自定义模型 (Modelfile)](#7-自定义模型-modelfile)
8. [硬件要求](#8-硬件要求)
9. [生态集成](#9-生态集成)

---

## 1. Ollama 是什么？解决了什么问题？

### 核心定位
Ollama 是一个**本地 LLM 运行平台**，让开发者和普通用户可以在自己的电脑上运行开源大语言模型。

### 它解决的三个痛点

| 痛点 | 传统方式 | Ollama 的方式 |
|------|----------|---------------|
| **部署复杂** | 手动下载模型权重、安装 PyTorch、配置 CUDA、写推理代码 | `ollama run qwen2.5` 一行命令搞定 |
| **隐私顾虑** | 数据发送到云端 API（OpenAI、Claude 等） | 所有数据在本地处理，不出网络 |
| **持续费用** | 按 Token 计费，用越多花越多 | 一次下载，无限使用，零费用 |

### 一句话理解
**Ollama 之于大模型，就像 Docker 之于应用部署** —— 把复杂的环境配置封装起来，让你只需关心"跑什么模型"，而不用关心"怎么跑起来"。

---

## 2. 架构与技术原理

```
┌──────────────────────────────────────────┐
│              用户交互层                    │
│   CLI (ollama run)  /  REST API (:11434) │
└──────────────┬───────────────────────────┘
               │
┌──────────────▼───────────────────────────┐
│             Ollama 服务层                  │
│  ┌─────────┐  ┌──────────┐  ┌─────────┐ │
│  │模型管理  │  │会话管理   │  │API路由  │ │
│  │下载/删除 │  │上下文维护 │  │请求分发  │ │
│  └─────────┘  └──────────┘  └─────────┘ │
└──────────────┬───────────────────────────┘
               │
┌──────────────▼───────────────────────────┐
│           推理引擎层 (llama.cpp)           │
│  ┌─────────┐  ┌──────────┐  ┌─────────┐ │
│  │CPU推理   │  │GPU加速    │  │量化优化  │ │
│  │(AVX/ARM) │  │(CUDA/Metal│  │(Q4/Q8)  │ │
│  └─────────┘  └──────────┘  └─────────┘ │
└──────────────────────────────────────────┘
```

**关键技术点：**

- **底层引擎**：基于 [llama.cpp](https://github.com/ggerganov/llama.cpp)，一个用 C/C++ 编写的高性能 LLM 推理库
- **模型格式**：使用 GGUF 格式，支持多种量化级别（Q4_0、Q4_K_M、Q8_0 等），用更少的内存跑更大的模型
- **GPU 加速**：macOS 上使用 Metal（Apple 芯片原生支持），Linux/Windows 上使用 CUDA（NVIDIA 显卡）
- **服务模式**：后台运行一个 HTTP 服务（默认端口 11434），通过 REST API 对外提供推理能力

---

## 3. 安装部署

### macOS（推荐 Homebrew）
```bash
brew install ollama
```

### macOS / Linux（官方脚本）
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Windows（PowerShell）
```powershell
irm https://ollama.com/install.ps1 | iex
```

### Docker
```bash
# CPU 模式
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama

# NVIDIA GPU 模式
docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama

# 在容器内运行模型
docker exec -it ollama ollama run llama3
```

### 验证安装
```bash
ollama --version        # 查看版本
ollama serve            # 启动服务（macOS 桌面版自动启动）
```

---

## 4. 基础使用

### 常用命令速查

```bash
# 运行模型（自动下载 + 启动对话）
ollama run qwen2.5:7b

# 模型管理
ollama list             # 查看已下载的模型
ollama pull llama3      # 只下载不运行
ollama rm llama3        # 删除模型
ollama show llama3      # 查看模型详情

# 从文件创建自定义模型
ollama create mymodel -f Modelfile

# 复制模型
ollama cp llama3 my-llama3
```

### 对话示例
```bash
$ ollama run qwen2.5:7b
>>> 用一句话解释什么是 Transformer
Transformer 是一种基于自注意力机制的神经网络架构，它让模型能够同时关注输入序列中的所有位置，
从而高效地处理序列到序列的任务。

>>> /bye    # 退出对话
```

### 多模态（视觉模型）
```bash
# 运行支持图片的模型
ollama run llava

# 在对话中传入图片
>>> 描述一下这张图片 /path/to/image.jpg
```

---

## 5. 支持的模型

完整模型库：https://ollama.com/library

### 热门模型推荐

| 模型 | 参数量 | 磁盘占用 | 特点 | 命令 |
|------|--------|----------|------|------|
| **Qwen 2.5** | 7B | ~4.7GB | 中文能力强，阿里出品 | `ollama run qwen2.5:7b` |
| **Llama 3** | 8B | ~4.7GB | Meta 出品，英文综合能力强 | `ollama run llama3` |
| **DeepSeek-R1** | 7B | ~4.7GB | 推理能力突出 | `ollama run deepseek-r1:7b` |
| **Gemma 3** | 4B | ~3.3GB | Google 出品，轻量高效 | `ollama run gemma3:4b` |
| **Phi-4** | 14B | ~9.1GB | 微软出品，小模型大智慧 | `ollama run phi4` |
| **Mistral** | 7B | ~4.1GB | 欧洲开源之光 | `ollama run mistral` |
| **LLaVA** | 7B | ~4.5GB | 多模态，支持图片理解 | `ollama run llava` |
| **CodeLlama** | 7B | ~3.8GB | 专注代码生成 | `ollama run codellama` |

### 模型标签说明
```bash
ollama run qwen2.5:7b       # 7B 参数版本
ollama run qwen2.5:72b      # 72B 参数版本（需要大内存）
ollama run qwen2.5:7b-q4_0  # 4-bit 量化版（更省内存）
```

---

## 6. REST API 详解

Ollama 启动后默认在 `http://localhost:11434` 提供 REST API，可以直接被应用程序调用。

### 生成补全（Generate）
```bash
curl http://localhost:11434/api/generate -d '{
  "model": "qwen2.5:7b",
  "prompt": "为什么天空是蓝色的？",
  "stream": false
}'
```

### 对话（Chat）
```bash
curl http://localhost:11434/api/chat -d '{
  "model": "qwen2.5:7b",
  "messages": [
    {"role": "system", "content": "你是一个友好的AI助手"},
    {"role": "user", "content": "你好"}
  ],
  "stream": false
}'
```

### 生成 Embedding 向量
```bash
curl http://localhost:11434/api/embed -d '{
  "model": "qwen2.5:7b",
  "input": "Ollama is awesome"
}'
```

### 查看本地模型列表
```bash
curl http://localhost:11434/api/tags
```

### Python SDK
```python
# pip install ollama
import ollama

# 对话
response = ollama.chat(
    model='qwen2.5:7b',
    messages=[{'role': 'user', 'content': '你好'}]
)
print(response['message']['content'])

# 流式输出
for chunk in ollama.chat(model='qwen2.5:7b',
                         messages=[{'role': 'user', 'content': '讲个笑话'}],
                         stream=True):
    print(chunk['message']['content'], end='')
```

### JavaScript SDK
```javascript
// npm install ollama
import ollama from 'ollama';

const response = await ollama.chat({
  model: 'qwen2.5:7b',
  messages: [{ role: 'user', content: '你好' }],
});
console.log(response.message.content);
```

---

## 7. 自定义模型 (Modelfile)

Modelfile 类似 Dockerfile，用于定义自定义模型的配置。

### 示例：创建一个中文翻译助手
```dockerfile
# 基于 qwen2.5 创建
FROM qwen2.5:7b

# 设置系统提示词
SYSTEM """
你是一个专业的中英文翻译助手。用户输入中文，你翻译成英文；用户输入英文，你翻译成中文。
只输出翻译结果，不要解释。
"""

# 调低温度，让翻译更准确
PARAMETER temperature 0.3

# 设置上下文长度
PARAMETER num_ctx 4096
```

### 创建并运行
```bash
ollama create translator -f Modelfile
ollama run translator
>>> 今天天气真好
The weather is really nice today.
```

### 导入第三方 GGUF 模型
```dockerfile
# 直接从本地 GGUF 文件创建
FROM ./my-model.gguf
```

---

## 8. 硬件要求

### 内存需求（经验法则）

| 模型参数量 | 最低内存 | 推荐内存 | 适合场景 |
|-----------|---------|---------|---------|
| 1B - 3B | 4GB | 8GB | 简单问答、轻量任务 |
| 7B - 8B | 8GB | 16GB | 日常使用、代码辅助 |
| 13B - 14B | 16GB | 32GB | 复杂推理、长文本 |
| 30B - 34B | 32GB | 64GB | 高质量生成 |
| 70B+ | 64GB+ | 128GB+ | 接近商业模型效果 |

### GPU 加速支持

| 平台 | GPU 类型 | 说明 |
|------|---------|------|
| macOS | Apple Silicon (M1/M2/M3/M4) | Metal 原生加速，开箱即用 |
| Linux/Windows | NVIDIA (RTX 3060+) | 需安装 CUDA 驱动 |
| Linux | AMD (RX 6000+) | ROCm 支持 |

### Apple Silicon 实际体验
- **M1/M2 (8GB)**：能跑 7B 量化模型，速度约 10-20 tokens/s
- **M1/M2 Pro (16GB)**：流畅跑 7B-13B，速度约 20-40 tokens/s
- **M1/M2 Max/Ultra (32GB+)**：可跑 30B-70B 模型

---

## 9. 生态集成

Ollama 提供标准的 REST API，已被大量工具和框架集成。

### 聊天界面
| 工具 | 说明 |
|------|------|
| [Open WebUI](https://github.com/open-webui/open-webui) | 最流行的 Ollama Web 界面，类似 ChatGPT 体验 |
| [AnythingLLM](https://github.com/Mintplex-Labs/anything-llm) | 支持 RAG 的桌面客户端 |

### 开发框架
| 框架 | 说明 |
|------|------|
| [LangChain](https://github.com/langchain-ai/langchain) | 主流 LLM 应用开发框架，原生支持 Ollama |
| [LlamaIndex](https://github.com/run-llama/llama_index) | RAG 框架，可用 Ollama 作为本地 LLM |
| [CrewAI](https://github.com/crewAIInc/crewAI) | 多 Agent 框架，支持 Ollama 后端 |

### 编辑器集成
| 工具 | 说明 |
|------|------|
| [Continue](https://github.com/continuedev/continue) | VS Code / JetBrains 的 AI 编程助手，支持 Ollama |
| [Cline](https://github.com/cline/cline) | VS Code AI Agent，可用 Ollama 本地模型 |

### 实际应用场景
前面我们看过的 [World Monitor](https://github.com/koala73/worldmonitor) 项目就是一个典型案例——用 Ollama 在本地跑模型做新闻摘要和情报分析，数据完全不出本地。
