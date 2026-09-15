# 🚀 AI Camp: LangChain & LangGraph 应用开发实战项目

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/Framework-LangChain-green.svg)](https://www.langchain.com/)
[![LangGraph](https://img.shields.io/badge/Framework-LangGraph-orange.svg)](https://www.langchain.com/langgraph)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

本项目是我在参加 **AI 大模型应用开发训练营（AI Camp）** 期间的沉淀与实战代码库。项目涵盖了从 **LangChain** 基础组件（Prompt、Chain、Memory、Agent）到 **LangGraph** 高级状态图编排（State、Node、Conditional Edge、Multi-Agent Workflow）的完整学习链路与 Lab 实验复现。

---

## 目录

- [项目亮点](#-项目亮点)
- [目录结构](#-目录结构)
- [核心模块说明](#-核心模块说明)
  - [1. LangChain 核心与 Lab 实验](#1-langchain-核心与-lab-实验)
  - [2. LangGraph 状态图与高级编排](#2-langgraph-状态图与高级编排)
- [环境配置与快速开始](#-环境配置与快速开始)
  - [依赖安装](#依赖安装)
  - [环境变量配置](#环境变量配置)
  - [运行示例](#运行示例)
- [总结与心得](#-总结与心得)

---

## ✨ 项目亮点

1. **体系化渐进式学习**：从最基础的单链（LLMChain）与记忆机制，逐步演进至复杂的逻辑顺序链（SequentialChain）与智能体图结构编排。
2. **丰富的 Memory 管理实践**：系统测试并对比了 `ConversationBufferWindowMemory` 与 `ConversationSummaryMemory` 在长文本会话及上下文压缩中的应用效果。
3. **LangGraph 状态驱动编排**：全面实践基于 `TypedDict` 的全局 State 管理、节点（Node）任务处理逻辑以及条件边（Conditional Edge）的动态路由跳转。
4. **完整实验复现逻辑**：包含 Lab 1 至 Lab 5 的原始代码、优化重构版本（`reproduce-lab`），方便对比调试与深入理解实现机制。

---

## 📁 目录结构

```text
ai-camp/
├── 1. Langchain/                    # 模块一：LangChain 核心组件与基础 Lab 实验
│   ├── chain/                       # 链式结构 (Chains) 实践
│   │   ├── LLMChain .py             # 基础 LLMChain 调用示例
│   │   ├── LLMChain + Memory .py    # 结合 Memory 的持久化对话链
│   │   └── SequentialChain .py      # 多任务顺序链 (SequentialChain) 编排
│   ├── memory/                      # 记忆组件 (Memory) 实践
│   │   ├── 1. .py                   # 基础 Memory 测试
│   │   ├── ConversationBufferWindowMemory.py  # 滑动窗口记忆机制
│   │   ├── ConversationSummaryMemory.py       # LLM 自动摘要记忆机制
│   │   ├── memory-lab1.py                     # Lab1 记忆集成测试
│   │   ├── session-memory.py                  # 会话级状态隔离与内存管理
│   │   └── memory.py                          # 记忆接口通用实践
│   ├── lab1.py                      # Lab 1: Prompt 提示词与模型调用基础
│   ├── lab2.py                      # Lab 2: 链式逻辑与数据流处理
│   ├── reproduce_lab2.py            # Lab 2 代码重构与细节复现
│   ├── lab3.py / lab3-backup.py     # Lab 3: 复杂 Task 拆解与上下文链
│   ├── reproduce-lab3.py            # Lab 3 改进与复现版本
│   ├── lab4.py                      # Lab 4: Tool Calling 与 Agent 基础
│   └── produce-lab4.py              # Lab 4 生产级实践扩展
│
├── 2. Langgraph/                    # 模块二：LangGraph 状态图与 Agent 编排
│   ├── edge/                        # 图节点连接与路由控制 (Edges)
│   │   ├── edge.py                  # 标准静态逻辑边
│   │   └── conditional-edge.py      # 基于状态判断的条件边 (Conditional Edge)
│   ├── check-word.py                # 文本敏感词审查与 Guardrail 校验节点
│   ├── node.py                      # Graph 业务处理节点定义 (Nodes)
│   ├── state.py                     # 全局状态定义 (TypedDict State)
│   ├── test_langgraph.py            # LangGraph 图结构单元测试与运行
│   ├── lab5.py                      # Lab 5: 基于 LangGraph 的复杂状态图 Workflow
│   └── reproduce-lab5.py            # Lab 5 完整复现与多分支优化
│
└── README.md                        # 本项目说明文档
