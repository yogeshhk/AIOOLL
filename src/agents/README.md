# Agents Module — AI Agents with LangGraph

## Overview

Multi-agent workflows using LangGraph state machines. All agents use local Ollama LLMs — no OpenAI, no cloud.

## Agents

| Agent | Graph Pattern | Description |
|-------|--------------|-------------|
| `ResearchAgent` | Linear (plan → research → synthesize) | Generates a research plan, gathers facts, writes a mini-report |
| `CodeReviewAgent` | Conditional branching (pass / minor / major) | Runs syntax check then routes to the appropriate review depth |

## Offline Tools (no Ollama needed)

| Tool | What it does |
|------|-------------|
| `calculator` | Safe `eval` with math functions whitelist |
| `word_counter` | Counts words, sentences, characters |
| `python_syntax_checker` | AST parse + counts functions/classes/imports |
| `text_summarizer_tool` | Extractive summary (first, middle, last sentences) |

## Files

```
agents/
├── driver.py           # Main entry point
├── ui/app.py           # Streamlit agent UI
└── tests/test_agents.py # Tests for all tools (no Ollama required)
```

## Setup

```bash
ollama serve &
ollama pull gemma2:2b
python driver.py
```

## Key Academic Concepts

- **State machine (LangGraph):** Each node is a function that reads and writes a typed state dict; edges define allowed transitions — explicit control flow vs. free-form ReAct loops
- **Conditional routing:** `add_conditional_edges` inspects state after a node and branches to different next nodes based on a routing function
- **Tool use:** `@tool` decorator exposes Python functions to the LLM with typed signatures and docstrings; the LLM decides when and how to call them
- **Safe eval:** Restricting `__builtins__` and providing an explicit allowlist prevents code injection in the calculator tool
