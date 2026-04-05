"""
PrajnaAI — AI Agents with LangGraph
=====================================
Multi-agent workflows using LangGraph state machines.
All agents use local Ollama LLMs — zero cloud dependency.

Agents implemented:
  1. ResearchAgent   — Plan → Retrieve → Synthesize
  2. CodeReviewAgent — Analyze → Critique → Suggest
  3. MathAgent       — Parse → Calculate → Explain

Graph patterns: Linear, Conditional branching, Loops
"""

import sys
import time
import json
import math
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from typing import TypedDict, Annotated, List, Optional
import operator

from loguru import logger

# LangGraph
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# LangChain
from langchain_community.llms import Ollama
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

# ── Config ─────────────────────────────────────────────────────────────────
LLM_MODEL = "gemma2:2b"
ROOT = Path(__file__).parent


# ═══════════════════════════════════════════════════════════════════════════
# TOOLS (CPU-friendly, offline)
# ═══════════════════════════════════════════════════════════════════════════

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely. Input: math expression string."""
    try:
        # Safe evaluation whitelist
        allowed = {
            "abs": abs, "round": round, "min": min, "max": max,
            "sum": sum, "pow": pow, "sqrt": math.sqrt,
            "log": math.log, "log10": math.log10, "exp": math.exp,
            "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "pi": math.pi, "e": math.e,
        }
        result = eval(expression, {"__builtins__": {}}, allowed)
        return f"Result: {result}"
    except Exception as e:
        return f"Error evaluating expression: {e}"


@tool
def word_counter(text: str) -> str:
    """Count words, sentences, and characters in text."""
    words = len(text.split())
    sentences = text.count(".") + text.count("!") + text.count("?")
    chars = len(text)
    return json.dumps({
        "words": words,
        "sentences": max(1, sentences),
        "characters": chars,
        "avg_word_length": round(chars / max(1, words), 1)
    })


@tool
def python_syntax_checker(code: str) -> str:
    """Check Python code for syntax errors."""
    import ast
    try:
        tree = ast.parse(code)
        # Count node types
        funcs = sum(1 for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
        classes = sum(1 for n in ast.walk(tree) if isinstance(n, ast.ClassDef))
        imports = sum(1 for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)))
        return json.dumps({
            "status": "valid",
            "functions": funcs,
            "classes": classes,
            "imports": imports,
            "lines": len(code.splitlines())
        })
    except SyntaxError as e:
        return json.dumps({"status": "syntax_error", "message": str(e), "line": e.lineno})


@tool
def text_summarizer_tool(text: str) -> str:
    """Extract key sentences from text as a simple extractive summary."""
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if len(s.strip()) > 30]
    # Simple frequency-based selection: first, middle, last sentences
    if len(sentences) <= 3:
        return ". ".join(sentences)
    selected = [sentences[0], sentences[len(sentences)//2], sentences[-1]]
    return ". ".join(selected) + "."


# ═══════════════════════════════════════════════════════════════════════════
# AGENT 1: RESEARCH AGENT (Linear Pipeline)
# ═══════════════════════════════════════════════════════════════════════════

class ResearchState(TypedDict):
    topic: str
    research_plan: str
    gathered_info: str
    final_report: str
    step_count: int


class ResearchAgent:
    """
    A 3-step research agent:
      plan → research → synthesize
    Uses LangGraph linear state machine.
    """

    def __init__(self, model: str = LLM_MODEL):
        self.llm = Ollama(model=model, temperature=0.3, num_predict=400)
        self.graph = self._build_graph()

    def _plan_step(self, state: ResearchState) -> ResearchState:
        """Generate a research plan for the topic."""
        prompt = f"""You are a research planner. Create a brief 3-point research plan for:
Topic: {state['topic']}

Format:
1. [First aspect to investigate]
2. [Second aspect to investigate]  
3. [Key question to answer]

Keep it concise (under 100 words)."""
        plan = self.llm.invoke(prompt)
        return {**state, "research_plan": plan, "step_count": state["step_count"] + 1}

    def _research_step(self, state: ResearchState) -> ResearchState:
        """Gather information based on the plan."""
        prompt = f"""Based on this research plan, provide factual information:

Topic: {state['topic']}
Plan: {state['research_plan']}

Provide 3 key facts or insights. Be specific and accurate. Under 150 words."""
        info = self.llm.invoke(prompt)
        return {**state, "gathered_info": info, "step_count": state["step_count"] + 1}

    def _synthesize_step(self, state: ResearchState) -> ResearchState:
        """Synthesize gathered information into a final report."""
        prompt = f"""Synthesize this research into a clear, structured mini-report:

Topic: {state['topic']}
Information: {state['gathered_info']}

Write a 2-paragraph report with: overview + key takeaways. Under 200 words."""
        report = self.llm.invoke(prompt)
        return {**state, "final_report": report, "step_count": state["step_count"] + 1}

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(ResearchState)
        workflow.add_node("plan", self._plan_step)
        workflow.add_node("research", self._research_step)
        workflow.add_node("synthesize", self._synthesize_step)
        workflow.set_entry_point("plan")
        workflow.add_edge("plan", "research")
        workflow.add_edge("research", "synthesize")
        workflow.add_edge("synthesize", END)
        return workflow.compile()

    def run(self, topic: str) -> dict:
        initial = ResearchState(
            topic=topic, research_plan="", gathered_info="",
            final_report="", step_count=0
        )
        t0 = time.time()
        result = self.graph.invoke(initial)
        elapsed = time.time() - t0
        return {**result, "latency_s": elapsed}


# ═══════════════════════════════════════════════════════════════════════════
# AGENT 2: CODE REVIEW AGENT (Conditional Branching)
# ═══════════════════════════════════════════════════════════════════════════

class CodeReviewState(TypedDict):
    code: str
    language: str
    syntax_check: str
    severity: str          # "pass" | "minor" | "major"
    review_feedback: str
    suggestions: str
    step_count: int


class CodeReviewAgent:
    """
    Code review agent with conditional routing:
      - PASS  → brief positive feedback
      - MINOR → suggestions for improvement
      - MAJOR → detailed fix recommendations
    """

    def __init__(self, model: str = LLM_MODEL):
        self.llm = Ollama(model=model, temperature=0.2, num_predict=350)
        self.graph = self._build_graph()

    def _syntax_check_node(self, state: CodeReviewState) -> CodeReviewState:
        result = python_syntax_checker.invoke(state["code"])
        return {**state, "syntax_check": result, "step_count": state["step_count"] + 1}

    def _assess_severity(self, state: CodeReviewState) -> str:
        """Route based on syntax check result."""
        try:
            data = json.loads(state["syntax_check"])
            if data["status"] == "syntax_error":
                return "major"
            elif data.get("functions", 0) == 0 and data.get("lines", 0) > 5:
                return "minor"
            else:
                return "pass"
        except Exception:
            return "minor"

    def _pass_review(self, state: CodeReviewState) -> CodeReviewState:
        feedback = self.llm.invoke(
            f"This Python code looks good. Give a 2-sentence positive review:\n{state['code'][:500]}"
        )
        return {**state, "severity": "pass", "review_feedback": feedback,
                "suggestions": "No major issues found.", "step_count": state["step_count"] + 1}

    def _minor_review(self, state: CodeReviewState) -> CodeReviewState:
        feedback = self.llm.invoke(
            f"Review this Python code and suggest 2-3 minor improvements:\n{state['code'][:500]}"
        )
        return {**state, "severity": "minor", "review_feedback": feedback,
                "suggestions": feedback, "step_count": state["step_count"] + 1}

    def _major_review(self, state: CodeReviewState) -> CodeReviewState:
        syntax_info = state["syntax_check"]
        feedback = self.llm.invoke(
            f"This Python code has errors. Syntax check: {syntax_info}\nCode:\n{state['code'][:500]}\n\nProvide specific fix instructions."
        )
        return {**state, "severity": "major", "review_feedback": feedback,
                "suggestions": feedback, "step_count": state["step_count"] + 1}

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(CodeReviewState)
        workflow.add_node("syntax_check", self._syntax_check_node)
        workflow.add_node("pass_review", self._pass_review)
        workflow.add_node("minor_review", self._minor_review)
        workflow.add_node("major_review", self._major_review)
        workflow.set_entry_point("syntax_check")
        workflow.add_conditional_edges(
            "syntax_check",
            self._assess_severity,
            {"pass": "pass_review", "minor": "minor_review", "major": "major_review"}
        )
        workflow.add_edge("pass_review", END)
        workflow.add_edge("minor_review", END)
        workflow.add_edge("major_review", END)
        return workflow.compile()

    def run(self, code: str, language: str = "python") -> dict:
        initial = CodeReviewState(
            code=code, language=language, syntax_check="",
            severity="", review_feedback="", suggestions="", step_count=0
        )
        t0 = time.time()
        result = self.graph.invoke(initial)
        return {**result, "latency_s": time.time() - t0}


# ═══════════════════════════════════════════════════════════════════════════
# MAIN DRIVER
# ═══════════════════════════════════════════════════════════════════════════

def main():
    logger.info("🕉️  PrajnaAI — AI Agents Module (LangGraph)")
    logger.info("="*60)

    # 1. Research Agent
    logger.info("\n🔬 Agent 1: Research Agent")
    research_agent = ResearchAgent()
    topics = ["CPU-based AI inference", "quantized language models"]

    for topic in topics:
        logger.info(f"\nResearching: '{topic}'")
        result = research_agent.run(topic)
        logger.info(f"Steps completed: {result['step_count']}")
        logger.info(f"Research Plan:\n{result['research_plan'][:200]}...")
        logger.info(f"Final Report:\n{result['final_report'][:300]}...")
        logger.info(f"Latency: {result['latency_s']:.2f}s")

    # 2. Code Review Agent
    logger.info("\n\n💻 Agent 2: Code Review Agent")
    review_agent = CodeReviewAgent()

    test_codes = {
        "Good Code": """
def fibonacci(n: int) -> list[int]:
    \"\"\"Generate Fibonacci sequence up to n terms.\"\"\"
    if n <= 0:
        return []
    sequence = [0, 1]
    while len(sequence) < n:
        sequence.append(sequence[-1] + sequence[-2])
    return sequence[:n]
""",
        "Buggy Code": """
def calculate_average(numbers)
    total = sum(numbers
    return total / len(numbers)
""",
    }

    for code_name, code in test_codes.items():
        logger.info(f"\nReviewing: {code_name}")
        result = review_agent.run(code)
        logger.info(f"Severity: {result['severity'].upper()}")
        logger.info(f"Feedback: {result['review_feedback'][:200]}...")

    # 3. Tool demo
    logger.info("\n\n🔧 Tool Demos:")
    logger.info(f"Calculator: sqrt(144) + 3^2 = {calculator.invoke('sqrt(144) + pow(3,2)')}")
    logger.info(f"Word Counter: {word_counter.invoke('Hello world. This is a test sentence!')}")

    logger.info("\n✅ Agents module complete!")
    logger.info("🎨 Launch UI: streamlit run src/agents/ui/app.py")


if __name__ == "__main__":
    main()
