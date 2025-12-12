# LangChain Workspace — What You Can Do

This repo is a practical playground to build, compose, and run AI workflows with LangChain. Instead of listing files, this guide focuses on what you can do using the provided examples.

## Capabilities

- Build and compose model workflows
  - Run a simple prompt→LLM pipeline.
  - Chain multiple steps sequentially (output of one feeds the next).
  - Fan-out/fan-in parallel steps to compare or aggregate results.

- Create embedding-powered chatbots and RAG-style utilities
  - Generate vector embeddings for texts.
  - Retrieve relevant context and answer user queries.
  - Prototype a chatbot interface using embeddings.

- Parse and validate LLM outputs reliably
  - Enforce JSON output format and parse it safely.
  - Convert free-form strings to structured objects.
  - Use Pydantic models to validate and coerce outputs.

- Use LangChain Runnables for composable pipelines
  - Treat steps as `Runnable` units you can compose, map, and transform.

- Experiment with YouTube content
  - Prototype a chatbot over YouTube transcripts/metadata (ensure loaders/keys).
  - Run demos that stitch together embeddings, retrieval, and responses.

## Install & Setup (Windows-friendly)

Prerequisites
- Python 3.10+ is recommended.
- API keys as needed (e.g., OpenAI). Store them in environment variables.

Environment setup

```bash
# From the repo root
python -m venv .venv
. .venv/Scripts/activate  # PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Configure model provider keys (examples)

```bash
# OpenAI (PowerShell)
$env:OPENAI_API_KEY="your-key"

# OpenAI (CMD)
setx OPENAI_API_KEY "your-key"  # restart shell after
```

## Quick Start Tasks

Run a simple model chain

```bash
python Chains/simpleChain.py
```

Compose multiple steps sequentially

```bash
python Chains/seqChain.py
```

Compare or aggregate via parallel branches

```bash
python Chains/paraChain.py
```

Spin up an embedding-powered chatbot

```bash
python Embedding/simpleChatbot.py
```

Guarantee well-formed JSON from the LLM

```bash
python Embedding/Jsonoutput.py
```

Parse free-form strings into structured data

```bash
python Embedding/stringoutputparser.py
python Embedding/outputParser.py
```

Validate outputs with Pydantic schemas

```bash
python Embedding/pydantic1.py
python Embedding/struc_output.py
python Embedding/strucoutput.py
```

Work with composable `Runnable` primitives

```bash
python Runnable/1.py
```

Prototype YouTube chatbot experiments

```bash
python Youtube/main.py
python Youtube/chatbot.py
```

## Tips

- Some demos require provider-specific keys; check the script for env vars.
- Prefer structured parsers (Pydantic) when strict output formats are needed.
- Keep any generated client JSON private if it contains credentials.

## Extend This Playground

- Centralize configuration (API keys, model names) in one module.
- Add transcript loaders for consistent YouTube experiments.
- Add lightweight tests around parsers and chain composition.
