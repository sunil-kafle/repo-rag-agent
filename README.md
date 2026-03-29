# Course RAG Agent

## Overview

Course RAG Agent is a Python application for repository-aware question answering over a structured documentation corpus. It combines lexical retrieval, vector retrieval, hybrid retrieval, and an agent layer behind a FastAPI interface. The system is organized so retrieval, answer generation, evaluation, and API delivery remain modular, reusable, and independently testable.

At runtime, a user submits a question through the API. The system retrieves relevant chunks from the indexed repository, prepares grounded context, invokes the answer-generation layer, and returns a response with citations and optional debug information.

## Purpose

The project is designed to answer questions about a repository using the repository’s own indexed content rather than unsupported generic responses. It supports:
- Repository search using multiple retrieval strategies
- Grounded answer generation with source references
- Retrieval quality evaluation using standard IR metrics
- Judge-based evaluation utilities for saved interaction logs
- Evaluation-question generation from repository documents
- API delivery for local development and future deployment

## Architecture

### Runtime Flow

User question -> API route -> retrieval service -> retrieval strategy -> normalized context -> agent -> answer response

### Application Layers

- `app/`
  - FastAPI application layer
  - Route definitions
  - Request and response schemas
  - Service orchestration
- `src/`
  - Core project logic
  - Retrieval modules
  - Agent modules
  - Evaluation utilities
  - Configuration and artifact loading
- `data/`
  - Saved retrieval artifacts
  - Processed corpus outputs
  - Evaluation outputs
- `scripts/`
  - Runnable utilities such as smoke tests and evaluation scripts
- `tests/`
  - API, retrieval, evaluation, and script validation

## Project Structure

    course/
      app/
        main.py
        routes/
          ask.py
          debug.py
          health.py
        schemas/
          api.py
        services/
          agent_service.py
          retrieval_service.py

      src/
        agent/
          builder.py
          prompts.py
          tools.py
        evaluation/
          data_generation.py
          ir_metrics.py
          judge.py
          logging_utils.py
          metrics.py
          schemas.py
        retrieval/
          base.py
          formatting.py
          hybrid.py
          lexical.py
          query.py
          vector.py
        artifacts.py
        config.py
        exceptions.py

      data/
        artifacts/
          documents.json
          embedding_matrix.npy
          lexical_index.pkl
        processed/
          all_chunks.json
        eval/

      logs/
      notebooks/
        repo_ingestion.ipynb
      scripts/
        __init__.py
        generate_eval_data.py
        run_eval.py
        smoke_test.py
      tests/
        fixtures/
        conftest.py
        test_api_ask.py
        test_api_health.py
        test_evaluation_judge.py
        test_evaluation_metrics.py
        test_logging_utils.py
        test_query_formulation.py
        test_retrieval_bm25.py
        test_retrieval_hybrid.py
        test_retrieval_vector.py
        test_run_eval.py
      README.md
      pyproject.toml
      uv.lock

## Core Components

### Configuration

`src/config.py` centralizes project settings, artifact paths, retrieval defaults, application metadata, and external key resolution. The application supports OpenAI key loading through an external file path or environment variable so secrets remain outside the repository.

### Artifact Loading

`src/artifacts.py` restores the saved retrieval state from disk. It loads:
- `documents.json`
- `embedding_matrix.npy`
- `lexical_index.pkl`

It validates artifact presence and structure, rebuilds the document lookup map, and exposes a cached retrieval artifact container for reuse across the application.

### Retrieval Contracts

`src/retrieval/base.py` defines the canonical data models used across retrieval:
- `Document`
- `RetrievalResult`
- `RetrievalResponse`

These shared contracts keep BM25, vector, and hybrid search outputs consistent regardless of strategy.

### Query Formulation

`src/retrieval/query.py` converts user questions into compact lexical search queries. This is especially useful for BM25 retrieval, where shorter keyword-style search terms often perform better than raw natural-language questions.

Example:
- User question: `How do embeddings work in this repo?`
- Effective lexical query: `embeddings work`

### Retrieval Methods

#### BM25 Retrieval

`src/retrieval/lexical.py` implements BM25-based search over tokenized document chunks using:
- Precomputed term frequencies
- Document frequencies
- Document lengths
- Average document length

This is the default runtime retrieval method when combined with query formulation.

#### Vector Retrieval

`src/retrieval/vector.py` implements semantic search using:
- `sentence-transformers/all-MiniLM-L6-v2`
- A saved embedding matrix
- Normalized query embeddings
- Dot-product similarity over normalized vectors

This method is useful when semantic similarity matters more than exact lexical overlap.

#### Hybrid Retrieval

`src/retrieval/hybrid.py` combines BM25 and vector search using Reciprocal Rank Fusion (RRF). This avoids comparing raw lexical and vector scores directly while still taking advantage of both ranking methods.

### Formatting and Source Handling

`src/retrieval/formatting.py` handles:
- Repository path normalization
- GitHub blob URL construction
- Source label formatting
- Snippet generation

This keeps source display logic separate from retrieval logic.

### Retrieval Service

`app/services/retrieval_service.py` is the application-level retrieval orchestrator. It decides:
- Which retrieval strategy to run
- When query formulation should be applied
- How to prepare normalized results for the API layer
- How to expose retrieval debug information

Current runtime policy:
- Default strategy: BM25
- Default BM25 behavior: query formulation applied before retrieval

### Agent Layer

The agent layer lives under `src/agent/`.

#### `prompts.py`
Defines the system instructions for repository-grounded answering.

#### `tools.py`
Exposes the retrieval tool used by the agent. The tool returns compact, citation-friendly search results with normalized paths and GitHub URLs.

#### `builder.py`
Builds the runtime agent and resolves the OpenAI API key before model initialization.

### Agent Service

`app/services/agent_service.py` runs the runtime agent and extracts citations from markdown links in the generated answer. It returns a structured answer payload used by the `/ask` route.

### API Layer

#### `GET /health`
Returns a basic readiness response.

#### `POST /ask`
Runs full retrieval-backed answer generation and returns:
- Answer text
- Citations
- Retrieval strategy used
- Formulated query in debug mode
- Retrieved results in debug mode

#### `POST /debug/retrieve`
Returns retrieval-only output without running the agent. This is useful for:
- Comparing retrieval strategies
- Inspecting effective queries
- Checking ranking behavior
- Debugging retrieval quality independently from answer generation

## Evaluation Layer

The project now includes a reusable evaluation layer under `src/evaluation/`.

### Evaluation Schemas

`src/evaluation/schemas.py` defines structured evaluation contracts, including:
- `EvaluationCheck`
- `EvaluationChecklist`
- `GeneratedQuestions`
- `EvaluationRow`
- `EvaluationSummary`

These models keep judge-based evaluation, flattened results, and aggregate summaries consistent.

### Judge-Based Evaluation

`src/evaluation/judge.py` contains reusable helpers for evaluating saved agent logs with a judge agent. It includes:
- Evaluation prompt definition
- Evaluation input template
- Judge agent builder
- Log loading helper
- Log simplification helper
- `evaluate_log_record(...)`

This separates evaluation logic from notebook-only experimentation and makes it reusable from scripts or later workflows.

### Evaluation Question Generation

`src/evaluation/data_generation.py` contains the question-generation workflow used to create evaluation questions from sampled repository documents. It includes:
- Question-generation prompt
- Question generator builder
- Prompt construction helper
- Async generation helper

This supports repeatable creation of evaluation inputs directly from the indexed corpus.

### Evaluation Metrics and Flattening

`src/evaluation/metrics.py` contains helpers for:
- Converting structured judge output into flattened evaluation rows
- Building rows from notebook-style evaluation payloads
- Computing aggregate pass counts, totals, and pass rates

This separates evaluation summarization from runtime answer generation.

### Retrieval IR Metrics

`src/evaluation/ir_metrics.py` supports retrieval evaluation using:
- Hit Rate
- Mean Reciprocal Rank (MRR)

It provides:
- `evaluate_search_quality(...)`
- `summarize_ir_metrics(...)`

## Logging Utilities

`src/evaluation/logging_utils.py` provides reusable helpers for:
- Building normalized interaction records
- Ensuring the logs directory exists
- Saving logs as JSON
- Loading saved log files
- Simplifying messages for later evaluation workflows

This supports structured run analysis and later judge-based evaluation.

## Retrieval Evaluation

The project includes a retrieval evaluation workflow using standard information retrieval metrics.

### Methods Compared

- BM25 without query formulation
- BM25 with query formulation
- Vector retrieval
- Hybrid retrieval

### Current Evaluation Snapshot

- `bm25_raw: hit_rate=0.60, mrr=0.350`
- `bm25_formulated: hit_rate=0.80, mrr=0.700`
- `vector: hit_rate=0.80, mrr=0.567`
- `hybrid: hit_rate=0.80, mrr=0.667`

Current runtime default:
- BM25 with query formulation

## Running the Project

### Start the API

From the project root:

    uv run uvicorn app.main:app --reload
    - open http://127.0.0.1:8000/
    - explain that /ask powers the frontend chat

Then open:
- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/docs`

### Run the Smoke Test

    uv run python -m scripts.smoke_test

This verifies:
- Configuration loading
- Artifact loading
- BM25 retrieval
- Retrieval service behavior
- Agent tool integration

### Generate Evaluation Questions

    uv run python -m scripts.generate_eval_data

This samples repository documents, generates evaluation questions, and saves them to:

    data/eval/generated_questions.json

### Run Retrieval Evaluation

    uv run python -m scripts.run_eval

This compares retrieval methods, prints evaluation summaries, and saves the retrieval evaluation output to:

    data/eval/retrieval_eval_summary.json

### Run Tests

    uv run pytest -q

## Test Coverage

The project currently includes automated tests for:
- API health route
- API ask route
- Query formulation
- BM25 retrieval
- Vector retrieval
- Hybrid retrieval
- Evaluation metrics helpers
- Evaluation judge helpers
- Logging utilities
- Retrieval evaluation script behavior

Current validated suite state:
- `31 passed`

## Environment and Secrets

The application supports external OpenAI key loading so secrets remain outside the repository.

Expected external key file:

    C:\projects\OPEN_AI.txt

Accepted file contents:

    sk-...

or

    OPENAI_API_KEY=sk-...

## Design Principles

- Python modules are the source of truth
- Retrieval logic remains framework-agnostic
- API routes stay thin
- Services handle orchestration
- Retrieval outputs use a standardized schema
- Artifact loading is centralized
- Evaluation is separated from runtime request handling
- External secrets remain outside the repository
- Debugging and testing are part of the application structure

## Current Capabilities

The project currently provides:
- A working FastAPI application
- Grounded repository question answering
- Retrieval-only debugging
- Reusable retrieval modules
- Reusable evaluation utilities
- Automated tests for API, retrieval, evaluation, and scripts
- Smoke and evaluation scripts for quick validation

## Extension Paths

The current structure supports future additions such as:
- Expanded judge workflows and evaluation datasets
- Artifact manifest validation
- Ingestion pipeline modularization
- Richer developer documentation
- Deployment-oriented packaging improvements
- Optional frontend integration on top of the FastAPI backend