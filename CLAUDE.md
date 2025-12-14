<!-- OPENSPEC:START -->
# OpenSpec Instructions

These instructions are for AI assistants working in this project.

Always open `@/openspec/AGENTS.md` when the request:
- Mentions planning or proposals (words like proposal, spec, change, plan)
- Introduces new capabilities, breaking changes, architecture shifts, or big performance/security work
- Sounds ambiguous and you need the authoritative spec before coding

Use `@/openspec/AGENTS.md` to learn:
- How to create and apply change proposals
- Spec format and conventions
- Project structure and guidelines

Keep this managed block so 'openspec update' can refresh the instructions.

<!-- OPENSPEC:END -->

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a personal AI learning knowledge base (tech_blog/AI-docs) documenting deep research into AI concepts, theories, and implementations. Documentation is primarily in Chinese with technical terms in English.

## Current Research Topics

- **Sequence Modeling** (`rnn_transformer_mamba/`): RNN → Transformer → Linear Attention → Mamba/SSM evolution
- **Mixture of Experts** (`MoE/`): Historical evolution from 1991-2024, includes Marp presentations
- **World Models** (`world_models/`): Model-based RL, Dreamer series, with practical experiments

## Commands

### World Models Experiments
```bash
# Dependencies
pip install torch gymnasium numpy matplotlib

# Run experiments (in world_models/experiments/)
python 1_baseline_dqn.py
python 2_simple_world_model.py
python 3_mini_dreamer.py
python compare_results.py
```

### Marp Presentations
```bash
npm install -g @marp-team/marp-cli
marp presentation.md -o presentation.pptx
marp presentation.md -o presentation.pdf
```

## Architecture

- **Research Plans**: Each topic has a structured `research_plan.md` or `learning_plan.md` with historical context, core papers, and implementation projects
- **Experiments**: Comparative approach (baseline vs improved methods) with sample efficiency metrics
- **Platform**: MacBook with MPS (Apple Silicon) for PyTorch experiments
