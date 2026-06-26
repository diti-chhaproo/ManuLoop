# ManuLoop
ManuLoop is an explainable manufacturability co-pilot for mechanical engineers. It analyzes STEP CAD models using graph neural networks, identifies feature-level manufacturability risks, recommends actionable design changes, and closes the loop by comparing redesigned parts and quantifying their impact on manufacturing risk through FMEA.

Unlike traditional manufacturability analysis tools that stop at prediction, ManuLoop is designed to help engineers iterate on their designs **before costly manufacturing issues reach the shop floor.**

---

## The Problem

Manufacturing issues are often discovered after a part reaches process planning, tooling, or production, resulting in expensive engineering change orders and repeated design-manufacturing iteration.

Modern AI models can often predict that a part is difficult to manufacture, but they rarely answer the questions engineers actually care about:

* **Which feature caused the problem?**
* **Why is it problematic?**
* **What should I change?**
* **Did my redesign actually improve manufacturability?**

ManuLoop aims to answer those questions.

---

## Features

* Import STEP CAD models
* Convert B-Rep geometry into graph representations
* Predict manufacturability risk using Graph Neural Networks
* Highlight problematic faces/features with explainable AI
* Generate feature-level redesign recommendations
* Compare original and redesigned CAD models
* Visualize before/after manufacturability metrics
* Estimate manufacturing risk improvements through FMEA mapping

---

## System Workflow

```text
Engineer
    │
    ▼
Upload STEP File
    │
    ▼
Geometry Parsing
    │
    ▼
B-Rep Graph Construction
    │
    ▼
Manufacturability Assessment
    │
    ▼
Feature-Level Explainability
    │
    ▼
Redesign Recommendation
    │
    ▼
Engineer Redesigns Part
    │
    ▼
Re-analysis
    │
    ▼
Before / After Comparison
    │
    ▼
FMEA Risk Comparison
```

---

## System Architecture

The system is composed of the following modules:

* CAD Importer
* B-Rep Graph Builder
* Manufacturability Assessment Engine
* Explainability Layer
* Recommendation Engine
* Revision Comparator
* FMEA Mapper
* Dashboard UI

---

## Tech Stack

**CAD**

* STEP (ISO 10303)
* SolidWorks

**Machine Learning**

* PyTorch
* PyTorch Geometric
* Graph Neural Networks

**Geometry Processing**

* B-Rep parsing
* Face adjacency graphs

**Systems Engineering**

* SysML
* Requirements Engineering
* Functional Decomposition
* FMEA
* Design Verification

---

## Repository Structure

```text
ManuLoop/

├── docs/                  # Design documentation
├── diagrams/              # SysML diagrams
├── datasets/              # Dataset metadata
├── models/                # Trained GNN models
├── preprocessing/         # STEP → Graph pipeline
├── recommendation/        # Recommendation engine
├── explainability/        # Feature attribution
├── dashboard/             # UI
├── fmea/                  # FMEA comparison
├── tests/
└── README.md
```

---

## Current Development Status

This project is currently under active development.

### Planned Milestones

* STEP file parser
* B-Rep graph generation
* Manufacturability prediction model
* Explainability module
* Recommendation engine
* Revision comparison
* FMEA integration
* Dashboard
* Case study validation

---

## Project Goal

ManuLoop is intended to demonstrate how AI can support—not replace—mechanical engineers by providing explainable, actionable manufacturability feedback early in the design process.

The objective is to reduce costly design-manufacturing iteration loops while preserving engineering decision-making.

---

## Future Work

* Support for multiple manufacturing processes (CNC machining, sheet metal, injection molding)
* Integration with PLM/PDM workflows
* Multi-objective optimization (cost, manufacturability, sustainability)
* Automated engineering change impact analysis
* Human-in-the-loop recommendation refinement

---
