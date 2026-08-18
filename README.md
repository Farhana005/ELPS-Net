# Context-Aware LLM-Guided Neural Architecture Search for Medical Image Segmentation

## Overview

Medical image segmentation requires architectures that can achieve high segmentation accuracy while maintaining computational efficiency across heterogeneous imaging settings. Existing evolutionary neural architecture search (NAS) methods commonly employ fixed or randomly scheduled mutation strategies, which limit their ability to adapt the search process according to the evolving optimisation state.

To address this limitation, we propose **ELPS-Net**, a context-aware LLM-guided evolutionary NAS framework for lightweight medical image segmentation. ELPS-Net combines a **Particle Swarm Optimisation (PSO)-inspired mutation-intensity proposal** with bounded **Large Language Model (LLM) guidance** for mutation-intensity correction and targeted gene-level modification. Rather than directly generating architectures, the LLM receives contextual information from the current evolutionary state and provides constrained guidance on both the mutation intensity and the architectural genes to be modified. This enables adaptive exploration and exploitation throughout the search process.

The search is conducted over operator-level configurations within a fixed hierarchical encoder--decoder supernet containing three task-oriented modules:

* **Agra** – structural refinement for preserving anatomically relevant representations.
* **Rota** – lightweight multi-scale contextual aggregation for capturing information across different receptive fields.
* **Petra** – adaptive encoder--decoder fusion for selectively integrating features across network stages.

ELPS-Net employs a constrained **multi-objective evolutionary optimisation** strategy with Pareto-based selection to balance segmentation quality and architectural complexity. On the validation splits of **ACDC**, **MnM**, and **BraTS 2021**, the discovered architecture achieves DSC scores of **94.29%**, **91.13%**, and **92.45%**, respectively, while requiring only **0.48M parameters** and **4.38 GFLOPs**. Cross-dataset experiments on **MMWHS**, **CHAOS**, and **BraTS 2020** further demonstrate competitive validation performance after fine-tuning.

Overall, ELPS-Net provides a lightweight and adaptive evolutionary NAS framework in which PSO-inspired mutation control and bounded context-aware LLM guidance jointly improve architecture exploration without allowing the LLM to directly generate candidate architectures.

## Key Features

* **Context-aware evolutionary NAS** for lightweight medical image segmentation.
* **PSO-inspired adaptive mutation-intensity proposal** based on the current search state.
* **Bounded LLM-guided mutation correction** rather than unrestricted LLM-based architecture generation.
* **Targeted gene-level mutation**, allowing the LLM to identify architectural variables most relevant to the current evolutionary context.
* **Operator-level architecture search** within a fixed hierarchical encoder--decoder supernet.
* **Task-oriented Agra, Rota, and Petra modules** for structural refinement, multi-scale contextual aggregation, and adaptive feature fusion.
* **Pareto-based multi-objective optimisation** balancing segmentation quality and architectural complexity.
* **Compact architecture discovery**, achieving competitive segmentation performance with only **0.48M parameters and 4.38 GFLOPs**.
* **Cross-dataset adaptability** demonstrated through fine-tuning on heterogeneous medical imaging datasets.


## Experimental Results

ELPS-Net was extensively evaluated on multiple benchmark medical image segmentation datasets.

| Dataset    | DSC (%) | IoU (%) | HD95 (mm) |
| ---------- | ------- | ------- | --------- |
| ACDC       | 94.29   | 89.30   | 1.21      |
| MnM        | 91.13   | 83.86   | 1.38      |
| BraTS 2021 | 92.45   | 86.12   | 1.28      |

### Efficiency

* Parameters: **0.48M**
* FLOPs: **4.38 GFLOPs**
* Inference Speed: **51.8 FPS**

### Generalisation Evaluation

The discovered architecture was further fine-tuned and evaluated on **MMWHS, CHAOS, and BraTS 2020**, demonstrating reliable cross-domain adaptability while maintaining computational efficiency.

## Framework

ELPS-Net consists of:

1. **Evolutionary Neural Architecture Search (ENAS)**
2. **PSO-Inspired Adaptive Mutation Control**
3. **LLM-Guided Targeted Mutation**
4. **Agra** – Structural refinement
5. **Rota** – Multi-scale contextual aggregation
6. **Petra** – Adaptive feature fusion
7. **Pareto-Based Multi-Objective Architecture Selection**
