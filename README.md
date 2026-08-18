# Context-Aware LLM-Guided Neural Architecture Search for Medical Image Segmentation

## Overview

Medical image segmentation remains a fundamental yet challenging task due to heterogeneous anatomical structures, low-contrast boundaries, domain variability, and the difficulty of designing lightweight yet generalisable segmentation architectures. Although recent deep learning and neural architecture search (NAS) approaches have achieved promising performance, many existing methods still depend on manually designed feature interaction strategies, fixed optimisation behaviour, and computationally expensive architectures that limit scalability and cross-domain adaptability.

To address these challenges, we propose **ELPS-Net**, a reliability-aware evolutionary neural architecture search framework that integrates **Particle Swarm Optimisation (PSO)** and **Large Language Models (LLMs)** for adaptive architecture evolution in medical image segmentation. The proposed framework automatically discovers compact and high-performing segmentation architectures while balancing segmentation accuracy, computational efficiency, and reliability-aware generalisation.

ELPS-Net incorporates three specialised architectural modules:

* **Agra** – Structural representation preservation for enhanced anatomical boundary retention.
* **Rota** – Lightweight multi-scale contextual aggregation for effective long-range feature modelling.
* **Petra** – Reliability-aware feature reconstruction for stable decoder feature refinement.

Furthermore, ELPS-Net introduces a novel **PSO-LLM fusion mutation strategy**, where PSO dynamically optimises mutation behaviour while an LLM provides targeted architecture-level mutation guidance. This enables adaptive exploration and exploitation during evolutionary search, resulting in more efficient discovery of compact and accurate segmentation architectures.

To the best of our knowledge, ELPS-Net is the first reliability-aware medical image segmentation NAS framework that combines **PSO-based adaptive mutation optimisation** with **LLM-guided targeted architectural mutation** within a unified evolutionary search process.

## Key Features

* Reliability-aware evolutionary neural architecture search.
* PSO-guided adaptive mutation rate optimisation.
* LLM-guided targeted architectural mutation.
* Lightweight segmentation architecture discovery.
* Multi-objective optimisation of accuracy, efficiency, and reliability.
* Strong cross-domain generalisation across heterogeneous medical imaging datasets.

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

The discovered architectures were further evaluated on:

* MMWHS
* CHAOS
* BraTS 2020

demonstrating strong cross-domain robustness and reliability while maintaining computational efficiency.

## Framework

ELPS-Net consists of:

1. Evolutionary Neural Architecture Search (ENAS)
2. PSO-Based Adaptive Mutation Optimisation
3. LLM-Guided Targeted Mutation Strategy
4. Agra Module
5. Rota Module
6. Petra Module
7. Multi-Objective Reliability-Aware Architecture Selection

