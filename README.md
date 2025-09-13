# ArchiveDiff-LLM: Detecting Important Changes in Archived Web Content
Pipeline for detecting and assessing important changes in archived web content. Implements ArchiveDiff-LLM from our JCDL 2025 paper, combining memento retrieval, sentence-level alignment, and LLM-based importance classification.

This repository contains the code and data accompanying the paper:

> **ArchiveDiff-LLM: Automating the Detection and Assessment of Important Changes in Archived Web Content**   
> Submitted to JCDL 2025.

---

## Overview

ArchiveDiff-LLM is a pipeline that leverages Large Language Models (LLMs) to automatically detect and assess important changes across archived versions (*mementos*) of web pages.  
Unlike traditional diff approaches, which operate at the character or token level, ArchiveDiff-LLM integrates:

- **Extraction of text in mementos from the Wayback Machine**
- **LLM-based classification of importance of content changes**

This repository provides:
- Dataset of archived news articles (CDX and extracted text)
- Code for memento retrieval, LLM analysis, metrics, and creation of charts
- Outputs of the experiment
- An interactive HTML viewer for exploring results
  
---

## Repository Structure
Results/ # Plots and aggregated evaluation results
analysis_all_output/ # LLM outputs of memento comparisons
cdx_files/ # CDX index files from Wayback Machine
dataset/ # Extracted mementos of news articles
figures/ # Figures and charts used in the paper
utils/ # Helper scripts and modules
viewer/ # HTML viewer for interactive exploration

analysis_all_LLM.py # Script to run LLM-based analysis
compute_metrics.py # Script to compute evaluation metrics
count_mementos.py # Utility to count mementos per article
create_charts.py # Script to generate plots
memento_retriever.py # Script to fetch mementos via CDX API
requirements.txt # Python dependencies




