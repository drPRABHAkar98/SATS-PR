# ELISA Synthetic Data Simulator 🧬📱

**A mobile-compatible, web-based computational tool for biochemical assay simulation and data pipeline stress-testing.**

## Overview
This tool acts as a "digital twin" for ELISA and other biochemical assays. It allows researchers to input target biological parameters (Mean, SD, *n*) and standard curve anchoring data to reverse-engineer and generate synthetic individual sample reads, dilution adjustments, and raw Optical Densities (ODs). 

## Tech Stack & Accessibility
* **Platform:** Web Browser (Mobile & Desktop Responsive)
* **Built On:** Firebase studio
* **Output:** Downloadable `.csv` structured for downstream automated analysis (LIMS, Python, R, SPSS).

## Key Features
* **Standard Curve Anchoring:** Enforces real-world physical assay limits (LOD/LOQ).
* **Reverse OD Calculation:** Simulates individual sample ODs that mathematically satisfy the target summary statistics.
* **Dilution Optimization:** Identifies samples exceeding spectrophotometer limits prior to wet-lab execution.

## Disclaimer & Ethical Use
**This tool is strictly designed for generating synthetic datasets for computational modeling, software testing, and statistical simulation.** It is explicitly NOT intended, nor should it ever be used, to fabricate, falsify, or reconstruct missing raw data for actual scientific reporting, regulatory submissions, or academic publication. 

---
*Developed by [Prabhakaran]*
