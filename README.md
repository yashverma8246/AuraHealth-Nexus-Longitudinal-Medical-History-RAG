# AuraHealth Nexus — Longitudinal Medical History RAG

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![FAISS](https://img.shields.io/badge/VectorStore-FAISS--CPU-00599C)](https://github.com/facebookresearch/faiss)
[![Embeddings](https://img.shields.io/badge/Embeddings-all--MiniLM--L6--v2-orange)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
[![LLM Engine](https://img.shields.io/badge/LLM-Groq%20Cloud-F55036)](https://groq.com/)
[![WCAG](https://img.shields.io/badge/Accessibility-WCAG%20AA%20Compliant-15803D)](https://www.w3.org/WAI/standards-guidelines/wcag/)
[![License: MIT](https://img.shields.io/badge/License-MIT-0F766E.svg)](LICENSE)
[![Educational Use](https://img.shields.io/badge/Free%20for-Educational%20Use-blue.svg)](#educational-purpose-free-to-use-certificate)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-0F766E)](#)

> **Clinical Disclaimer & Research Notice**: The data used in this project represents a fully synthetic, fictional patient (`SYN-PAT-001`, Arjun Mehta) spanning 15 annual records (2010–2024). This project is strictly engineered for artificial intelligence, retrieval-augmented generation (RAG), and healthcare informatics research. It must **not** be used for actual clinical diagnosis, triage, or medical treatment decisions.

---

## Table of Contents

- [Overview](#overview)
- [Why Longitudinal RAG?](#why-longitudinal-rag)
- [Key Architectural Features](#key-architectural-features)
- [System Architecture & Dataflow](#system-architecture--dataflow)
- [Project Directory Structure](#project-directory-structure)
- [Step-by-Step Setup on the New System (Compulsory Guide)](#step-by-step-setup-on-the-new-system-compulsory-guide)
  - [Prerequisites](#prerequisites)
  - [Step 1: Clone or Copy the Repository](#step-1-clone-or-copy-the-repository)
  - [Step 2: Create a Clean Virtual Environment](#step-2-create-a-clean-virtual-environment)
  - [Step 3: Activate the Virtual Environment](#step-3-activate-the-virtual-environment)
  - [Step 4: Upgrade Pip & Install Dependencies](#step-4-upgrade-pip--install-dependencies)
  - [Step 5: Configure Environment Variables (.env)](#step-5-configure-environment-variables-env)
  - [Step 6: Build the Patient FAISS Vector Store](#step-6-build-the-patient-faiss-vector-store)
  - [Step 7: Launch the AuraHealth Nexus Dashboard](#step-7-launch-the-aurahealth-nexus-dashboard)
  - [Step 8: Run the Evaluation Benchmark Suite](#step-8-run-the-evaluation-benchmark-suite)
- [Troubleshooting & Common Gotchas](#troubleshooting--common-gotchas)
- [System-to-System Transfer Guide](#system-to-system-transfer-guide)
- [Clinical Minimalism UI & Accessibility](#clinical-minimalism-ui--accessibility)
- [Longitudinal Query Routing Engine](#longitudinal-query-routing-engine)
- [Synthetic Patient Timeline (SYN-PAT-001)](#synthetic-patient-timeline-syn-pat-001)
- [Example Queries](#example-queries)
- [Evaluation Benchmark Results](#evaluation-benchmark-results)
- [Multi-Patient Extensibility](#multi-patient-extensibility)
- [License & Educational Use Certificate](#license--educational-use-certificate)
  - [MIT Open Source License](#mit-open-source-license)
  - [Educational Purpose & Free-to-Use Certificate](#educational-purpose--free-to-use-certificate)
  - [Author & Acknowledgments](#author--acknowledgments)

---

## Overview

**AuraHealth Nexus** is an advanced, domain-specialized Retrieval-Augmented Generation (RAG) platform purpose-built for **multi-year longitudinal medical record reasoning**. Traditional RAG frameworks treat retrieved document chunks as isolated semantic fragments, frequently suffering from temporal hallucinations, chronological inversions, and retrospective bias. 

AuraHealth Nexus addresses these fundamental limitations by integrating **temporal intent detection**, **clinical section-aware chunking**, **chronological context sequencing**, and **grounded year-by-year citation synthesis**. It enables physicians, clinical researchers, and data scientists to ask complex temporal questions about long-term lab trajectories, drug titration timelines, and disease progression patterns across a patient's complete medical history.

---

## Why Longitudinal RAG?

Standard RAG architectures fail in clinical record tracking due to four structural problems:

| Challenge | Traditional Semantic RAG | AuraHealth Nexus Longitudinal RAG |
| :--- | :--- | :--- |
| **Temporal Context** | Retrieves top-$k$ chunks based purely on similarity; mixes 2012 and 2024 records indiscriminately. | **Chronological Sequencing**: Analyzes timeline intent and sorts retrieved clinical evidence from earliest year to latest year (2010 $\to$ 2024). |
| **Lab Trajectories** | Hallucinates or confuses baseline labs with current labs (e.g., confusing 2010 HbA1c 5.6% with 2024 HbA1c 7.1%). | **Delta Awareness**: Explicitly formats timeline trajectories and measures rate of change over multi-year intervals. |
| **Medication Titration** | Struggles to identify when a drug was initiated vs. increased vs. discontinued. | **Event Anchoring**: Distinguishes between first-documented initiation (e.g., Amlodipine in 2016) and subsequent additions (e.g., Losartan in 2020). |
| **Hallucination Prevention** | Summarizes missing years as if they were normal. | **Strict Grounding Rules**: Explicitly declares when a condition or lab is undocumented for specific timeline intervals. |

---

## Key Architectural Features

- **Clinical Section-Aware Chunking**: Intelligently segments yearly medical records while preserving section headers (Subjective, Objective, Assessment, Plan, Medications, Labs, Vital Signs) and patient metadata.
- **Dynamic Temporal Intent Routing**: The pipeline analyzes clinical queries using regex and semantic pattern matching to categorize requests into:
  - `TIMELINE_SUMMARY`: Broad multi-year overviews across all documented visits.
  - `LONGITUDINAL_TREND`: Trajectory analysis across labs, vitals, or progressive conditions.
  - `SPECIFIC_YEAR`: Pinpointed extraction focused on an exact clinical visit.
  - `STANDARD_SEMANTIC`: General semantic query matching.
- **Chronological Sorting Guarantee**: Regardless of vector similarity score rank, all retrieved chunks are sequenced chronologically before being passed to the LLM generator, eliminating chronological confusion.
- **Isolated Patient Namespaces**: Vector stores and index directories are segregated by patient identifier (`vector_index/<PATIENT_ID>/`), preventing data leakage across patient cohorts.
- **Clinical Minimalism Light Theme**: Features a high-contrast clinical interface styled with accessible teal tones (`#0F766E`), compliant with WCAG AA/AAA standards for readability, focus states, and disabled button behaviors.
- **Sub-Second Groq LLM Generation**: Powered by Groq's high-throughput LPU inference engine, delivering sub-1.5 second responses with inline year citations (e.g., `(2019 record)`).

---

## System Architecture & Dataflow

```
 ┌────────────────────────────────────────────────────────────────────────┐
 │       SYN-PAT-001_15yr_RAG_medical_history/ (2010 through 2024)        │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼  loader.py
 ┌────────────────────────────────────────────────────────────────────────┐
 │ Document Ingestion & Metadata Tagging:                                 │
 │ { patient_id: "SYN-PAT-001", year: 2019, source: "2019_SYN-PAT-001.txt"}│
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼  chunker.py
 ┌────────────────────────────────────────────────────────────────────────┐
 │ Clinical Section-Aware Chunking:                                       │
 │ Preserves headings, clinical sections, and propagates year metadata   │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼  vectorstore.py
 ┌────────────────────────────────────────────────────────────────────────┐
 │ Embedding Model: sentence-transformers/all-MiniLM-L6-v2 (384-dim)      │
 │ Vector Store: FAISS IndexFlatIP (Saved in vector_index/SYN-PAT-001/)   │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
                     User Clinical Query (via Streamlit)
                                     │
                                     ▼  rag_pipeline.py
 ┌────────────────────────────────────────────────────────────────────────┐
 │ 1. Temporal Query Analysis:                                            │
 │    Extracts years, classifies strategy: TIMELINE, TREND, SPECIFIC_YEAR│
 │ 2. Multi-Year Stratified Semantic Retrieval (top_k=6 to 12)            │
 │ 3. Chronological Sequencing: Sorts chunks ascending (2010 ───> 2024)   │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼  generator.py
 ┌────────────────────────────────────────────────────────────────────────┐
 │ Clinical Prompt Formatting & Strict Grounding Rules                    │
 │ LLM: Groq LPU (Llama-3.1-8b-instant / openai/gpt-oss-20b)              │
 │ Inlined Year Citations & Undocumented Findings Warnings                │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │ Streamlit Clinical Minimalism Dashboard (app.py)                       │
 │ - Real-Time Temporal Badges  - Latency & Strategy Metrics              │
 │ - Expandable Source Evidence - Interactive Chronological Explorations   │
 └────────────────────────────────────────────────────────────────────────┘
```

---

## Project Directory Structure

```
aiml-crash-yash-verma-Aurahealth_final_project/
├── SYN-PAT-001_15yr_RAG_medical_history/    # 15 annual synthetic clinical records
│   ├── 2010_SYN-PAT-001.txt                 # Baseline visit record
│   ├── 2011_SYN-PAT-001.txt                 # Annual follow-up
│   ├── ...                                  # Intermediate records (2012–2023)
│   ├── 2024_SYN-PAT-001.txt                 # Most recent follow-up
│   └── README_RAG_DATASET.txt               # Clinical dataset documentation
├── vector_index/                            # Isolated FAISS vector store directory
│   └── SYN-PAT-001/                         # Patient-specific vector index
│       ├── index.faiss                      # FAISS flat inner-product vector index
│       └── chunks.pkl                       # Serialized chunk metadata records
├── app.py                                   # Streamlit dashboard (Clinical Minimalism UI)
├── rag_pipeline.py                          # Core RAG orchestration & temporal routing
├── vectorstore.py                           # FAISS index wrapper & embedding manager
├── chunker.py                               # Clinical section-aware document chunker
├── loader.py                                # File parser & metadata extractor
├── generator.py                             # Groq API client & clinical grounding prompt
├── evaluate.py                              # RAG benchmark evaluation test suite
├── evaluation_results.json                  # Pre-computed benchmark outputs & scores
├── styles.css                               # Light-theme Clinical Minimalism CSS tokens
├── requirements.txt                         # Python dependencies
├── .env.example                             # Environment variable template
├── .gitignore                               # Git exclusions (protects secrets & binaries)
└── README.md                                # Comprehensive system documentation
```

---

## Step-by-Step Setup on the New System (Compulsory Guide)

Follow these exact steps to set up, configure, and execute AuraHealth Nexus on any new computer (Windows, macOS, or Linux).

### Prerequisites

- **Operating System**: Windows 10/11 (64-bit), macOS 12+ (Apple Silicon or Intel), or Linux (Ubuntu 20.04+).
- **Python Version**: **Python 3.10** or **Python 3.11** is **strongly recommended**.
  > [!IMPORTANT]
  > Avoid pre-release or experimental Python versions (such as Python 3.14 alpha/beta). Cutting-edge Python builds lack pre-compiled binary wheels for PyTorch, Sentence-Transformers, and FAISS, causing install failures.
- **Git**: Installed and configured on your system.
- **Groq API Key**: Free API key from [console.groq.com](https://console.groq.com/).

---

### Step 1: Clone or Copy the Repository

If cloning from GitHub:
```bash
git clone https://github.com/your-username/aiml-crash-yash-verma-Aurahealth_final_project.git
cd aiml-crash-yash-verma-Aurahealth_final_project
```

Or open your terminal in the extracted project folder:
```bash
cd path/to/aiml-crash-yash-verma-Aurahealth_final_project
```

---

### Step 2: Create a Clean Virtual Environment

Always create an isolated virtual environment to prevent dependency conflicts with your global Python installation.

- **Windows (PowerShell or Command Prompt)**:
  ```bash
  python -m venv .venv
  ```
  *(If you have multiple Python versions installed, specify Python 3.10 or 3.11 explicitly: `py -3.11 -m venv .venv`)*

- **macOS / Linux**:
  ```bash
  python3 -m venv .venv
  ```

---

### Step 3: Activate the Virtual Environment

- **Windows (PowerShell)**:
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
  *(If PowerShell displays an execution policy error, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`, then re-run the activation script).*

- **Windows (Command Prompt / CMD)**:
  ```cmd
  .\.venv\Scripts\activate.bat
  ```

- **macOS / Linux (Bash / Zsh)**:
  ```bash
  source .venv/bin/activate
  ```

Your terminal prompt should now display `(.venv)` at the beginning of the line.

---

### Step 4: Upgrade Pip & Install Dependencies

With the virtual environment active, install the required packages:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

#### What `requirements.txt` Installs:
- `streamlit`: Web dashboard and interactive visualization interface.
- `faiss-cpu`: High-performance similarity search and dense vector indexing.
- `sentence-transformers`: Local text embedding generation (`all-MiniLM-L6-v2`).
- `requests`: HTTP communication with the Groq API.
- `python-dotenv`: Automatic loading of `.env` configuration files.
- `numpy`: Numerical operations on embedding vectors.
- `pypdf` & `python-docx`: Document loaders for clinical PDF/Word attachments.

---

### Step 5: Configure Environment Variables (.env)

AuraHealth Nexus uses Groq's high-speed inference engine for clinical reasoning.

1. Copy the provided `.env.example` template to `.env`:
   - **Windows (PowerShell)**:
     ```powershell
     Copy-Item .env.example .env
     ```
   - **macOS / Linux / Git Bash**:
     ```bash
     cp .env.example .env
     ```

2. Open `.env` in any text editor and supply your Groq API key:
   ```env
   # AuraHealth Nexus Configuration
   GROQ_API_KEY=gsk_yourActualGroqApiKeyHere
   GROQ_MODEL=llama-3.1-8b-instant
   GROQ_API_BASE_URL=https://api.groq.com/openai/v1
   ```
   *(A free API key can be obtained in 30 seconds at [console.groq.com](https://console.groq.com)).*

---

### Step 6: Build the Patient FAISS Vector Store

Run the following one-line command to parse all 15 patient records, generate 384-dimensional dense embeddings, and save the indexed vector store:

```bash
python -c "from rag_pipeline import RAGPipeline; p = RAGPipeline(); p.index(force_rebuild=True)"
```

**Expected Terminal Output**:
```
INFO - Building clean vector index from patient records in '.../SYN-PAT-001_15yr_RAG_medical_history'...
INFO - Loaded 16 document files from ...
INFO - Generated 48 chunks with longitudinal clinical section tags.
INFO - Building FAISS IndexFlatIP with 48 embeddings...
INFO - Successfully indexed 48 chunks and saved to '.../vector_index/SYN-PAT-001'.
```

---

### Step 7: Launch the AuraHealth Nexus Dashboard

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will start, and Streamlit will automatically open your default browser to:
```
Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

Once loaded, you can ask custom clinical questions or click any of the pre-configured sample queries to explore Arjun Mehta's 15-year medical progression!

---

### Step 8: Run the Evaluation Benchmark Suite

To evaluate the pipeline's retrieval precision, confidence scores, and generation latency across 10 golden clinical test cases:

```bash
python evaluate.py
```

Results will be evaluated live and stored in [evaluation_results.json](file:///c:/Users/yashv/Downloads/aiml-crash-yash-verma-Aurahealth_final_project/aiml-crash-yash-verma-Aurahealth_final_project/evaluation_results.json).

---

## Troubleshooting & Common Gotchas

### 1. `ModuleNotFoundError: No module named 'torchvision'`
- **Cause**: Occurs if using Python 3.14 (pre-release) or if a sub-dependency of `sentence-transformers` attempts to import optional vision modules.
- **Solution**: 
  1. Ensure you are using **Python 3.10** or **Python 3.11**.
  2. If using Python 3.12+, run:
     ```bash
     pip install torch --index-url https://download.pytorch.org/whl/cpu
     ```

### 2. PowerShell Script Execution Policy Error (`Activate.ps1 cannot be loaded`)
- **Cause**: Windows PowerShell security policy restricts unsigned script execution.
- **Solution**: Run this command once in PowerShell:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```
  Then reactivate: `.\.venv\Scripts\Activate.ps1`.

### 3. Groq API Key Authentication Error (`401 Unauthorized`)
- **Cause**: Missing or invalid `GROQ_API_KEY` in your `.env` file.
- **Solution**: Verify that `.env` is located in the root project folder (same folder as `app.py`) and does not have double quotes or trailing whitespace.

### 4. Vector Store Corrupted or Out-of-Sync
- **Solution**: Delete the `vector_index/` directory or click **"🔄 Force Rebuild Patient Index"** in the Streamlit sidebar.

---

## System-to-System Transfer Guide

When moving this project to another laptop, PC, USB drive, or GitHub repository, follow these best practices:

| Item | Action Before Transfer | Why |
| :--- | :--- | :--- |
| `.venv/` | **DELETE** | Virtual environments contain hardcoded binary paths specific to one machine. Never copy `.venv`. Recreate it on the target system using Step 2 & 3. |
| `__pycache__/` | **DELETE** | Cached `.pyc` bytecode files are machine-dependent. |
| `vector_index/` | **OPTIONAL TO DELETE** | Rebuilds automatically in under 5 seconds with Step 6. |
| `.env` | **NEVER COMMIT TO GIT** | Contains private API keys. Use `.env.example` as a template and create a new `.env` on the new machine. |
| `SYN-PAT-001_15yr_RAG_medical_history/` | **ALWAYS KEEP** | Contains the 15 synthetic annual records required for indexing. |

---

## Clinical Minimalism UI & Accessibility

AuraHealth Nexus features the **Clinical Minimalism** design system—a light, modern, clinical interface engineered for high readability, visual clarity, and accessibility compliance.

### Design Tokens & Visual Hierarchy
- **Primary Clinical Teal**: `#0F766E` (Active buttons, headings, accents)
- **Deep Slate Text**: `#172B2A` (Primary typography; 14.5:1 contrast against white)
- **Muted Clinical Slate**: `#475569` and `#64748B` (Secondary labels, metadata)
- **Canvas Background**: `#F7F9F8` (Off-white clinical backdrop preventing eye fatigue)
- **Card Surface**: `#FFFFFF` with `#DDE7E3` subtle borders and delicate drop shadows

### Accessibility (WCAG 2.1 AA/AAA)
- **Buttons**: White text (`#FFFFFF`) on enabled teal backgrounds (`#0F766E`), yielding a **5.0:1** contrast ratio (exceeds WCAG AA 4.5:1 requirement).
- **Disabled State**: `#E2E8F0` surface with `#64748B` text and `#CBD5E1` border for unmistakable affordance.
- **Focus Rings**: `outline: 2px solid #0F766E` with 2px offset for keyboard navigation.
- **Responsive Layout**: Fluid CSS Grid and Flexbox structures supporting desktop, tablet, and mobile viewing.

---

## Longitudinal Query Routing Engine

The pipeline classifies clinical questions to apply specialized multi-year retrieval heuristics:

| Query Type | Strategy Badge | Heuristic Applied |
| :--- | :--- | :--- |
| **Chronological Timeline** | `TIMELINE_SUMMARY` | Expands retrieval budget ($k=12$), samples chunks across all 15 recorded years, and sorts ascending. |
| **Multi-Year Trend** | `LONGITUDINAL_TREND` | Focuses on comparative delta trajectories (e.g., baseline vs. recent values) across matching lab/vitals sections. |
| **Specific Visit Year** | `SPECIFIC_YEAR` | Filters candidates with high priority for the target year (e.g., `year == 2016`) while providing surrounding context. |
| **Semantic Query** | `STANDARD_SEMANTIC` | Dense inner-product vector similarity retrieval over all patient clinical chunks. |

---

## Synthetic Patient Timeline (SYN-PAT-001)

The bundled dataset covers a synthetic 15-year clinical trajectory for **Arjun Mehta** (DOB: 1968):

```
 2010 ─── Baseline wellness exam. Normal vitals. eGFR 92, HbA1c 5.4%, BP 122/78.
  │
 2013 ─── First documentation of elevated fasting blood glucose (104 mg/dL).
  │
 2015 ─── Borderline elevated blood pressure (138/86 mmHg). Lifestyle modifications advised.
  │
 2016 ─── Essential Hypertension diagnosed. Amlodipine 5 mg daily initiated (First medication).
  │
 2018 ─── Progression to Impaired Glucose Tolerance / Prediabetes (HbA1c 6.3%).
  │
 2019 ─── Formal Diagnosis: Type 2 Diabetes Mellitus. Metformin 500 mg BID initiated.
  │
 2020 ─── Diabetic Nephropathy screening: Microalbuminuria detected. Losartan 25 mg added.
  │
 2022 ─── Moderate renal decline observed: eGFR decreases to 64 mL/min/1.73m², Creatinine 1.28 mg/dL.
  │
 2024 ─── Current Regimen: Metformin 1000 mg BID, Amlodipine 10 mg, Losartan 50 mg, Atorvastatin 20 mg.
           eGFR stable at 61 mL/min/1.73m², HbA1c 7.1%.
```

---

## Example Queries

Test these sample queries in the dashboard:

1. **Longitudinal Lab Trajectory**:
   > *"How did the patient's kidney function (eGFR and Creatinine) change from baseline in 2010 to 2024?"*
2. **Medication Addition & Titration**:
   > *"When was Losartan added and what medications were documented in 2024?"*
3. **Disease Progression Milestone**:
   > *"When was type 2 diabetes first documented and what was the progression from prediabetes?"*
4. **Metabolic Health Evolution**:
   > *"What were the major metabolic changes between 2010 and 2024?"*
5. **Complete Longitudinal Summary**:
   > *"Give a chronological summary of the patient's major medical events from 2010 to 2024."*

---

## Evaluation Benchmark Results

The pipeline was benchmarked using [evaluate.py](file:///c:/Users/yashv/Downloads/aiml-crash-yash-verma-Aurahealth_final_project/aiml-crash-yash-verma-Aurahealth_final_project/evaluate.py) across 10 representative clinical questions:

| Metric | Benchmark Result | Target Standard |
| :--- | :--- | :--- |
| **Retrieval Precision** | **100%** relevant years retrieved | $\ge 90\%$ |
| **Chronological Ordering** | **100%** ascending sequence | $100\%$ |
| **Average Generation Latency** | **1.14 seconds** (Groq LPU) | $< 2.50\text{s}$ |
| **Clinical Grounding Rate** | **100%** with explicit year citations | $\ge 95\%$ |
| **Temporal Hallucination Rate** | **0.0%** across test suite | $< 2.0\%$ |

Full benchmark runs and query logs are preserved in [evaluation_results.json](file:///c:/Users/yashv/Downloads/aiml-crash-yash-verma-Aurahealth_final_project/aiml-crash-yash-verma-Aurahealth_final_project/evaluation_results.json).

---

## Multi-Patient Extensibility

Although configured out-of-the-box for `SYN-PAT-001`, the underlying architecture supports multi-patient expansion:
1. **Patient Metadata Scoping**: Every document and chunk retains `patient_id`.
2. **Index Isolation**: Separate vector index folders can be instantiated (`vector_index/<PATIENT_ID>/`).
3. **Dynamic Cohort Filtering**: `VectorStore.search(query, patient_id="SYN-PAT-002")` filters vector similarities to target patient records.
4. **Interactive Ingestion**: The sidebar file uploader allows dragging and dropping new clinical records (TXT, PDF, DOCX) to index additional years or patients in real time.

---

## License & Educational Use Certificate

### MIT Open Source License

This project is open-source software licensed under the **[MIT License](LICENSE)**.

```text
MIT License

Copyright (c) 2024-2026 Yash Verma

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

### Educational Purpose & Free-to-Use Certificate

```
┌────────────────────────────────────────────────────────────────────────┐
│             CERTIFICATE OF FREE EDUCATIONAL & RESEARCH USE             │
├────────────────────────────────────────────────────────────────────────┤
│ Project:    AuraHealth Nexus — Longitudinal Medical History RAG        │
│ Author:     Yash Verma                                                 │
│ License:    MIT Open Source License (Permissive & Free to Use)         │
│ Status:     100% Free & Open Access for All Educational Purposes      │
└────────────────────────────────────────────────────────────────────────┘
```

> **Grant of Educational Freedom**:  
> Permission is hereby granted, unconditionally and free of charge, to students, educators, researchers, clinicians, and developers worldwide to use, study, copy, modify, merge, publish, distribute, sublicense, and/or teach with this project and all associated code, prompts, synthetic clinical records, and documentation for **any educational, academic, learning, personal, or research-driven endeavor**.

#### Permitted & Encouraged Educational Uses:
- 🎓 **Academic Study & Coursework**: Use freely in university computer science, biomedical engineering, data science, and AI/ML courses.
- 🔬 **Research & Benchmarking**: Adapt the longitudinal chunking and chronological retrieval algorithms for research papers, thesis projects, and scientific benchmarking.
- 🏆 **Hackathons & Competitions**: Fork, re-architect, or integrate this repository as a starting boilerplate or reference solution.
- 👨‍🏫 **Instructional Demonstration**: Demonstrate clinical Retrieval-Augmented Generation (RAG), FAISS vector stores, Groq LLM inference, and WCAG-compliant Streamlit design in workshops and classrooms.
- 💻 **Self-Paced Learning**: Inspect, experiment, break, and modify the code locally without licensing fees, subscriptions, or restrictive covenants.

#### Clinical & Research Safe Harbor Notice:
- All patient timelines, clinical lab values, medications, and progress notes contained within this repository (`SYN-PAT-001`, Arjun Mehta) are **100% synthetic and fictional**.
- This platform is engineered strictly for computational research, educational demonstrations, and informatics exploration. It is **not** a certified medical diagnostic device and must not be used for direct patient care, clinical triage, or treatment decisions.

---

### Author & Acknowledgments

- **Lead Developer**: Yash Verma
- **Curriculum**: AIML Crash — AuraHealth Capstone Project
- **Core Technologies**: Python, Streamlit, FAISS CPU, Sentence-Transformers, and Groq Cloud LPU

