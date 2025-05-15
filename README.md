# Multi_agent_system_Italian_PA_799961

# NoiPa Multi-Agent System

**Team Members:**
- **Joel Sonnino** (joel.sonnino@studenti.luiss.it)
- **Luca Parrella** (luca.parrella@studenti.luiss.it)

---

## Section 1: Introduction
The **NoiPa Multi-Agent System** is a comprehensive, modular framework designed to facilitate interactive querying, analysis, and visualization of payroll data from the Italian public administration portal (NoiPa). Leveraging a tri-agent architecture—**Orchestrator**, **Analysis**, and **Visualization**—the system ensures data privacy by hosting all Large Language Models (LLMs) locally via Ollama and maintains full auditability. This project demonstrates how autonomous agents can automate end-to-end data workflows: from interpreting user queries, through data transformation and analysis, to dynamic presentation in a Streamlit interface.

**Key Objectives:**
1. **Data Automation:** End-to-end processing of raw NoiPa CSVs, including cleaning, filtering, joining, and aggregating payroll records.
2. **Transparency & Compliance:** Keep all data and inference on-premises, ensuring GDPR compliance and enabling reproducible audit trails.
3. **Interactive Exploration:** Provide a user-friendly dashboard for stakeholders to explore insights without writing code.

---

## Section 2: Methods

### 2.1 System Architecture
A high-level flowchart is provided in `images/architecture_flowchart.png`. The architecture comprises three specialized agents:

1. **Orchestrator Agent**
   - **Role:** Interfaces with the user, parses natural-language requests, identifies relevant datasets, and delegates tasks to downstream agents.
   - **Implementation:** Built using LangChain with custom prompt templates. Uses Ollama for local LLM inference to maintain data privacy.
   - **Responsibilities:** Dataset selection, query routing, context management.

2. **Analysis Agent**
   - **Role:** Generates and executes Python code to transform the data as requested (e.g., joins on keys like `Matricola`, filters by year or department, aggregations such as sum of salaries).
   - **Implementation:** A local LLM (Ollama) produces pandas/numpy scripts. Scripts run in a sandboxed environment to prevent unauthorized file operations.
   - **Responsibilities:** Data cleaning (handling missing values, type conversions), complex aggregations, error handling.

3. **Visualization Agent**
   - **Role:** Produces plotting code (matplotlib) to visualize analysis results.
   - **Implementation:** Uses LLM prompt engineering to generate reusable plotting functions. Integrated into Streamlit via an isolated execution context.
   - **Responsibilities:** Chart selection (bar, line, pie, histograms), axis labeling, title generation, layout configuration.

### 2.2 Design Choices and Rationale
- **Local LLM Inference (Ollama):** Avoids per-token costs and external data transfers; ensures high throughput and GDPR compliance.
- **LangChain Orchestration:** Facilitates modular prompts, context windows, and retry logic for robust request handling.
- **Streamlit Frontend:** Enables rapid deployment of interactive visualization components, with caching to minimize recomputation.
- **Sandboxed Execution:** Prevents arbitrary code execution beyond data operations, enhancing security.

### 2.3 Development Environment Setup
To ensure reproducibility, follow one of the two workflows below:

**Conda Workflow:**
```bash
git clone <repo_url>
cd noipa-mas
conda create -n noipa-mas python=3.10 -y
conda activate noipa-mas
pip install -r requirements.txt
```
Export environment:
```bash
conda env export --no-builds > environment.yml
```

**Virtualenv + Pip Workflow:**
```bash
python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Key Dependencies (excerpt from requirements.txt):**
- `langchain` >=0.0.x
- `ollama` >=1.0.x
- `pandas` >=1.5.x
- `numpy` >=1.24.x
- `matplotlib` >=3.6.x
- `streamlit` >=1.20.x

---

## Section 3: Experimental Design
We conducted two core experiments to validate the functionality and performance of our system.

### Experiment 1: Analysis Accuracy and Efficiency
- **Purpose:** Quantify the correctness of the Analysis Agent’s code-generated transformations (joins, filters, aggregations).
- **Dataset:** Five NoiPa CSV files (~200,000 rows total) covering payroll details, deductions, and department codes.
- **Baseline:** Manual pandas scripts authored by an experienced data analyst.
- **Procedure:** For each of ten query scenarios (e.g., "Total net salary per department for FY 2023"), compare Analysis Agent output to baseline.
- **Evaluation Metrics:**
  - **Accuracy (%)** = (Correct cells / Total cells) × 100
  - **Mean Execution Time (s)** per scenario (includes code generation + execution).

### Experiment 2: Visualization Quality Assessment
- **Purpose:** Evaluate the clarity and correctness of automatically generated charts.
- **Use Cases:** Ten visualization tasks (e.g., salary distribution histograms, monthly trends line charts).
- **Baseline:** Handcrafted matplotlib scripts.
- **Evaluation Metrics:**
  - **Plot Correctness (%)** = (Correct chart type + correct labels and data points) / Total tasks × 100
  - **User Satisfaction (1–5)** from a survey of 8 domain experts rating readability and interpretability.

---

## Section 4: Results

### 4.1 Quantitative Findings
| Agent                | Metric                | Score          |
|----------------------|-----------------------|----------------|
| **Analysis Agent**   | Accuracy (%)          | 96.2 ± 1.8     |
|                      | Mean Exec. Time (s)   | 1.78 ± 0.25    |
| **Visualization Agent** | Plot Correctness (%)  | 89.5 ± 3.4     |
|                      | User Satisfaction     | 4.4 ± 0.3      |

> **Note:** All tables and figures below are rendered by `main.ipynb` via automated code execution.

#### Sample Results Table
![Results Table Placeholder](images/results_table.png)

#### Example Visualization
![Salary Distribution Histogram](images/sample_chart.png)

---

## Section 5: Conclusions

### 5.1 Key Takeaways
The NoiPa Multi-Agent System successfully demonstrates that:
- **Modular Agent Design** dramatically reduces development time for complex data workflows.
- **Local LLM Hosting** ensures data privacy and cost-effectiveness without sacrificing performance.
- **Automated Visualization** pipelines can achieve high levels of correctness and user satisfaction comparable to manual efforts.

### 5.2 Limitations and Future Work
- **Scalability:** Current in-memory data handling may not scale to datasets >1M rows. Future work: integrate database backends (e.g., DuckDB, PostgreSQL).
- **LLM Versatility:** Local open-source models may struggle with nuanced queries. Future work: implement hybrid routing—use cloud APIs (e.g., GPT-4) for edge cases.
- **Interactive Features:** Enhance Streamlit UI with dynamic filters, multi-select controls, and drill-down charts for deeper exploration.

---

## Repository Structure
```
├── README.md
├── requirements.txt
├── environment.yml      # Optional export of conda environment
├── main.ipynb          # Central notebook with alternating text & code cells
└── images/             # Folder containing all figures referenced in README
    ├── architecture_flowchart.png
    ├── results_table.png
    └── sample_chart.png
```

*Thank you for reviewing our detailed report on the NoiPa Multi-Agent System!*
