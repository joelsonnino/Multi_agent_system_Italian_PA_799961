# Multi_agent_system_Italian_PA

**Team Members:**
- **Joel Sonnino** (joel.sonnino@studenti.luiss.it)
- **Luca Parrella** (luca.parrella@studenti.luiss.it)

---

## Section 1: Introduction
To meet the growing demand for faster & transparent public services while managing increasing volumes of data, we developed a local multi-agent system designed to address user's query about analysis & and visualization on data from the Italian public administration portal (NoiPa). 
Leveraging a tri-agent architecture: **Orchestrator**, **Analysis**, and **Visualization**. The system ensures data privacy by hosting all the system locally via Ollama.
This project demonstrates how autonomous agents can automate end-to-end data workflows: from interpreting user queries, through data transformation and analysis, to dynamic presentation in a Streamlit interface.

**Key Objectives:**
1. **Data Analysis & Visualization:** Ability to process, analyze and visualize data of the portal NoiPa CSVs to receive insights.
2. **Transparency & Compliance:** Keep all data locally with no third-party sharing, ensuring GDPR compliance.
3. **Interactive Exploration:** Provide a user-friendly interface to explore insights without writing code.

---
<br /><br />
## Section 2: 
## 2.1 Data cleaning and processing

- **Normalizes Schema**: Translated all column headers into English, creating a consistent naming convention for LLM parsing.
- **Validates & Casts Types**: Checked numeric fields, corrected malformed entries, and enforced proper data types.
- **Generates Features**: Created group features such as: age groups: 18-24, 25-34 e.g or income bracket ranges
- **Standardizes Categories**: Converted uppercase category labels to lowercase for uniformity and downstream matching.

## 2.2: Brainstorming
In the initial phase we had a deep brainstorming session where we thought about different design patterns, communication flows, and integration strategies between agents. 
We carefully analyzed the roles and responsibilities that each agent should take on, and we decided to structure the system as follows:

## 2.3: System Structure
The system’s architecture begins with a Streamlit interface that captures the user’s natural-language request.

![System Architecture](System%20architecture.png)

This request is passed to the Orchestrator Agent, which selects the relevant dataset(s) and routes the task to either the Analysis Agent or the Visualization Agent. Here are the details:

1. **Orchestrator Agent**
   - **Role:** Interfaces with the user, parses natural-language requests, identifies relevant datasets, and route tasks to the right agent.
   - **Implementation:** Built using LangChain with custom prompt templates. Uses Ollama for local LLM inference to maintain data privacy.
   - **Responsibilities:** Dataset selection, query routing, context management.

2. **Analysis Agent**
   - **Role:**
        - Receive a natural-language query along with a list of relevant datasets and their schemas.
        - Apply any necessary data transformations, merging only when requested, grouping, aggregations, and special-case logic.
        - Generate clean, executable Python code with no comments or narrative, ensuring syntax correctness and using only the approved libraries and column names.

3. **Visualization Agent**
   - **Role:**
        - Receives a natural-language chart request, the list of relevant CSV filenames, and their column schemas.
        - Apply any necessary data transformations, merging only when requested, grouping, aggregations, and special-case logic
        - Chart Generation: Emits clean, comment-free Python code that builds exactly the requested chart type (bar, line, pie, etc.) with titles & labels.
   
<br /><br />

## Section 3: Implementation Plan

## 3.1: Choice of the LLM
- We Evaluated 3 different LLMs: Mistral, QWEN, and LLaMA, ultimately selecting llama3.2.
- We found that llama 3.2 was the best performer, by testing the three different models on a list of 10 basic queries, to understand which models was performing better.

## 3.2 Tool Orchestration
- Leveraged LangChain tools to integrate and manage our orchestrator, analysis, and visualization agents.

## 3.3 Prompt Engineering
- Crafted and iterated custom prompts to guide each agent’s reasoning, code generation, and charting tasks.
- The prompt templates have a clear structure, starting with context, followed by the instructions and at the end the constraints

## 3.4 Streamlit Interface
- Created and interactive streamlit interface for an optimal user experience.

<br /><br />

## Section 4: Reproducibility Guidelines
- To ensure the reproducibility of our multi-agent system, we have made the entire setup process transparent and modular.
- Follow the steps below to replicate the system in your local environment.

## 4.1: Clone the Repository

```bash
git clone https://github.com/joelsonnino/Multi_agent_system_Italian_PA_799961.git
cd Multi_agent_system_Italian_PA_799961
```

## 4.2: Environment Setup

Create a virtual environment and install the required dependencies.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 4.3: Model Installation
- We use a locally hosted LLM: llama3.2 via [Ollama](https://ollama.com/). Ensure Ollama is installed and the appropriate model is pulled:

```bash
ollama pull llama3.2
```

## 4.4 IMPORTANT STEPS TO REPRODUCE THE PROJECT:
1) Clone the repository
2) Check to have in the project folder the raw_datasets files
3) Run the cleaning_and_EDA_def to obtain the cleaned datasets under the folder called "datasets"
4) Launch the Application

## Here below the instructions for launching the application

- Run the Streamlit interface to interact with the multi-agent system:

```bash
streamlit run streamlit_app.py
```

This will start a local server at `http://localhost:8501`, where you can interact with the system via a user-friendly interface.

<br /><br />

## DISCLAIMER: WHEN YOU TRY TO ASK SOME QUERIES AND YOU DON'T GET THE RIGHT RESULTS, CLEAR THE MEMORY AND TRY TO SEND AGAIN THE REQUEST 

<br /><br />

## Section 5: Evalutation
- **Purpose:** Quantify the correctness of the Analysis and Visualization Agent’s code-generated answers.
- **Procedure:** We have assembled an Excel workbook that includes 2 sheets:
     - A curated list of 21 questions for the **Analysis Agent**, designed to test dataset joins, filters, and aggregations across multiple CSV files.
     - A parallel set of 12 questions of visualization challenges for the **Visualization Agent**, focused on bar charts and pie charts.

- **Evaluation Metrics:**
  - **Accuracy (%)** = (Correct answers / Total answers) × 100

<br /><br />
## Section 6: Results
| Agent                   | Metric                | Score          |
|-------------------------|-----------------------|----------------|
| **Analysis Agent**      | Accuracy (%)          | 95 %           |
| **Visualization Agent** | Accuracy  (%)         | 83 %           |


<br /><br />

## Section 7: Security & Cost Advantages

- **Zero API Fees**: Host LLMs locally via Ollama with no per-token or per-call charges.
- **Predictable TCO**: One‐time hardware investment versus variable cloud expenses. There are no surprise bills or usage-based increases.
- **Reduced Attack Surface**: All informations remain within the secure network, no third-party sharing. Eliminating the risk of data-leaks or non-compliant data transfers.
- **Compliance & Auditability**: Full control over data lifecycle, encryption, and access policies. Comprehensive audit logs of every query, transformation, and user interaction, ensuring GDPR regulations are met.

<br /><br />

## Section 8: LIMITATION AND FUTURE STEP
- **Infrastructure Overhead**: Locally hosted LLM agents requires significant computational resources. Limited scalability on non-specialized hardware.
- **Performance Trade-Offs**: While local models like LLaMA and Mistral ensure control and privacy, they may underperform compared to cloud-based APIs (e.g., GPT-4) on complex tasks.
- **Hybrid Model Integration**: Implement smart routing that leverages local models for standard tasks and uses GPT-4 API selectively for complex queries.
<br /><br />

## Section 9: Conclusions

The adoption of multi-agent systems like the one we have presented is fundamental for modernizing Italian Public Administrations. Thanks to a modular architecture that integrates orchestration, analysis, and visualization, PAs will be able to:

- **Respond more quickly** to citizens’ requests by automating the entire workflow—from natural-language input to the generation of charts and reports.  
- **Ensure maximum transparency** and GDPR compliance by keeping all data and models on-premises, without sharing anything with third-party services.  
- **Scale sustainably** by leveraging dedicated hardware infrastructure and drastically reducing the variable costs associated with external APIs.  

We hope this project is only the first of many: our expertise in integrating local LLMs and data-analysis tools is ready to support new digital challenges within both central and local administrations.
