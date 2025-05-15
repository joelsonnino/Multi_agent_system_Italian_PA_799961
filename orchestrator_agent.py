
"""
Orchestrator Agent:
1. Selects the most relevant dataset(s) based on the user query and available dataset structures.
2. Routes requests to Analysis or Visualization Agent for the selected dataset(s).
Utilizes conversational memory.
"""
from langchain.llms import Ollama
from langchain import PromptTemplate, LLMChain
import logging
import traceback

logger = logging.getLogger(__name__)

llm = Ollama(model="llama3.2:latest", base_url="http://localhost:11434", timeout=120_000)


# --- 1. Dataset Selection Prompt Template ---
dataset_selector_template = """
CONTEXT
You are an expert data handler and you are in a multi agentic system based on the data of the Italian public administration.
You are given a user query, chat history, and a list of available datasets with their filenames and key columns.
Your task is to select the most relevant dataset(s) for the user query from a list of available datasets.

How you should approach  Tasks:
1. Analyze the request to understand well what's being asked
2. Break down complex problems into manageable steps and look at each section below to understand how to solve it
3. Use appropriate tools and methods to address each step
4. Deliver results in a clear and organized manner

INSTRUCTIONS:
- When ONLY one dataset is useful for the RESPONSE, list ONLY that one.
- When multiple datasets seem relevant to answer the query and ONLY if they could be merged trough a common key, list all of them. 

TASK:
- Your response MUST be a comma-separated list of filenames.
- Do NOT explain your choice. Only provide the filenames. If no dataset seems relevant, respond with "NONE".

[CHAT HISTORY]
{chat_history}

[AVAILABLE DATASETS SUMMARY]
(Filename: Columns List)
{available_datasets_summary}

[USER QUERY]
"{query}"

Selected Dataset Filenames (comma-separated, or NONE):
"""
dataset_selector_prompt = PromptTemplate(
    input_variables=["chat_history", "available_datasets_summary", "query"],
    template=dataset_selector_template
)

# --- 2. Task Routing Prompt Template ---
router_template = """
You are an expert orchestrator AI agent in a multi agentic system based on data on the italian public administration.
Your task is to ROUTE the user's request, which will be processed using a pre-selected set of dataset(s),
Route it to the analysis agent if the query is related to 'analysis', such as: numerical insights, statistics, data summaries, dataset infos,) 
Route it to the visualization agent if the query is related to'visualization', such as (charts, graphs, plots, histograms, pie charts, scatterlpot).

[INSTRUCTIONS]
- If the user start the query with List, Give me, Provide me, What is, How many you have to route it to the analysis agent
- If the user start the query with Show me, Plot me, Draw, Plot, Visualize you have to route it to the visualization agent

The user wants to answer the following query:
[USER QUERY]
"{query}"

The following dataset(s) have been pre-selected to answer this query:
[PRE-SELECTED DATASET FILENAMES]
{selected_filenames_str}

Their structures are:
[SCHEMAS FOR PRE-SELECTED DATASETS]
{selected_dataset_structures_str}

[CHAT HISTORY]
{chat_history}

Based on the [USER QUERY] and its intent (even if it implies operations on the [PRE-SELECTED DATASET FILENAMES]),
respond with exactly one lowercase word: "analysis" or "visualization". Do not add any explanation.

Decision:
"""
router_prompt = PromptTemplate(
    input_variables=["chat_history", "query", "selected_filenames_str", "selected_dataset_structures_str"],
    template=router_template
)

def create_available_datasets_summary(all_dataset_structures: dict) -> str:
    """Creates a string summary of available datasets and their columns."""
    summary_lines = []
    if not all_dataset_structures:
        return "No datasets available."
    for fname, columns in all_dataset_structures.items():
        col_str = ", ".join(columns) if columns else "No columns listed"
        summary_lines.append(f"- {fname}: ({col_str})")
    return "\n".join(summary_lines)

def orchestrate(query: str, all_dataset_structures: dict, memory):
    """
    Selects relevant dataset(s), then routes the query to the appropriate agent.
    Args:
        query: The user's natural language request.
        all_dataset_structures: Dict of structures for ALL available datasets.
                                  {"file1.csv": ["colA", "colB"], ...}
        memory: The shared ConversationBufferWindowMemory object.
    Returns:
        A dictionary: {"code": "python_code", "selected_files": ["filename.csv"], "agent_type": "analysis/visualization"}
                      or {"error": "error message"} on failure.
    """
    logger.info(f"Orchestrator: Received query='{query}'. Starting dataset selection.")
    logger.debug(f"Orchestrator: All dataset structures available: {list(all_dataset_structures.keys())}")

    # --- Stage 1: Dataset Selection ---
    dataset_selector_chain = LLMChain(
        llm=llm,
        prompt=dataset_selector_prompt,
        memory=memory # Pass memory object
    )
    available_summary = create_available_datasets_summary(all_dataset_structures)
    logger.debug(f"Orchestrator: Summary for dataset selector LLM:\n{available_summary}")

    try:
        selected_filenames_raw_str = dataset_selector_chain.run(
            query=query,
            available_datasets_summary=available_summary
        ).strip()
    except Exception as e:
        logger.error(f"Orchestrator: LLM call for dataset selection failed: {e}\n{traceback.format_exc()}")
        return {"error": f"Error during dataset selection: Could not get response from language model. {e}"}

    if not selected_filenames_raw_str or selected_filenames_raw_str.upper() == "NONE":
        logger.info("Orchestrator: Dataset selector returned no relevant files or 'NONE'.")
        return {"error": "The system could not identify any relevant datasets for your query. Please try rephrasing or ensure relevant data is available."}

    # Clean and validate selected filenames
    raw_filenames_list = [fn.strip() for fn in selected_filenames_raw_str.split(',') if fn.strip()]
    valid_selected_filenames = []
    unrecognized_filenames = []

    for fname_raw in raw_filenames_list:
        cleaned_fname = fname_raw.replace("`", "").replace("'", "").replace('"',"").strip()
        if not cleaned_fname: continue

        if cleaned_fname in all_dataset_structures:
            valid_selected_filenames.append(cleaned_fname)
        else:
            # Try a lenient match (e.g. case-insensitive)
            matched_key = None
            for key in all_dataset_structures.keys():
                if cleaned_fname.lower() == key.lower():
                    matched_key = key
                    break
            if matched_key:
                valid_selected_filenames.append(matched_key)
                logger.info(f"Orchestrator: Matched raw filename '{fname_raw}' to '{matched_key}' via lenient matching.")
            else:
                unrecognized_filenames.append(fname_raw)
    
    if unrecognized_filenames:
        logger.warning(f"Orchestrator: Dataset selector returned unrecognized/unavailable filenames: {unrecognized_filenames}. These will be ignored.")

    if not valid_selected_filenames:
        logger.error(f"Orchestrator: No valid datasets selected after cleaning. Original selection: '{selected_filenames_raw_str}'.")
        return {"error": f"Error: The system identified dataset(s) '{selected_filenames_raw_str}', but none are recognized or available after validation. Please check dataset names or rephrase your query."}

    logger.info(f"Orchestrator: Dataset selector identified valid files: {valid_selected_filenames}")

    # Prepare structures for only the selected valid files
    selected_dataset_structures = {
        fname: all_dataset_structures[fname] for fname in valid_selected_filenames if fname in all_dataset_structures and all_dataset_structures[fname] is not None
    }
    if len(selected_dataset_structures) != len(valid_selected_filenames):
         logger.warning("Orchestrator: Mismatch between valid selected filenames and found structures. Some structures might be missing.")
         # Potentially filter valid_selected_filenames again if a structure is truly None
         valid_selected_filenames = [fn for fn in valid_selected_filenames if fn in selected_dataset_structures]
         if not valid_selected_filenames:
              return {"error": "Error: Structures for all selected datasets are missing or invalid."}


    # --- Stage 2: Task Routing (Analysis vs. Visualization) ---
    router_chain = LLMChain(
        llm=llm,
        prompt=router_prompt,
        memory=memory
    )

    try:
        # chat_history is automatically pulled from memory by the chain
        decision = router_chain.run(
            query=query,
            selected_filenames_str=", ".join(valid_selected_filenames),
            selected_dataset_structures_str=str(selected_dataset_structures)
        ).strip().lower()
    except Exception as e:
        logger.error(f"Orchestrator: LLM call for task routing failed: {e}\n{traceback.format_exc()}")
        return {"error": f"Error during task routing for selected files: Could not get response from language model. {e}"}

    logger.info(f"Orchestrator: Routing decision for files {valid_selected_filenames}: {decision}")

    # Agent calling logic
    agent_module_name = "Unknown Agent"
    agent_function = None

    if decision == "visualization":
        try:
            from visualization_agent import generate_visualization
            agent_module_name = "Visualization Agent"
            agent_function = generate_visualization
        except ImportError:
            logger.error("Orchestrator: Failed to import visualization_agent.")
            return {"error": "Critical Error: Visualization agent module could not be loaded."}
    elif decision == "analysis":
        try:
            from analysis_agent import generate_analysis
            agent_module_name = "Analysis Agent"
            agent_function = generate_analysis
        except ImportError:
            logger.error("Orchestrator: Failed to import analysis_agent.")
            return {"error": "Critical Error: Analysis agent module could not be loaded."}
    else:
        logger.warning(f"Orchestrator: Router returned unexpected decision: '{decision}'. Defaulting to analysis.")
        try:
            from analysis_agent import generate_analysis # Fallback
            agent_module_name = "Analysis Agent (fallback)"
            agent_function = generate_analysis
        except ImportError:
            logger.error("Orchestrator: Failed to import analysis_agent for fallback.")
            return {"error": "Critical Error: Analysis agent module (fallback) could not be loaded."}
            
    try:
        # Agents now expect: query, relevant_filenames (list), relevant_structures (dict for those files), memory
        generated_code = agent_function(
            query=query,
            relevant_filenames=valid_selected_filenames,
            relevant_structures=selected_dataset_structures,
            memory=memory
        )
        return {"code": generated_code, "selected_files": valid_selected_filenames, "agent_type": decision}
    except Exception as e:
        logger.error(f"Orchestrator: Error executing {agent_module_name}: {e}\n{traceback.format_exc()}")
        return {"error": f"Error during code generation by {agent_module_name} for files {valid_selected_filenames}: {e}"}
