import os
from uuid import uuid4
from typing import Any, Dict, List

import yaml
import mlflow
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse

from langchain.agents import create_agent
from databricks_langchain import ChatDatabricks, UCFunctionToolkit
from langgraph.checkpoint.memory import InMemorySaver


# load agent config from yaml file (in the same directory)
def _load_config(path: str = "agent-config.yaml") -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    llm_endpoint = cfg.get("llm_endpoint")
    llm_temperature = float(cfg.get("llm_temperature"))
    system_prompt = cfg.get("system_prompt")
    function_names = cfg.get("function_names")

    return {
        "llm_endpoint": llm_endpoint,
        "llm_temperature": llm_temperature,
        "system_prompt": system_prompt,
        "function_names": function_names,
    }


# build LangChain agent with the config above:
# this is the same code as the smoke test above
def build_agent(
    llm_endpoint: str,
    system_prompt: str,
    llm_temperature: float = 0.1,
    function_names: list[str] = None,
):
    """
    Creates a UC-tool-calling ReAct agent with LangChain.
    Args:
        llm_endpoint (str): The endpoint of the LLM.
        system_prompt (str): The system prompt for the agent.
        llm_temperature (float): The temperature for the LLM.
        function_names (list[str]): The Unity Catalog fully-qualified names of the functions to be called. E.g., catalog.schema.function

    """

    # init the model with OpenAI standard I/O schemas
    llm = ChatDatabricks(endpoint=llm_endpoint, temperature=llm_temperature)

    # Use UCFunctionToolkit to integrate UC-registered tools
    if function_names:
        toolkit = UCFunctionToolkit(function_names=function_names)
        tools = toolkit.tools
    else:
        tools = []

    # Optional: use an in-memory saver to save the agent's state
    checkpointer = InMemorySaver()

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        checkpointer=checkpointer,
    )
    return agent


# MLflow ResponsesAgent interface implementation for LangChain agent
class LangChainResponsesAgent(ResponsesAgent):
    def __init__(self):
        agent_config = _load_config()
        self._agent_config = agent_config
        self._agent = build_agent(
            llm_endpoint=agent_config["llm_endpoint"],
            system_prompt=agent_config["system_prompt"],
            llm_temperature=agent_config["llm_temperature"],
            function_names=agent_config["function_names"]
        )

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        msgs = [m.model_dump() for m in request.input] # example: [{'role': 'user' | 'assistant', 'content': '...'}, ...]
        # _ = _last_user_text(msgs) if msgs else ""

        # Generate a unique thread ID for each pred (reset memory):
        # do this if you want each API call to be independent from each other
        thread_id = f"imda-{uuid4()}"
        result = self._agent.invoke(
            {"messages": msgs},
            config={"configurable": {"thread_id": thread_id}},
        )

        # Extract agent response text
        try:
            text = result["messages"][-1].content
        except Exception:
            text = str(result)
        return ResponsesAgentResponse(
            output=[self.create_text_output_item(text, str(uuid4()))],
            custom_outputs=request.custom_inputs,
        )

# get the model obj for mlflow:
AGENT = LangChainResponsesAgent()
mlflow.models.set_model(AGENT)