import os
from uuid import uuid4
from typing import Any, Dict, List

import yaml
import mlflow
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse

from langchain.agents import create_agent
from databricks_langchain import ChatDatabricks, VectorSearchRetrieverTool
from langgraph.checkpoint.memory import InMemorySaver


# load agent config from yaml file (in the same directory)
def _load_config(path: str = "agent-config.yaml") -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    llm_endpoint = cfg.get("llm_endpoint_name")
    vs = cfg.get("vector_search", {}) or {}
    index_name = vs.get("index_name")
    num_results = int(vs.get("num_results", 3))
    return {
        "llm_endpoint_name": llm_endpoint,
        "index_name": index_name,
        "num_results": num_results
    }


# build LangChain agent with the config above:
# this is the same code as the smoke test above
def build_agent(llm_endpoint: str, index_name: str, num_results: int = 3):
    # init the model with OpenAI standard I/O schemas
    model = ChatDatabricks(endpoint=llm_endpoint, max_tokens=500)

    # define the vector search tool
    vs_tool = VectorSearchRetrieverTool(
        name="imda_llm_testing_knowledge_search",
        index_name=index_name,
        description="Search the IMDA's document `Starter Kit for Testing LLM-Based Applications for Safety and Reliability` for relevant information on testing LLM-based applications.",
        num_results=num_results
    )

    # Optional: use an in-memory saver to save the agent's state
    checkpointer = InMemorySaver()

    system_prompt = """
    You are the knowledge assistant for Singapore's Infocomm Media Development Authority (IMDA), specialized on  testing LLM-based applications for safety and reliability. Respond in a clear, professional, and factual tone appropriate for developers. Use only verified information from the internal documents, and include source references when available. If the answer cannot be found, clearly state that, and suggest related sections or next steps. Do not speculate, make assumptions, or provide informaiton outside of the provided context.
    """

    agent = create_agent(
        model=model,
        tools=[vs_tool],
        system_prompt=system_prompt,
        checkpointer=checkpointer
    )
    return agent

# MLflow ResponsesAgent interface implementation for LangChain agent
class LangChainResponsesAgent(ResponsesAgent):
    def __init__(self):
        cfg = _load_config()
        self._cfg = cfg
        self._agent = build_agent(
            llm_endpoint=cfg["llm_endpoint_name"],
            index_name=cfg["index_name"],
            num_results=cfg["num_results"]
        )

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        msgs = [m.model_dump() for m in request.input] # example: [{'role': 'user' | 'assistant', 'content': '...'}, ...]
        # _ = _last_user_text(msgs) if msgs else ""

        # Generate a unique thread ID for each pred:
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

