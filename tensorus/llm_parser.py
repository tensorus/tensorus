import os
import json
import logging
from typing import List, Any, Optional
from pydantic import BaseModel, Field

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import PydanticOutputParser
    from langchain_core.messages import SystemMessage, HumanMessage
    LANGCHAIN_AVAILABLE = True
except Exception as e:  # pragma: no cover - library may be missing in test env
    ChatGoogleGenerativeAI = None
    ChatPromptTemplate = None
    PydanticOutputParser = None
    SystemMessage = None
    HumanMessage = None
    LANGCHAIN_AVAILABLE = False

logger = logging.getLogger(__name__)

class QueryCondition(BaseModel):
    key: str
    operator: str
    value: Any

class FilterClause(BaseModel):
    joiner: str = Field("AND", description="Logical join for conditions")
    conditions: List[QueryCondition] = Field(default_factory=list)

class NQLQuery(BaseModel):
    dataset: str
    filters: List[FilterClause] = Field(default_factory=list)

class LLMParser:
    """Parse natural language queries into ``NQLQuery`` objects using Gemini."""

    def __init__(self, model_name: Optional[str] = None):
        model_name = model_name or os.getenv("NQL_LLM_MODEL", "gemini-2.0-flash")
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("langchain-google-genai and langchain-core are required for LLM parsing")
        self.model = ChatGoogleGenerativeAI(model=model_name, temperature=0)
        self.output_parser = PydanticOutputParser(pydantic_object=NQLQuery)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You convert user queries about datasets into JSON."
                 " Use this format: {format_instructions}.\nDataset schema: {schema}"),
                ("human", "{query}")
            ]
        )
        self.chain = self.prompt | self.model | self.output_parser

    def parse(self, query: str, schema: dict) -> Optional[NQLQuery]:
        """Return ``NQLQuery`` parsed from ``query`` or ``None`` on failure."""
        try:
            return self.chain.invoke({
                "query": query,
                "schema": json.dumps(schema),
                "format_instructions": self.output_parser.get_format_instructions(),
            })
        except Exception as e:  # pragma: no cover - runtime failures
            logger.error(f"LLM parsing failed: {e}")
            return None
