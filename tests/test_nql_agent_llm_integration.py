import pytest
import torch
import os

# Set the API key environment variable before importing the NQLAgent
# Note: This assumes the API key is passed to the subtask environment
os.environ["GEMINI_API_KEY"] = "AIzaSyC4TzidyvCnvBKekXHUjuZHpjpa7zbXG40"

from tensorus.nql_agent import NQLAgent
from tensorus.tensor_storage import TensorStorage
from tensorus.llm_parser import LLMParser # Ensure NQL_LLM_MODEL is also available if needed by LLMParser

# Try to set a default model if not set, to avoid issues if LLMParser directly uses it
os.environ.setdefault("NQL_LLM_MODEL", "gemini-1.5-flash-latest")


@pytest.fixture
def llm_storage():
    storage = TensorStorage(storage_path=None)  # In-memory for tests
    storage.create_dataset("products_llm")
    storage.insert("products_llm", torch.tensor([1.0, 0.0]), metadata={"id": "P001", "name": "Laptop Pro", "category": "Electronics", "price": 1200.00, "stock": 50, "rating": 4.5})
    storage.insert("products_llm", torch.tensor([2.0, 0.0]), metadata={"id": "P002", "name": "Coffee Maker", "category": "Appliances", "price": 80.00, "stock": 200, "rating": 4.0})
    storage.insert("products_llm", torch.tensor([3.0, 0.0]), metadata={"id": "P003", "name": "Gaming Mouse", "category": "Electronics", "price": 75.00, "stock": 150, "rating": 4.8})
    storage.insert("products_llm", torch.tensor([4.0, 0.0]), metadata={"id": "P004", "name": "Blender", "category": "Appliances", "price": 120.00, "stock": 0, "rating": 3.9})
    storage.insert("products_llm", torch.tensor([5.0, 0.0]), metadata={"id": "P005", "name": "Wireless Keyboard", "category": "Electronics", "price": 99.99, "stock": 120, "rating": 4.2})
    # Add a dataset with a slightly different schema for more complex queries
    storage.create_dataset("employees_llm")
    storage.insert("employees_llm", torch.tensor([1.0]), metadata={"emp_id": "E101", "name": "Alice Smith", "department": "Engineering", "salary": 90000, "years_experience": 5})
    storage.insert("employees_llm", torch.tensor([2.0]), metadata={"emp_id": "E102", "name": "Bob Johnson", "department": "Sales", "salary": 75000, "years_experience": 3})
    storage.insert("employees_llm", torch.tensor([3.0]), metadata={"emp_id": "E103", "name": "Carol White", "department": "Engineering", "salary": 95000, "years_experience": 7})
    return storage

@pytest.fixture
def llm_nql_agent(llm_storage):
    # Ensure the API key is available when NQLAgent initializes LLMParser
    if not os.getenv("GEMINI_API_KEY"):
        pytest.fail("GEMINI_API_KEY environment variable not set for test execution.")

    # Explicitly set NQL_USE_LLM for this agent instance
    os.environ["NQL_USE_LLM"] = "true"
    actual_api_key = os.getenv("GEMINI_API_KEY")
    agent = NQLAgent(llm_storage, use_llm=True, gemini_api_key=actual_api_key)
    # We need to ensure the LLMParser is actually initialized if tests depend on it.
    # The NQLAgent tries to init it if use_llm is True.
    if not agent.llm_parser:
        pytest.fail("LLMParser was not initialized in NQLAgent. Check API key and model availability.")
    return agent

# Test basic LLM parsing - queries that would likely fail regex
def test_llm_simple_query_electronics(llm_nql_agent):
    query = "Show me all electronic gadgets" # "gadgets" as synonym, "electronic" instead of "Electronics"
    # Expected: P001 (Laptop Pro), P003 (Gaming Mouse), P005 (Wireless Keyboard)
    # This query relies on the LLM understanding "electronic gadgets" and mapping to "category: Electronics"

    # For this test, we'll make it simpler and assume the LLM can map "electronic gadgets" to category "Electronics"
    # A more robust test would mock the LLM response for this specific query.
    # Given we are doing integration testing, we expect the actual LLM to handle this.

    result = llm_nql_agent.process_query(query + " from products_llm") # Add dataset specifier
    assert result["success"], f"Query failed: {result.get('message', 'No message')}"
    assert result["count"] == 3
    found_ids = sorted([res["metadata"]["id"] for res in result["results"]])
    assert found_ids == sorted(["P001", "P003", "P005"])

def test_llm_query_multiple_conditions(llm_nql_agent):
    query = "Find electronics that cost less than $100 and have a rating above 4.0 from products_llm"
    # Expected: P003 (Gaming Mouse - $75, 4.8), P005 (Wireless Keyboard - $99.99, 4.2)
    result = llm_nql_agent.process_query(query)
    assert result["success"], f"Query failed: {result.get('message', 'No message')}"
    assert result["count"] == 2
    found_ids = sorted([res["metadata"]["id"] for res in result["results"]])
    assert found_ids == sorted(["P003", "P005"])

def test_llm_query_out_of_stock(llm_nql_agent):
    query = "Which appliances are out of stock from products_llm?"
    # Expected: P004 (Blender - stock 0)
    result = llm_nql_agent.process_query(query)
    assert result["success"], f"Query failed: {result.get('message', 'No message')}"
    assert result["count"] == 1
    assert result["results"][0]["metadata"]["id"] == "P004"

def test_llm_query_natural_phrasing_salary(llm_nql_agent):
    query = "Show me engineers who make more than 92000 dollars from employees_llm"
    # Expected: E103 (Carol White - Engineering, $95000)
    result = llm_nql_agent.process_query(query)
    assert result["success"], f"Query failed: {result.get('message', 'No message')}"
    assert result["count"] == 1
    assert result["results"][0]["metadata"]["emp_id"] == "E103"

def test_llm_query_with_negation(llm_nql_agent):
    query = "List all products that are not in the electronics category from products_llm"
    # Expected: P002 (Coffee Maker), P004 (Blender)
    result = llm_nql_agent.process_query(query)
    assert result["success"], f"Query failed: {result.get('message', 'No message')}"
    assert result["count"] == 2
    found_ids = sorted([res["metadata"]["id"] for res in result["results"]])
    assert found_ids == sorted(["P002", "P004"])

def test_llm_query_unrelated_to_schema(llm_nql_agent):
    query = "What is the color of the sky from products_llm?"
    # Expect the LLM to not find relevant filters or the agent to return an empty/failed result gracefully.
    result = llm_nql_agent.process_query(query)
    # This might be a success=True with 0 results if the LLM tries to find e.g. a "color" field.
    # Or success=False if it can't parse it into a query structure.
    # For now, let's assume it successfully parses to an empty filter set or a filter that matches nothing.
    # LLMParser's current behavior when no filters are found is to return an NQLQuery
    # that would fetch all items if _execute_parsed_query doesn't special case it.
    # However, a smart LLM might say "this field doesn't exist".
    # The NQLAgent's _execute_parsed_query with an empty parsed.filters list would fetch all.
    # This needs to be handled better in NQLAgent or LLMParser.
    # For now, we'll check that it doesn't crash and likely returns 0 relevant items or fails gracefully.

    # If the LLM is good, it might translate to "find products where sky_color = X", which would yield 0.
    # If the LLM is confused, it might return all.
    # Let's assume the LLM is smart enough to not find a "color of the sky" field and thus returns 0.
    if result["success"]:
        assert result["count"] == 0, "Query about sky color should not return products."
    else:
        # Or it could fail to parse into a valid query for the schema
        assert "understand" in result["message"].lower() or "could not parse" in result["message"].lower()


# It's good practice to ensure the environment variable for API key is cleaned up if set by the test suite
# However, pytest fixtures and setup/teardown are better for this.
# For now, this is a simple script.
