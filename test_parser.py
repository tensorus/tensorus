import sys
from tensorus.tensor_operation_api import TensorQueryParser

def test_parser():
    print("Testing TensorQueryParser...")
    
    # Test simple query
    query_str = "SELECT tensors WHERE category = 'test'"
    print(f"\nTesting query: {query_str}")
    
    try:
        parser = TensorQueryParser()
        query = parser.parse_query(query_str)
        print("Parsed query:")
        print(f"- Operation: {query.operation}")
        print(f"- Target: {query.target}")
        print(f"- Conditions: {query.conditions}")
        print(f"- Operations: {query.operations}")
        print("Success!")
    except Exception as e:
        print(f"Error parsing query: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_parser()
