from flask import Flask, request, jsonify
import numpy as np
import logging
import os
import json
from typing import Dict, Any

from tensor_database import TensorDatabase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the app
app = Flask(__name__)

# Initialize the database
db = TensorDatabase(
    storage_path=os.environ.get("TENSORUS_STORAGE_PATH", "data/tensor_db.h5"),
    index_path=os.environ.get("TENSORUS_INDEX_PATH", "data/tensor_index.pkl"),
    config_path=os.environ.get("TENSORUS_CONFIG_PATH", "config/db_config.json"),
    use_gpu=os.environ.get("TENSORUS_USE_GPU", "false").lower() == "true"
)

@app.route("/health", methods=["GET"])
def health_check():
    """Simple health check endpoint."""
    return jsonify({"status": "healthy", "version": "0.1.0"}), 200

@app.route("/tensor", methods=["POST"])
def store_tensor():
    """Store a tensor in the database."""
    try:
        data = request.json
        if not data or "tensor" not in data:
            return jsonify({"error": "No tensor data provided"}), 400
        
        tensor = np.array(data["tensor"])
        metadata = data.get("metadata", {})
        index = data.get("index", None)
        
        tensor_id = db.save(tensor, metadata, index)
        
        return jsonify({"tensor_id": tensor_id}), 201
    except Exception as e:
        logger.error(f"Error storing tensor: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/tensor/<tensor_id>", methods=["GET"])
def get_tensor(tensor_id):
    """Retrieve a tensor from the database."""
    try:
        tensor, metadata = db.get(tensor_id)
        return jsonify({"tensor_id": tensor_id, 
                        "tensor": tensor.tolist(), 
                        "metadata": metadata}), 200
    except KeyError:
        return jsonify({"error": f"Tensor {tensor_id} not found"}), 404
    except Exception as e:
        logger.error(f"Error retrieving tensor: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/tensor/<tensor_id>", methods=["PUT"])
def update_tensor(tensor_id):
    """Update a tensor or its metadata."""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No update data provided"}), 400
        
        tensor = np.array(data["tensor"]) if "tensor" in data else None
        metadata = data.get("metadata", None)
        
        success = db.update(tensor_id, tensor, metadata)
        
        if success:
            return jsonify({"message": f"Tensor {tensor_id} updated successfully"}), 200
        else:
            return jsonify({"error": f"Failed to update tensor {tensor_id}"}), 500
    except KeyError:
        return jsonify({"error": f"Tensor {tensor_id} not found"}), 404
    except Exception as e:
        logger.error(f"Error updating tensor: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/tensor/<tensor_id>", methods=["DELETE"])
def delete_tensor(tensor_id):
    """Delete a tensor from the database."""
    try:
        success = db.delete(tensor_id)
        
        if success:
            return jsonify({"message": f"Tensor {tensor_id} deleted successfully"}), 200
        else:
            return jsonify({"error": f"Tensor {tensor_id} not found"}), 404
    except Exception as e:
        logger.error(f"Error deleting tensor: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/tensors", methods=["GET"])
def list_tensors():
    """List all tensors in the database."""
    try:
        tensors = db.list_tensors()
        return jsonify({"tensors": tensors}), 200
    except Exception as e:
        logger.error(f"Error listing tensors: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/search", methods=["POST"])
def search_similar():
    """Search for similar tensors."""
    try:
        data = request.json
        if not data or "tensor" not in data:
            return jsonify({"error": "No query tensor provided"}), 400
        
        query_tensor = np.array(data["tensor"])
        k = int(data.get("k", 5))
        
        results = db.search_similar(query_tensor, k)
        
        # Format results for JSON response
        formatted_results = []
        for result in results:
            formatted_results.append({
                "tensor_id": result["tensor_id"],
                "distance": float(result["distance"]),
                "tensor": result["tensor"].tolist() if data.get("include_tensors", False) else None,
                "metadata": result["metadata"]
            })
        
        return jsonify({"results": formatted_results}), 200
    except Exception as e:
        logger.error(f"Error searching: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/process", methods=["POST"])
def process_tensors():
    """Process tensors using the specified operation."""
    try:
        data = request.json
        if not data or "operation" not in data or "tensors" not in data:
            return jsonify({"error": "Missing required fields"}), 400
        
        operation = data["operation"]
        tensors = data["tensors"]
        kwargs = data.get("args", {})
        
        # Process the tensors
        result = db.process(operation, tensors, **kwargs)
        
        return jsonify({"result": result.tolist()}), 200
    except Exception as e:
        logger.error(f"Error processing tensors: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/batch", methods=["POST"])
def batch_save():
    """Save multiple tensors in a single request."""
    try:
        data = request.json
        if not data or "tensors" not in data:
            return jsonify({"error": "No tensors provided"}), 400
        
        tensors = [np.array(t) for t in data["tensors"]]
        metadatas = data.get("metadatas", None)
        
        tensor_ids = db.batch_save(tensors, metadatas)
        
        return jsonify({"tensor_ids": tensor_ids}), 201
    except Exception as e:
        logger.error(f"Error batch saving tensors: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/metrics", methods=["GET"])
def get_metrics():
    """Get database performance metrics."""
    try:
        metrics = db.get_metrics()
        return jsonify({"metrics": metrics}), 200
    except Exception as e:
        logger.error(f"Error retrieving metrics: {e}")
        return jsonify({"error": str(e)}), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    # Make sure directories exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("config", exist_ok=True)
    
    # Create default config if it doesn't exist
    config_path = os.environ.get("TENSORUS_CONFIG_PATH", "config/db_config.json")
    if not os.path.exists(config_path):
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        default_config = {
            "index_type": "flat",
            "metric": "l2",
            "default_dimension": 1024,
            "auto_index": True
        }
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
    
    # Run the app
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("DEBUG", "false").lower() == "true") 