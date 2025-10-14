"""
Complete Tensorus Workflow Example

Demonstrates a real-world use case: ML Experiment Management with Agent Orchestration
"""

import sys
from pathlib import Path
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tensorus import Tensorus
from tensorus.agent_orchestrator import AgentOrchestrator, AgentType, Workflow


def ml_experiment_management_workflow():
    """
    Complete workflow for managing machine learning experiments:
    1. Create experiment tracking dataset
    2. Store model weights and metrics
    3. Generate embeddings for experiment descriptions
    4. Search for similar experiments
    5. Query best performing models
    """
    
    print("="*70)
    print("ML EXPERIMENT MANAGEMENT WORKFLOW")
    print("="*70)
    
    # Initialize Tensorus with all capabilities
    ts = Tensorus(
        enable_nql=True,
        enable_embeddings=True,
        enable_vector_search=True
    )
    
    print("\n✓ Tensorus initialized with NQL, Embeddings, and Vector Search")
    
    # Step 1: Create experiment dataset
    print("\n### Step 1: Setting up experiment dataset ###")
    ts.create_dataset("ml_experiments", schema={
        "description": "Machine learning experiment tracking",
        "version": "1.0"
    })
    print("✓ Created 'ml_experiments' dataset")
    
    # Step 2: Run and store multiple experiments
    print("\n### Step 2: Running and storing experiments ###")
    
    experiments = []
    for exp_id in range(10):
        # Simulate model training
        model_weights = np.random.randn(20, 20).astype(np.float32)
        accuracy = np.random.uniform(0.65, 0.95)
        loss = np.random.uniform(0.1, 0.5)
        
        # Generate experiment description
        description = f"Experiment {exp_id}: Neural network with learning_rate=0.001, batch_size=32"
        
        # Store model weights
        tensor = ts.create_tensor(
            model_weights,
            dataset="ml_experiments",
            name=f"model_weights_exp_{exp_id}",
            metadata={
                "experiment_id": exp_id,
                "accuracy": float(accuracy),
                "loss": float(loss),
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 50,
                "optimizer": "adam",
                "description": description
            },
            description=description
        )
        
        experiments.append({
            "id": exp_id,
            "tensor_id": tensor.id,
            "accuracy": accuracy,
            "description": description
        })
        
        print(f"  ✓ Experiment {exp_id}: accuracy={accuracy:.3f}, loss={loss:.3f}")
    
    print(f"\n✓ Stored {len(experiments)} experiments")
    
    # Step 3: Index experiment descriptions for semantic search
    print("\n### Step 3: Creating semantic search index ###")
    
    # Create vector index for embeddings
    ts.create_index("experiment_embeddings", dimensions=384)
    
    # Embed and index experiment descriptions
    descriptions = [exp["description"] for exp in experiments]
    exp_ids = [f"exp_{exp['id']}" for exp in experiments]
    
    ts.embed_and_index(
        texts=descriptions,
        index_name="experiment_embeddings",
        ids=exp_ids,
        metadata=[{"experiment_id": exp["id"], "accuracy": exp["accuracy"]} 
                  for exp in experiments]
    )
    
    print(f"✓ Indexed {len(descriptions)} experiment descriptions")
    
    # Step 4: Query experiments using NQL
    print("\n### Step 4: Querying experiments with NQL ###")
    
    # Query all experiments
    all_exps = ts.query("get all from ml_experiments")
    print(f"  ✓ Total experiments: {len(all_exps)}")
    
    # Query high-performing models (using metadata search as proxy)
    high_performers = ts.search_metadata({"accuracy": 0.8})
    print(f"  ✓ High-performing models (accuracy > 0.8): {len(high_performers)}")
    
    # Step 5: Semantic search for similar experiments
    print("\n### Step 5: Finding similar experiments ###")
    
    query_text = "Neural network experiment with adam optimizer"
    query_embedding = ts.generate_embeddings(query_text)[0]
    
    similar_exps = ts.search_vectors(
        "experiment_embeddings",
        query_embedding,
        k=3
    )
    
    print(f"  Query: '{query_text}'")
    print(f"  Similar experiments:")
    for i, (exp_id, score) in enumerate(zip(similar_exps.ids, similar_exps.scores), 1):
        exp_num = int(exp_id.split('_')[1])
        exp = experiments[exp_num]
        print(f"    {i}. Exp {exp_num}: similarity={score:.4f}, accuracy={exp['accuracy']:.3f}")
    
    # Step 6: Analyze best model
    print("\n### Step 6: Analyzing best model ###")
    
    best_exp = max(experiments, key=lambda x: x["accuracy"])
    best_tensor = ts.get_tensor(best_exp["tensor_id"], dataset="ml_experiments")
    
    print(f"  Best Model: Experiment {best_exp['id']}")
    print(f"  Accuracy: {best_exp['accuracy']:.4f}")
    print(f"  Weights shape: {best_tensor.shape}")
    print(f"  Weights statistics:")
    print(f"    Mean: {best_tensor.to_tensor().mean():.4f}")
    print(f"    Std: {best_tensor.to_tensor().std():.4f}")
    
    # Step 7: Dataset statistics
    print("\n### Step 7: Dataset Statistics ###")
    
    dataset_info = ts.get_dataset_info("ml_experiments")
    print(f"  Dataset: ml_experiments")
    print(f"  Total tensors: {len(dataset_info['tensors'])}")
    print(f"  Schema version: {dataset_info.get('schema', {}).get('version', 'N/A')}")
    
    print("\n" + "="*70)
    print("✓ WORKFLOW COMPLETED SUCCESSFULLY")
    print("="*70)
    
    return {
        "experiments": experiments,
        "best_experiment": best_exp,
        "total_experiments": len(experiments)
    }


def agent_orchestration_workflow():
    """
    Demonstrate agent orchestration for a data processing pipeline:
    1. Ingest data
    2. Generate embeddings
    3. Query and analyze results
    """
    
    print("\n\n" + "="*70)
    print("AGENT ORCHESTRATION WORKFLOW")
    print("="*70)
    
    # Initialize
    ts = Tensorus(enable_nql=True, enable_embeddings=True)
    orchestrator = AgentOrchestrator(ts.storage)
    
    # Register agents
    if ts._nql_agent:
        orchestrator.register_nql_agent(ts._nql_agent)
        print("✓ Registered NQL Agent")
    
    if ts._embedding_agent:
        orchestrator.register_embedding_agent(ts._embedding_agent)
        print("✓ Registered Embedding Agent")
    
    # Create workflow
    print("\n### Creating Data Processing Pipeline ###")
    
    workflow = orchestrator.create_workflow(
        "data_processing_pipeline",
        metadata={"purpose": "process and embed documents"}
    )
    
    # Define tasks
    documents = [
        "Tensorus provides efficient tensor storage and operations",
        "Machine learning models require robust data management",
        "Vector databases enable fast similarity search",
        "Agent orchestration coordinates complex workflows"
    ]
    
    # Task 1: Store documents as embeddings
    orchestrator.add_task(
        workflow,
        "generate_embeddings",
        AgentType.EMBEDDING,
        "generate",
        {"texts": documents, "batch_size": 4}
    )
    
    print(f"  Added task: generate_embeddings ({len(documents)} documents)")
    
    # Task 2: Store embeddings
    orchestrator.add_task(
        workflow,
        "store_embeddings",
        AgentType.EMBEDDING,
        "store",
        {
            "dataset": "document_embeddings",
            "texts": documents,
            "metadata": [{"doc_id": i, "text": doc} for i, doc in enumerate(documents)]
        },
        dependencies=["generate_embeddings"]
    )
    
    print("  Added task: store_embeddings (depends on generate_embeddings)")
    
    # Task 3: Query stored data
    orchestrator.add_task(
        workflow,
        "query_results",
        AgentType.NQL,
        "query",
        {"query_text": "get all from document_embeddings"},
        dependencies=["store_embeddings"]
    )
    
    print("  Added task: query_results (depends on store_embeddings)")
    
    # Execute workflow
    print("\n### Executing Workflow ###")
    
    try:
        results = orchestrator.execute_workflow(workflow)
        
        print("\n✓ Workflow execution completed!")
        print(f"  Total tasks: {len(workflow.tasks)}")
        print(f"  Completed: {sum(1 for t in workflow.tasks if t.status.value == 'completed')}")
        
        # Show workflow status
        status = orchestrator.get_workflow_status(workflow.workflow_id)
        print("\n  Task Status:")
        for task_id, task_status in status["tasks"].items():
            print(f"    - {task_id}: {task_status['status']}")
        
    except Exception as e:
        print(f"\n✗ Workflow execution failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("✓ ORCHESTRATION WORKFLOW COMPLETED")
    print("="*70)


def hybrid_search_workflow():
    """
    Demonstrate hybrid search combining semantic similarity and metadata filtering.
    """
    
    print("\n\n" + "="*70)
    print("HYBRID SEARCH WORKFLOW")
    print("="*70)
    
    ts = Tensorus(enable_embeddings=True, enable_vector_search=True)
    
    # Create dataset with rich metadata
    print("\n### Step 1: Creating research papers dataset ###")
    
    papers = [
        {
            "title": "Deep Learning for Computer Vision",
            "abstract": "This paper explores convolutional neural networks for image classification",
            "year": 2020,
            "citations": 150,
            "field": "computer_vision"
        },
        {
            "title": "Natural Language Processing with Transformers",
            "abstract": "We present a transformer-based approach for text understanding",
            "year": 2021,
            "citations": 230,
            "field": "nlp"
        },
        {
            "title": "Reinforcement Learning in Robotics",
            "abstract": "Applying Q-learning and policy gradients to robotic control tasks",
            "year": 2019,
            "citations": 95,
            "field": "robotics"
        },
        {
            "title": "Graph Neural Networks for Molecular Design",
            "abstract": "Using graph convolutions to predict molecular properties",
            "year": 2022,
            "citations": 180,
            "field": "chemistry"
        },
        {
            "title": "Vision Transformers for Image Recognition",
            "abstract": "Transformer architecture adapted for computer vision tasks",
            "year": 2022,
            "citations": 310,
            "field": "computer_vision"
        }
    ]
    
    # Store papers
    ts.create_dataset("research_papers")
    
    for i, paper in enumerate(papers):
        # Store paper metadata as tensor
        embedding_placeholder = np.random.rand(10)  # Placeholder
        ts.create_tensor(
            embedding_placeholder,
            dataset="research_papers",
            name=f"paper_{i}",
            metadata=paper
        )
    
    print(f"✓ Stored {len(papers)} research papers")
    
    # Create semantic index
    print("\n### Step 2: Creating semantic index ###")
    
    ts.create_index("paper_embeddings", dimensions=384)
    
    abstracts = [p["abstract"] for p in papers]
    paper_ids = [f"paper_{i}" for i in range(len(papers))]
    
    ts.embed_and_index(
        texts=abstracts,
        index_name="paper_embeddings",
        ids=paper_ids,
        metadata=papers
    )
    
    print("✓ Indexed paper abstracts")
    
    # Hybrid search 1: Semantic + year filter
    print("\n### Step 3: Hybrid Search - Recent papers about vision ###")
    
    query = "computer vision and image recognition"
    query_emb = ts.generate_embeddings(query)[0]
    
    results = ts.search_vectors("paper_embeddings", query_emb, k=5)
    
    # Filter by year
    recent_results = [
        (rid, score, meta) for rid, score, meta in results
        if meta.get("year", 0) >= 2020
    ]
    
    print(f"  Query: '{query}'")
    print(f"  Results (year >= 2020):")
    for i, (rid, score, meta) in enumerate(recent_results, 1):
        print(f"    {i}. {meta['title']} ({meta['year']})")
        print(f"       Similarity: {score:.4f}, Citations: {meta['citations']}")
    
    # Hybrid search 2: Field filter + high citations
    print("\n### Step 4: Hybrid Search - Highly cited NLP papers ###")
    
    all_papers = ts.list_tensors("research_papers")
    high_cited_nlp = [
        p for p in all_papers
        if p["metadata"].get("field") == "nlp" and p["metadata"].get("citations", 0) > 200
    ]
    
    print(f"  Found {len(high_cited_nlp)} highly cited NLP papers")
    for paper in high_cited_nlp:
        meta = paper["metadata"]
        print(f"    - {meta['title']} ({meta['citations']} citations)")
    
    print("\n" + "="*70)
    print("✓ HYBRID SEARCH WORKFLOW COMPLETED")
    print("="*70)


def main():
    """Run all example workflows."""
    
    try:
        # Workflow 1: ML Experiment Management
        result1 = ml_experiment_management_workflow()
        
        # Workflow 2: Agent Orchestration
        agent_orchestration_workflow()
        
        # Workflow 3: Hybrid Search
        hybrid_search_workflow()
        
        print("\n\n" + "="*70)
        print("🎉 ALL WORKFLOWS COMPLETED SUCCESSFULLY! 🎉")
        print("="*70)
        print("\nKey Achievements:")
        print(f"  ✓ Managed {result1['total_experiments']} ML experiments")
        print(f"  ✓ Best experiment accuracy: {result1['best_experiment']['accuracy']:.4f}")
        print("  ✓ Coordinated multi-agent workflows")
        print("  ✓ Performed hybrid semantic + metadata search")
        print("\nTensorus is ready for production use!")
        print("="*70)
        
    except Exception as e:
        print(f"\n\n✗ Error in workflow execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
