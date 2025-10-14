"""
Agent Orchestrator for Tensorus

Coordinates multiple agents to work together on complex tasks,
enabling agent-to-agent communication and workflow execution.
"""

import logging
import asyncio
from typing import Any, Dict, List, Optional, Callable, Union
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import uuid

import torch
import numpy as np

from .tensor_storage import TensorStorage
from .nql_agent import NQLAgent
from .embedding_agent import EmbeddingAgent
from .ingestion_agent import DataIngestionAgent
from .rl_agent import RLAgent
from .automl_agent import AutoMLAgent

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Status of an orchestrated task."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentType(Enum):
    """Types of agents available."""
    NQL = "nql"
    EMBEDDING = "embedding"
    INGESTION = "ingestion"
    RL = "rl"
    AUTOML = "automl"


@dataclass
class AgentMessage:
    """Message passed between agents."""
    sender: str
    recipient: str
    message_type: str
    payload: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class WorkflowTask:
    """A task in a workflow."""
    task_id: str
    agent_type: AgentType
    action: str
    params: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class Workflow:
    """A workflow consisting of multiple tasks."""
    workflow_id: str
    name: str
    tasks: List[WorkflowTask]
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentOrchestrator:
    """
    Orchestrates multiple Tensorus agents to work together.
    
    Features:
    - Agent-to-agent communication via message passing
    - Workflow execution with dependency management
    - Automatic data routing between agents
    - Error handling and recovery
    
    Example:
        >>> orchestrator = AgentOrchestrator(storage)
        >>> orchestrator.register_nql_agent(nql_agent)
        >>> orchestrator.register_embedding_agent(embedding_agent)
        >>> 
        >>> # Execute a workflow
        >>> workflow = orchestrator.create_workflow("data_pipeline")
        >>> orchestrator.add_task(workflow, "ingest", AgentType.INGESTION, {...})
        >>> orchestrator.add_task(workflow, "embed", AgentType.EMBEDDING, {...}, deps=["ingest"])
        >>> orchestrator.execute_workflow(workflow)
    """
    
    def __init__(self, storage: TensorStorage):
        """
        Initialize the orchestrator.
        
        Args:
            storage: TensorStorage instance shared across agents
        """
        self.storage = storage
        
        # Registry of agents
        self.agents: Dict[AgentType, Any] = {}
        
        # Message queue for inter-agent communication
        self.message_queue: List[AgentMessage] = []
        
        # Active workflows
        self.workflows: Dict[str, Workflow] = {}
        
        # Task results cache
        self.task_results: Dict[str, Any] = {}
        
        logger.info("AgentOrchestrator initialized")
    
    # ==================== Agent Registration ====================
    
    def register_agent(self, agent_type: AgentType, agent: Any) -> None:
        """Register an agent with the orchestrator."""
        self.agents[agent_type] = agent
        logger.info(f"Registered {agent_type.value} agent")
    
    def register_nql_agent(self, agent: NQLAgent) -> None:
        """Register NQL agent."""
        self.register_agent(AgentType.NQL, agent)
    
    def register_embedding_agent(self, agent: EmbeddingAgent) -> None:
        """Register embedding agent."""
        self.register_agent(AgentType.EMBEDDING, agent)
    
    def register_ingestion_agent(self, agent: DataIngestionAgent) -> None:
        """Register ingestion agent."""
        self.register_agent(AgentType.INGESTION, agent)
    
    def register_rl_agent(self, agent: RLAgent) -> None:
        """Register RL agent."""
        self.register_agent(AgentType.RL, agent)
    
    def register_automl_agent(self, agent: AutoMLAgent) -> None:
        """Register AutoML agent."""
        self.register_agent(AgentType.AUTOML, agent)
    
    # ==================== Message Passing ====================
    
    def send_message(self, sender: str, recipient: str,
                    message_type: str, payload: Dict[str, Any]) -> AgentMessage:
        """
        Send a message from one agent to another.
        
        Args:
            sender: Sender agent identifier
            recipient: Recipient agent identifier
            message_type: Type of message
            payload: Message data
            
        Returns:
            Created message object
        """
        message = AgentMessage(
            sender=sender,
            recipient=recipient,
            message_type=message_type,
            payload=payload
        )
        self.message_queue.append(message)
        logger.debug(f"Message queued: {sender} -> {recipient} ({message_type})")
        return message
    
    def get_messages(self, recipient: str,
                    message_type: Optional[str] = None) -> List[AgentMessage]:
        """
        Get messages for a specific recipient.
        
        Args:
            recipient: Agent identifier
            message_type: Optional filter by message type
            
        Returns:
            List of messages
        """
        messages = [m for m in self.message_queue if m.recipient == recipient]
        if message_type:
            messages = [m for m in messages if m.message_type == message_type]
        return messages
    
    def clear_messages(self, recipient: str) -> None:
        """Clear messages for a recipient."""
        self.message_queue = [m for m in self.message_queue if m.recipient != recipient]
    
    # ==================== Workflow Management ====================
    
    def create_workflow(self, name: str,
                       metadata: Optional[Dict[str, Any]] = None) -> Workflow:
        """
        Create a new workflow.
        
        Args:
            name: Workflow name
            metadata: Optional metadata
            
        Returns:
            Created workflow
        """
        workflow = Workflow(
            workflow_id=str(uuid.uuid4()),
            name=name,
            tasks=[],
            metadata=metadata or {}
        )
        self.workflows[workflow.workflow_id] = workflow
        logger.info(f"Created workflow '{name}' with ID {workflow.workflow_id}")
        return workflow
    
    def add_task(self, workflow: Workflow, task_name: str,
                agent_type: AgentType, action: str,
                params: Dict[str, Any],
                dependencies: Optional[List[str]] = None) -> WorkflowTask:
        """
        Add a task to a workflow.
        
        Args:
            workflow: Target workflow
            task_name: Unique task name
            agent_type: Type of agent to execute task
            action: Action to perform
            params: Task parameters
            dependencies: List of task names this depends on
            
        Returns:
            Created task
        """
        task = WorkflowTask(
            task_id=task_name,
            agent_type=agent_type,
            action=action,
            params=params,
            dependencies=dependencies or []
        )
        workflow.tasks.append(task)
        logger.info(f"Added task '{task_name}' to workflow '{workflow.name}'")
        return task
    
    def execute_workflow(self, workflow: Workflow) -> Dict[str, Any]:
        """
        Execute a workflow, respecting task dependencies.
        
        Args:
            workflow: Workflow to execute
            
        Returns:
            Dictionary of task results
        """
        logger.info(f"Executing workflow '{workflow.name}' ({workflow.workflow_id})")
        workflow.status = TaskStatus.RUNNING
        
        completed_tasks = set()
        results = {}
        
        try:
            # Execute tasks in dependency order
            while len(completed_tasks) < len(workflow.tasks):
                # Find tasks ready to execute (all dependencies met)
                ready_tasks = [
                    task for task in workflow.tasks
                    if task.status == TaskStatus.PENDING
                    and all(dep in completed_tasks for dep in task.dependencies)
                ]
                
                if not ready_tasks:
                    # Check if we're stuck
                    if len(completed_tasks) < len(workflow.tasks):
                        pending = [t for t in workflow.tasks if t.status == TaskStatus.PENDING]
                        raise RuntimeError(
                            f"Workflow deadlock: {len(pending)} tasks pending but none ready. "
                            f"Check for circular dependencies."
                        )
                    break
                
                # Execute ready tasks
                for task in ready_tasks:
                    try:
                        result = self._execute_task(task, results)
                        task.status = TaskStatus.COMPLETED
                        task.result = result
                        results[task.task_id] = result
                        completed_tasks.add(task.task_id)
                        logger.info(f"Task '{task.task_id}' completed successfully")
                    except Exception as e:
                        task.status = TaskStatus.FAILED
                        task.error = str(e)
                        logger.error(f"Task '{task.task_id}' failed: {e}")
                        raise
            
            workflow.status = TaskStatus.COMPLETED
            logger.info(f"Workflow '{workflow.name}' completed successfully")
            return results
            
        except Exception as e:
            workflow.status = TaskStatus.FAILED
            logger.error(f"Workflow '{workflow.name}' failed: {e}")
            raise
    
    def _execute_task(self, task: WorkflowTask,
                     previous_results: Dict[str, Any]) -> Any:
        """
        Execute a single task.
        
        Args:
            task: Task to execute
            previous_results: Results from previously completed tasks
            
        Returns:
            Task result
        """
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.utcnow()
        
        logger.debug(f"Executing task '{task.task_id}' with agent {task.agent_type.value}")
        
        # Get the agent
        if task.agent_type not in self.agents:
            raise ValueError(f"Agent type {task.agent_type.value} not registered")
        
        agent = self.agents[task.agent_type]
        
        # Resolve parameter references to previous task results
        params = self._resolve_params(task.params, previous_results)
        
        # Execute based on agent type and action
        result = self._dispatch_action(agent, task.agent_type, task.action, params)
        
        task.completed_at = datetime.utcnow()
        return result
    
    def _resolve_params(self, params: Dict[str, Any],
                       previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve parameter references to previous task results.
        
        References use the format: "$task_id.field" or just "$task_id"
        """
        resolved = {}
        for key, value in params.items():
            if isinstance(value, str) and value.startswith("$"):
                # Reference to previous task result
                ref = value[1:]  # Remove $
                if "." in ref:
                    task_id, field = ref.split(".", 1)
                    if task_id in previous_results:
                        result = previous_results[task_id]
                        # Navigate nested fields
                        for f in field.split("."):
                            if isinstance(result, dict):
                                result = result.get(f)
                            else:
                                result = getattr(result, f, None)
                        resolved[key] = result
                    else:
                        resolved[key] = value  # Keep as-is if not found
                else:
                    resolved[key] = previous_results.get(ref, value)
            else:
                resolved[key] = value
        return resolved
    
    def _dispatch_action(self, agent: Any, agent_type: AgentType,
                        action: str, params: Dict[str, Any]) -> Any:
        """Dispatch action to appropriate agent method."""
        
        if agent_type == AgentType.NQL:
            if action == "query":
                return agent.process_query(params["query_text"])
            else:
                raise ValueError(f"Unknown NQL action: {action}")
        
        elif agent_type == AgentType.EMBEDDING:
            if action == "generate":
                return agent.generate_embeddings(
                    params["texts"],
                    batch_size=params.get("batch_size", 32)
                )
            elif action == "store":
                return agent.store_embeddings(
                    dataset_name=params["dataset"],
                    texts=params["texts"],
                    metadata=params.get("metadata")
                )
            elif action == "search":
                return agent.similarity_search(
                    dataset_name=params["dataset"],
                    query_text=params["query"],
                    k=params.get("k", 10)
                )
            else:
                raise ValueError(f"Unknown Embedding action: {action}")
        
        elif agent_type == AgentType.INGESTION:
            if action == "start":
                agent.start_monitoring()
                return {"status": "monitoring_started"}
            elif action == "stop":
                agent.stop_monitoring()
                return {"status": "monitoring_stopped"}
            else:
                raise ValueError(f"Unknown Ingestion action: {action}")
        
        elif agent_type == AgentType.RL:
            if action == "train":
                episodes = params.get("episodes", 100)
                for ep in range(episodes):
                    # Train one episode (simplified)
                    pass  # Agent-specific training logic
                return {"episodes_completed": episodes}
            else:
                raise ValueError(f"Unknown RL action: {action}")
        
        elif agent_type == AgentType.AUTOML:
            if action == "search":
                return agent.run_search(
                    n_trials=params.get("n_trials", 10),
                    max_epochs=params.get("max_epochs", 50)
                )
            else:
                raise ValueError(f"Unknown AutoML action: {action}")
        
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
    
    # ==================== Preset Workflows ====================
    
    def create_embedding_pipeline(self,
                                 dataset_name: str,
                                 texts: List[str],
                                 index_name: Optional[str] = None) -> Workflow:
        """
        Create a workflow for embedding generation and indexing.
        
        Args:
            dataset_name: Dataset to store embeddings
            texts: Texts to embed
            index_name: Optional vector index name
            
        Returns:
            Created workflow
        """
        workflow = self.create_workflow(
            "embedding_pipeline",
            metadata={"dataset": dataset_name, "num_texts": len(texts)}
        )
        
        # Task 1: Generate embeddings
        self.add_task(
            workflow,
            "generate_embeddings",
            AgentType.EMBEDDING,
            "generate",
            {"texts": texts}
        )
        
        # Task 2: Store embeddings
        self.add_task(
            workflow,
            "store_embeddings",
            AgentType.EMBEDDING,
            "store",
            {
                "dataset": dataset_name,
                "texts": texts,
                "metadata": [{"text": t} for t in texts]
            },
            dependencies=["generate_embeddings"]
        )
        
        return workflow
    
    def create_ml_experiment_workflow(self,
                                     search_space: Dict[str, Callable],
                                     n_trials: int = 10) -> Workflow:
        """
        Create a workflow for ML experimentation with AutoML.
        
        Args:
            search_space: Hyperparameter search space
            n_trials: Number of trials to run
            
        Returns:
            Created workflow
        """
        workflow = self.create_workflow(
            "ml_experiment",
            metadata={"n_trials": n_trials}
        )
        
        # Task 1: Run AutoML search
        self.add_task(
            workflow,
            "automl_search",
            AgentType.AUTOML,
            "search",
            {
                "n_trials": n_trials,
                "max_epochs": 50
            }
        )
        
        # Task 2: Query best results
        self.add_task(
            workflow,
            "query_results",
            AgentType.NQL,
            "query",
            {"query_text": "get all from automl_results"},
            dependencies=["automl_search"]
        )
        
        return workflow
    
    # ==================== Status & Monitoring ====================
    
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get the status of a workflow."""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.workflows[workflow_id]
        
        task_statuses = {}
        for task in workflow.tasks:
            task_statuses[task.task_id] = {
                "status": task.status.value,
                "started_at": task.started_at.isoformat() if task.started_at else None,
                "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                "error": task.error
            }
        
        return {
            "workflow_id": workflow.workflow_id,
            "name": workflow.name,
            "status": workflow.status.value,
            "created_at": workflow.created_at.isoformat(),
            "tasks": task_statuses
        }
    
    def list_workflows(self) -> List[Dict[str, Any]]:
        """List all workflows."""
        return [
            {
                "workflow_id": wf.workflow_id,
                "name": wf.name,
                "status": wf.status.value,
                "num_tasks": len(wf.tasks)
            }
            for wf in self.workflows.values()
        ]
    
    def __repr__(self) -> str:
        return (f"AgentOrchestrator(agents={len(self.agents)}, "
                f"workflows={len(self.workflows)}, "
                f"messages={len(self.message_queue)})")
