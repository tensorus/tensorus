# automl_agent.py
"""
Implements the AutoML Agent for Tensorus.

This agent performs basic hyperparameter optimization using random search.
It trains a simple dummy model on synthetic data, evaluates its performance,
and logs the results, including storing trial results in TensorStorage.

Future Enhancements:
- Implement more advanced search strategies (Bayesian Optimization, Hyperband).
- Allow configuration of different model architectures.
- Integrate with real datasets from TensorStorage.
- Implement early stopping and other training optimizations.
- Store best model state_dict (requires serialization strategy).
- Parallelize trials for faster search.
- Use dedicated hyperparameter optimization libraries (Optuna, Ray Tune).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import logging
import time
from typing import Dict, Any, Callable, Tuple, Optional

from .tensor_storage import TensorStorage # Import our storage module

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Dummy Model Definition ---
class DummyMLP(nn.Module):
    """A simple Multi-Layer Perceptron for regression/classification."""
    def __init__(self, input_dim: int, output_dim: int, hidden_size: int = 64, activation_fn: Callable = nn.ReLU):
        super().__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_size)
        self.activation = activation_fn()
        self.layer_2 = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        x = self.activation(self.layer_1(x))
        x = self.layer_2(x)
        return x


# --- AutoML Agent Class ---
class AutoMLAgent:
    """Performs random search hyperparameter optimization."""

    def __init__(self,
                 tensor_storage: TensorStorage,
                 search_space: Dict[str, Callable[[], Any]],
                 input_dim: int,
                 output_dim: int,
                 task_type: str = 'regression', # 'regression' or 'classification'
                 results_dataset: str = "automl_results"):
        """
        Initializes the AutoML Agent.

        Args:
            tensor_storage: An instance of TensorStorage.
            search_space: Dictionary defining the hyperparameter search space.
                          Keys are param names (e.g., 'lr', 'hidden_size').
                          Values are functions that sample a value for that param (e.g., lambda: 10**random.uniform(-4,-2)).
            input_dim: Input dimension for the dummy model.
            output_dim: Output dimension for the dummy model.
            task_type: Type of task ('regression' or 'classification'), influences loss and data generation.
            results_dataset: Name of the dataset in TensorStorage to store trial results.
        """
        if not isinstance(tensor_storage, TensorStorage):
            raise TypeError("tensor_storage must be an instance of TensorStorage")
        if task_type not in ['regression', 'classification']:
             raise ValueError("task_type must be 'regression' or 'classification'")

        self.tensor_storage = tensor_storage
        self.search_space = search_space
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.task_type = task_type
        self.results_dataset = results_dataset

        # Ensure results dataset exists
        try:
            self.tensor_storage.get_dataset(self.results_dataset)
        except ValueError:
            logger.info(f"Dataset '{self.results_dataset}' not found. Creating it.")
            self.tensor_storage.create_dataset(self.results_dataset)

        # Track best results found during the search
        self.best_score: Optional[float] = None # Use negative infinity for maximization tasks if needed
        self.best_params: Optional[Dict[str, Any]] = None
        # Assuming lower score is better (e.g., loss)
        self.higher_score_is_better = False if task_type == 'regression' else True # Accuracy for classification


        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"AutoML Agent using device: {self.device}")
        logger.info(f"AutoML Agent initialized for {task_type} task. Results stored in '{results_dataset}'.")


    def _generate_synthetic_data(self, n_samples=500, batch_size=32) -> Tuple[Any, Any]:
        """Generates synthetic data loaders for training and validation."""
        X = torch.randn(n_samples, self.input_dim, device=self.device)

        if self.task_type == 'regression':
            # Simple linear relationship with noise
            true_weight = torch.randn(self.input_dim, self.output_dim, device=self.device) * 2
            true_bias = torch.randn(self.output_dim, device=self.device)
            y = X @ true_weight + true_bias + torch.randn(n_samples, self.output_dim, device=self.device) * 0.5
            loss_fn = nn.MSELoss()
        else: # classification
            # Simple linear separation + softmax for multi-class
            if self.output_dim <= 1:
                 raise ValueError("Output dimension must be > 1 for classification task example.")
            true_weight = torch.randn(self.input_dim, self.output_dim, device=self.device)
            logits = X @ true_weight
            y = torch.softmax(logits, dim=1).argmax(dim=1) # Get class labels
            loss_fn = nn.CrossEntropyLoss()

        # Simple split
        split_idx = int(n_samples * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        val_dataset = torch.utils.data.TensorDataset(X_val, y_val)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

        return train_loader, val_loader, loss_fn


    def _build_dummy_model(self, params: Dict[str, Any]) -> nn.Module:
        """Builds the dummy MLP model based on hyperparameters."""
        hidden_size = params.get('hidden_size', 64) # Default if not in params
        activation_name = params.get('activation', 'relu') # Default activation

        act_fn_map = {'relu': nn.ReLU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid}
        activation_fn = act_fn_map.get(activation_name.lower(), nn.ReLU) # Default to ReLU if unknown

        model = DummyMLP(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_size=hidden_size,
            activation_fn=activation_fn
        ).to(self.device)
        return model


    def _train_and_evaluate(self, params: Dict[str, Any], num_epochs: int = 5) -> Optional[float]:
        """Trains and evaluates a model with given hyperparameters."""
        logger.debug(f"Training trial with params: {params}")
        start_time = time.time()

        try:
            # 1. Build Model
            model = self._build_dummy_model(params)

            # 2. Get Data and Loss Function
            train_loader, val_loader, loss_fn = self._generate_synthetic_data()

            # 3. Setup Optimizer
            lr = params.get('lr', 1e-3) # Default LR
            optimizer = optim.Adam(model.parameters(), lr=lr)

            # 4. Training Loop
            model.train()
            for epoch in range(num_epochs):
                epoch_loss = 0
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = loss_fn(outputs, batch_y)

                    # Check for NaN/inf loss
                    if not torch.isfinite(loss):
                         logger.warning(f"Trial failed: Non-finite loss detected during training epoch {epoch}. Params: {params}")
                         return None # Indicate failure

                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                # logger.debug(f" Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss/len(train_loader):.4f}")


            # 5. Evaluation Loop
            model.eval()
            total_val_loss = 0
            total_correct = 0
            total_samples = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = model(batch_X)
                    loss = loss_fn(outputs, batch_y)
                    total_val_loss += loss.item()

                    if self.task_type == 'classification':
                         predicted = outputs.argmax(dim=1)
                         total_correct += (predicted == batch_y).sum().item()
                         total_samples += batch_y.size(0)

            avg_val_loss = total_val_loss / len(val_loader)
            duration = time.time() - start_time
            logger.debug(f"Trial completed in {duration:.2f}s. Val Loss: {avg_val_loss:.4f}")


            # 6. Return Score
            if self.task_type == 'regression':
                score = avg_val_loss # Lower is better
            else: # classification
                 accuracy = total_correct / total_samples if total_samples > 0 else 0
                 score = accuracy # Higher is better
                 logger.debug(f" Trial Val Accuracy: {accuracy:.4f}")

            return score

        except Exception as e:
            logger.error(f"Trial failed with exception for params {params}: {e}", exc_info=True)
            return None # Indicate failure


    def hyperparameter_search(self, trials: int, num_epochs_per_trial: int = 5) -> Optional[Dict[str, Any]]:
        """
        Performs random search for the specified number of trials.

        Args:
            trials: The number of hyperparameter configurations to try.
            num_epochs_per_trial: Number of epochs to train each model configuration.

        Returns:
            The dictionary of hyperparameters that achieved the best score, or None if no trial succeeded.
        """
        logger.info(f"--- Starting Hyperparameter Search ({trials} trials) ---")
        self.best_score = None
        self.best_params = None

        for i in range(trials):
            # 1. Sample hyperparameters
            current_params = {name: sampler() for name, sampler in self.search_space.items()}
            logger.info(f"Trial {i+1}/{trials}: Testing params: {current_params}")

            # 2. Train and evaluate
            score = self._train_and_evaluate(current_params, num_epochs=num_epochs_per_trial)

            # 3. Store results in TensorStorage (even if trial failed, record params and score=None)
            score_tensor = torch.tensor(float('nan') if score is None else score) # Store NaN for failed trials
            trial_metadata = {
                "trial_id": i + 1,
                "params": current_params, # Store params dict directly in metadata
                "score": score, # Store score also in metadata for easier querying
                "task_type": self.task_type,
                "search_timestamp_utc": time.time(),
                "created_by": "AutoMLAgent" # Add agent source
            }
            try:
                 record_id = self.tensor_storage.insert(
                     self.results_dataset,
                     score_tensor,
                     trial_metadata
                 )
                 logger.debug(f"Stored trial {i+1} results (Score: {score}) with record ID: {record_id}")
            except Exception as e:
                 logger.error(f"Failed to store trial {i+1} results in TensorStorage: {e}")


            # 4. Update best score if trial succeeded and is better
            if score is not None:
                is_better = False
                if self.best_score is None:
                    is_better = True
                elif self.higher_score_is_better and score > self.best_score:
                     is_better = True
                elif not self.higher_score_is_better and score < self.best_score:
                     is_better = True

                if is_better:
                     self.best_score = score
                     self.best_params = current_params
                     logger.info(f"*** New best score found! Trial {i+1}: Score={score:.4f}, Params={current_params} ***")


        logger.info(f"--- Hyperparameter Search Finished ---")
        if self.best_params:
            logger.info(f"Best score overall: {self.best_score:.4f}")
            logger.info(f"Best hyperparameters found: {self.best_params}")
            # Optional: Here you could trigger saving the best model's state_dict
            # e.g., self._save_best_model(self.best_params)
        else:
            logger.warning("No successful trials completed. Could not determine best parameters.")

        return self.best_params


# --- Example Usage ---
if __name__ == "__main__":
    print("--- Starting AutoML Agent Example ---")

    # 1. Setup TensorStorage
    storage = TensorStorage()

    # 2. Define Search Space
    # Simple example for MLP regression
    search_space_reg = {
        'lr': lambda: 10**random.uniform(-5, -2), # Log uniform for learning rate
        'hidden_size': lambda: random.choice([32, 64, 128, 256]),
        'activation': lambda: random.choice(['relu', 'tanh'])
    }

    # 3. Create the AutoML Agent
    input_dim = 10
    output_dim = 1 # Regression task
    automl_agent = AutoMLAgent(
        tensor_storage=storage,
        search_space=search_space_reg,
        input_dim=input_dim,
        output_dim=output_dim,
        task_type='regression',
        results_dataset="automl_regression_results"
    )

    # 4. Run the hyperparameter search
    num_trials = 20 # Number of random configurations to test
    num_epochs = 10 # Epochs per trial (keep low for speed)
    best_hyperparams = automl_agent.hyperparameter_search(trials=num_trials, num_epochs_per_trial=num_epochs)


    # 5. Optional: Check TensorStorage for results
    print("\n--- Checking TensorStorage contents (Sample) ---")
    try:
        results_count = len(storage.get_dataset(automl_agent.results_dataset))
        print(f"Found {results_count} trial records in '{automl_agent.results_dataset}'.")

        if results_count > 0:
            print("\nExample trial record (metadata):")
            # Sample one record to show structure
            sample_trial = storage.sample_dataset(automl_agent.results_dataset, 1)
            if sample_trial:
                 print(f"  Metadata: {sample_trial[0]['metadata']}")
                 print(f"  Score Tensor: {sample_trial[0]['tensor']}") # Should contain the score or NaN

            # You could also query for the best score using NQLAgent if needed
            # (e.g., find record where score = best_score) - requires parsing results first.

    except ValueError as e:
        print(f"Could not retrieve dataset '{automl_agent.results_dataset}': {e}")
    except Exception as e:
         print(f"An error occurred checking storage: {e}")


    print("\n--- AutoML Agent Example Finished ---")