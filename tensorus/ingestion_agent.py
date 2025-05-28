# ingestion_agent.py
"""
Implements the Autonomous Data Ingestion Agent for Tensorus.

This agent monitors a source directory for new data files (e.g., CSV, images),
preprocesses them into tensors using configurable functions, performs basic
validation, and inserts them into a specified dataset in TensorStorage.

Future Enhancements:
- Monitor cloud storage (S3, GCS) and APIs.
- More robust error handling for malformed files.
- More sophisticated duplicate detection (e.g., file hashing).
- Support for streaming data sources.
- Asynchronous processing for higher throughput.
- More complex and configurable preprocessing pipelines.
- Schema validation against predefined dataset schemas.
- Resource management controls.
"""

import os
import time
import glob
import logging
import threading
import csv
from PIL import Image
import torch
import torchvision.transforms as T # Use torchvision for image transforms
import collections # Added import

from typing import Dict, Callable, Optional, Tuple, List, Any
from .tensor_storage import TensorStorage # Import our storage module

# Configure basic logging (can be customized further)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Custom Log Handler ---
class AgentMemoryLogHandler(logging.Handler):
    """
    A custom logging handler that stores log records in a deque.
    """
    def __init__(self, deque: collections.deque):
        super().__init__()
        self.deque = deque

    def emit(self, record: logging.LogRecord) -> None:
        self.deque.append(self.format(record))


# --- Default Preprocessing Functions ---

def preprocess_csv(file_path: str) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
    """
    Basic CSV preprocessor. Assumes numeric data.
    Reads a CSV, converts rows to tensors (one tensor per row or one tensor for the whole file).
    Returns a single tensor representing the whole file for simplicity here.
    """
    metadata = {"source_file": file_path, "type": "csv"}
    data = []
    try:
        with open(file_path, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader, None) # Skip header row
            metadata["header"] = header
            for row in reader:
                # Attempt to convert row elements to floats
                try:
                    numeric_row = [float(item) for item in row]
                    data.append(numeric_row)
                except ValueError:
                    logger.warning(f"Skipping non-numeric row in {file_path}: {row}")
                    continue # Skip rows that can't be fully converted to float

        if not data:
            logger.warning(f"No numeric data found or processed in CSV file: {file_path}")
            return None, metadata

        tensor = torch.tensor(data, dtype=torch.float32)
        logger.debug(f"Successfully processed {file_path} into tensor shape {tensor.shape}")
        return tensor, metadata
    except Exception as e:
        logger.error(f"Failed to process CSV file {file_path}: {e}")
        return None, metadata


def preprocess_image(file_path: str) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
    """
    Basic Image preprocessor using Pillow and Torchvision transforms.
    Opens an image, applies standard transformations (resize, normalize),
    and returns it as a tensor.
    """
    metadata = {"source_file": file_path, "type": "image"}
    try:
        img = Image.open(file_path).convert('RGB') # Ensure 3 channels (RGB)

        # Example transform: Resize, convert to tensor, normalize
        # These should ideally be configurable
        transform = T.Compose([
            T.Resize((128, 128)), # Example fixed size
            T.ToTensor(), # Converts PIL image (H, W, C) [0,255] to Tensor (C, H, W) [0,1]
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet stats
        ])

        tensor = transform(img)
        metadata["original_size"] = img.size # (width, height)
        logger.debug(f"Successfully processed {file_path} into tensor shape {tensor.shape}")
        return tensor, metadata
    except FileNotFoundError:
         logger.error(f"Image file not found: {file_path}")
         return None, metadata
    except Exception as e:
        logger.error(f"Failed to process image file {file_path}: {e}")
        return None, metadata

# --- Data Ingestion Agent Class ---

class DataIngestionAgent:
    """
    Monitors a directory for new files and ingests them into TensorStorage.
    """

    def __init__(self,
                 tensor_storage: TensorStorage,
                 dataset_name: str,
                 source_directory: str,
                 polling_interval_sec: int = 10,
                 preprocessing_rules: Optional[Dict[str, Callable[[str], Tuple[Optional[torch.Tensor], Dict[str, Any]]]]] = None):
        """
        Initializes the DataIngestionAgent.

        Args:
            tensor_storage: An instance of the TensorStorage class.
            dataset_name: The name of the dataset in TensorStorage to ingest into.
            source_directory: The path to the local directory to monitor.
            polling_interval_sec: How often (in seconds) to check the directory.
            preprocessing_rules: A dictionary mapping lowercase file extensions
                                 (e.g., '.csv', '.png') to preprocessing functions.
                                 Each function takes a file path and returns a
                                 Tuple containing the processed Tensor (or None on failure)
                                 and a metadata dictionary. If None, default rules are used.
        """
        if not isinstance(tensor_storage, TensorStorage):
            raise TypeError("tensor_storage must be an instance of TensorStorage")
        if not os.path.isdir(source_directory):
            raise ValueError(f"Source directory '{source_directory}' does not exist or is not a directory.")

        self.tensor_storage = tensor_storage
        self.dataset_name = dataset_name
        self.source_directory = source_directory
        self.polling_interval = polling_interval_sec

        # Ensure dataset exists
        try:
            self.tensor_storage.get_dataset(self.dataset_name)
            logger.info(f"Agent targeting existing dataset '{self.dataset_name}'.")
        except ValueError:
            logger.info(f"Dataset '{self.dataset_name}' not found. Creating it.")
            self.tensor_storage.create_dataset(self.dataset_name)

        # Default preprocessing rules if none provided
        if preprocessing_rules is None:
            self.preprocessing_rules = {
                '.csv': preprocess_csv,
                '.png': preprocess_image,
                '.jpg': preprocess_image,
                '.jpeg': preprocess_image,
                '.tif': preprocess_image,
                '.tiff': preprocess_image,
            }
            logger.info("Using default preprocessing rules for CSV and common image formats.")
        else:
            self.preprocessing_rules = preprocessing_rules
            logger.info(f"Using custom preprocessing rules for extensions: {list(preprocessing_rules.keys())}")

        self.processed_files = set() # Keep track of files already processed in this session
        self._stop_event = threading.Event()
        self._monitor_thread = None
        self.status = "stopped" # Status reporting
        self.logs = collections.deque(maxlen=100) # Log capturing

        # Setup custom log handler
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        memory_handler = AgentMemoryLogHandler(self.logs)
        memory_handler.setFormatter(formatter)
        logger.addHandler(memory_handler) # Add handler to the specific logger instance

        logger.info(f"DataIngestionAgent initialized for dataset '{self.dataset_name}' monitoring '{self.source_directory}'.")

    def get_status(self) -> str:
        """Returns the current status of the agent."""
        return self.status

    def get_logs(self, max_lines: Optional[int] = None) -> List[str]:
        """Returns recent log messages from the agent."""
        if max_lines is None:
            return list(self.logs)
        else:
            return list(self.logs)[-max_lines:]

    def _validate_data(self, tensor: Optional[torch.Tensor], metadata: Dict[str, Any]) -> bool:
        """
        Performs basic validation on the preprocessed tensor.
        Returns True if valid, False otherwise.
        """
        if tensor is None:
            logger.warning(f"Validation failed: Tensor is None for {metadata.get('source_file', 'N/A')}")
            return False
        if not isinstance(tensor, torch.Tensor):
             logger.warning(f"Validation failed: Output is not a tensor for {metadata.get('source_file', 'N/A')}")
             return False
        # Add more specific checks if needed (e.g., tensor.numel() > 0)
        return True

    def _process_file(self, file_path: str) -> None:
        """Processes a single detected file."""
        logger.info(f"Detected new file: {file_path}")
        _, file_extension = os.path.splitext(file_path)
        file_extension = file_extension.lower()

        preprocessor = self.preprocessing_rules.get(file_extension)

        if preprocessor:
            logger.debug(f"Applying preprocessor for '{file_extension}' to {file_path}")
            try:
                tensor, metadata = preprocessor(file_path)

                if self._validate_data(tensor, metadata):
                    # Ensure tensor is not None before insertion
                    if tensor is not None:
                        metadata["created_by"] = "IngestionAgent" # Add agent source
                        record_id = self.tensor_storage.insert(self.dataset_name, tensor, metadata)
                        logger.info(f"Successfully ingested '{file_path}' into dataset '{self.dataset_name}' with record ID: {record_id} (created_by: IngestionAgent)")
                        self.processed_files.add(file_path) # Mark as processed only on success
                    else:
                         # Should have been caught by validation, but as safeguard:
                         logger.error(f"Validation passed but tensor is None for {file_path}. Skipping insertion.")

                else:
                    logger.warning(f"Data validation failed for {file_path}. Skipping insertion.")

            except Exception as e:
                logger.error(f"Unhandled error during preprocessing or insertion for {file_path}: {e}", exc_info=True)
        else:
            logger.debug(f"No preprocessor configured for file extension '{file_extension}'. Skipping file: {file_path}")


    def _scan_source_directory(self) -> None:
        """Scans the source directory for new files matching the rules."""
        logger.debug(f"Scanning directory: {self.source_directory}")
        supported_extensions = self.preprocessing_rules.keys()

        try:
             # Use glob to find all files, then filter
             # This might be inefficient for huge directories, consider os.scandir
             all_files = glob.glob(os.path.join(self.source_directory, '*'), recursive=False) # Non-recursive

             for file_path in all_files:
                 if not os.path.isfile(file_path):
                      continue # Skip directories

                 _, file_extension = os.path.splitext(file_path)
                 file_extension = file_extension.lower()

                 if file_extension in supported_extensions and file_path not in self.processed_files:
                     self._process_file(file_path)

        except Exception as e:
             logger.error(f"Error scanning source directory '{self.source_directory}': {e}", exc_info=True)


    def _monitor_loop(self) -> None:
        """The main loop executed by the background thread."""
        logger.info(f"Starting monitoring loop for '{self.source_directory}'. Polling interval: {self.polling_interval} seconds.")
        while not self._stop_event.is_set():
            self._scan_source_directory()
            # Wait for the specified interval or until stop event is set
            self._stop_event.wait(self.polling_interval)
        logger.info("Monitoring loop stopped.")


    def start(self) -> None:
        """Starts the monitoring process in a background thread."""
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            logger.warning("Agent monitoring is already running.")
            return

        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        if self._monitor_thread.is_alive(): # Check if thread actually started
            self.status = "running"
            logger.info("Data Ingestion Agent started monitoring.")
        else:
            self.status = "error" # Or some other appropriate error state
            logger.error("Data Ingestion Agent failed to start monitoring thread.")


    def stop(self) -> None:
        """Signals the monitoring thread to stop."""
        if self._monitor_thread is None or not self._monitor_thread.is_alive():
            logger.info("Agent monitoring is not running.")
            return

        logger.info("Stopping Data Ingestion Agent monitoring...")
        self._stop_event.set()
        self._monitor_thread.join(timeout=self.polling_interval + 5) # Wait for thread to finish

        if self._monitor_thread.is_alive():
             logger.warning("Monitoring thread did not stop gracefully after timeout.")
             # self.status remains "running" or could be set to "stopping_error"
        else:
             logger.info("Data Ingestion Agent monitoring stopped successfully.")
             self.status = "stopped"
        self._monitor_thread = None


# --- Example Usage ---
if __name__ == "__main__":
    run_example = os.getenv("RUN_INGESTION_AGENT_EXAMPLE", "False").lower() == "true"

    if run_example:
        logger.info("--- Starting Ingestion Agent Example (RUN_INGESTION_AGENT_EXAMPLE=True) ---")

        # 1. Setup TensorStorage
    storage = TensorStorage()

    # 2. Setup a temporary directory for the agent to monitor
    source_dir = "temp_ingestion_source"
    if not os.path.exists(source_dir):
        os.makedirs(source_dir)
        logger.info(f"Created temporary source directory: {source_dir}")

    # 3. Create the Ingestion Agent
    # We'll use a short polling interval for demonstration
    agent = DataIngestionAgent(
        tensor_storage=storage,
        dataset_name="raw_data",
        source_directory=source_dir,
        polling_interval_sec=5
    )

    # 4. Start the agent (runs in the background)
    agent.start()

    # 5. Simulate adding files to the source directory
    print("\nSimulating file creation...")
    time.sleep(2) # Give agent time to start initial scan

    # Create a dummy CSV file
    csv_path = os.path.join(source_dir, "data_1.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Value1", "Value2"])
        writer.writerow(["1678886400", "10.5", "20.1"])
        writer.writerow(["1678886460", "11.2", "20.5"])
        writer.writerow(["1678886520", "invalid", "20.9"]) # Test non-numeric row
        writer.writerow(["1678886580", "10.9", "21.0"])
    print(f"Created CSV: {csv_path}")

    # Create a dummy image file (requires Pillow)
    try:
        img_path = os.path.join(source_dir, "image_1.png")
        dummy_img = Image.new('RGB', (60, 30), color = 'red')
        dummy_img.save(img_path)
        print(f"Created Image: {img_path}")
    except ImportError:
        print("Pillow not installed, skipping image creation. Install with: pip install Pillow")
    except Exception as e:
        print(f"Could not create dummy image: {e}")


    # Create an unsupported file type
    txt_path = os.path.join(source_dir, "notes.txt")
    with open(txt_path, 'w') as f:
        f.write("This is a test file.")
    print(f"Created TXT: {txt_path} (should be skipped)")


    # 6. Let the agent run for a couple of polling cycles
    print(f"\nWaiting for agent to process files (polling interval {agent.polling_interval}s)...")
    time.sleep(agent.polling_interval * 2 + 1) # Wait for 2 cycles + buffer


    # 7. Check the contents of TensorStorage
    print("\n--- Checking TensorStorage contents ---")
    try:
        ingested_data = storage.get_dataset_with_metadata("raw_data")
        print(f"Found {len(ingested_data)} items in dataset 'raw_data':")
        for item in ingested_data:
            print(f"  Record ID: {item['metadata'].get('record_id')}, Source: {item['metadata'].get('source_file')}, Shape: {item['tensor'].shape}, Dtype: {item['tensor'].dtype}")
            # print(f" Tensor: {item['tensor']}") # Can be verbose
    except ValueError as e:
        print(f"Could not retrieve dataset 'raw_data': {e}")


    # 8. Stop the agent
    print("\n--- Stopping Agent ---")
    agent.stop()

    # 9. Clean up the temporary directory (optional)
    # print(f"\nCleaning up temporary directory: {source_dir}")
    # import shutil
    # shutil.rmtree(source_dir)

    logger.info("--- Ingestion Agent Example Finished ---")
else:
    logger.info("--- Ingestion Agent Example SKIPPED (RUN_INGESTION_AGENT_EXAMPLE not set to 'true') ---")
