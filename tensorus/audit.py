import logging
from typing import Optional, Dict, Any
import sys
from tensorus.config import settings

# Configure basic logger
# In a real app, this might be more complex (e.g., JSON logging, log rotation, external service)
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DEFAULT_HANDLERS = [logging.StreamHandler(sys.stdout)] # Log to stdout by default

# Attempt to create a log file handler.
# This is a simple file logger; in production, consider more robust solutions.
try:
    file_handler = logging.FileHandler(settings.AUDIT_LOG_PATH)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    LOG_DEFAULT_HANDLERS.append(file_handler)
except IOError:
    # Handle cases where file cannot be opened (e.g. permissions)
    print(
        f"Warning: Could not open {settings.AUDIT_LOG_PATH} for writing. Audit logs will go to stdout only.",
        file=sys.stderr,
    )


logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, handlers=LOG_DEFAULT_HANDLERS)
audit_logger = logging.getLogger("tensorus.audit")


def log_audit_event(
    action: str,
    user: Optional[str] = "anonymous",
    tensor_id: Optional[str] = None, # Make tensor_id a common, optional parameter
    details: Optional[Dict[str, Any]] = None
):
    """
    Logs an audit event.

    Args:
        action: A string describing the action performed (e.g., "CREATE_TENSOR_DESCRIPTOR").
        user: The user or API key performing the action. Defaults to "anonymous".
        tensor_id: The primary tensor_id involved in the action, if applicable.
        details: A dictionary of additional relevant information about the event.
    """
    log_message_parts = [f"Action: {action}"]

    log_message_parts.append(f"User: {user if user else 'unknown'}")

    if tensor_id:
        log_message_parts.append(f"TensorID: {tensor_id}")

    if details:
        # Convert details dict to a string format, e.g., key1=value1, key2=value2
        details_str = ", ".join([f"{k}={v}" for k, v in details.items()])
        log_message_parts.append(f"Details: [{details_str}]")

    audit_logger.info(" | ".join(log_message_parts))


# Example Usage (not part of the library code, just for demonstration):
if __name__ == "__main__":
    # This will only run if the script is executed directly
    log_audit_event(action="TEST_EVENT", user="test_user", tensor_id="dummy-uuid-123", details={"param1": "value1", "status": "success"})
    audit_logger.info("This is a direct info log from audit_logger for testing handlers.")
    audit_logger.warning("This is a warning log for testing handlers.")
    print(f"Audit logger '{audit_logger.name}' has handlers: {audit_logger.handlers}")
    if not audit_logger.handlers:
         print("Warning: Audit logger has no handlers configured if run outside main app context without direct basicConfig call here.")
    # In the application, basicConfig in this file should set up the handlers globally once.
