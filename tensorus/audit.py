if __package__ in (None, ""):
    import os, sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "tensorus"

import logging
from typing import Optional, Dict, Any
import sys
from tensorus.config import settings

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

audit_logger = logging.getLogger("tensorus.audit")
audit_logger.setLevel(logging.INFO)

try:
    file_handler = logging.FileHandler(settings.AUDIT_LOG_PATH)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    audit_logger.addHandler(file_handler)
except IOError:
    print(
        f"Warning: Could not open {settings.AUDIT_LOG_PATH} for writing. Audit logs will go to stdout only.",
        file=sys.stderr,
    )


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
