import numpy as np
import logging
import time
import uuid
import hashlib
import json
import os
import jwt
import secrets
from functools import wraps
from typing import Dict, List, Any, Tuple, Optional, Union, Callable
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TensorSecurity:
    """
    Security and governance layer for Tensorus.
    Handles authentication, authorization, and audit logging.
    """
    
    def __init__(self, 
                 config_path: Optional[str] = "config/security_config.json",
                 secret_key: Optional[str] = None,
                 token_expiry: int = 3600,  # 1 hour
                 enable_audit_log: bool = True):
        """
        Initialize the security layer.
        
        Args:
            config_path: Path to security configuration file
            secret_key: Secret key for JWT token generation
            token_expiry: Token expiration time in seconds
            enable_audit_log: Whether to enable audit logging
        """
        self.config = self._load_config(config_path)
        self.secret_key = secret_key or self.config.get("secret_key") or secrets.token_hex(32)
        self.token_expiry = token_expiry or self.config.get("token_expiry", 3600)
        self.enable_audit_log = enable_audit_log
        
        # User and role storage
        self.users: Dict[str, Dict[str, Any]] = self.config.get("users", {})
        self.roles: Dict[str, Dict[str, Any]] = self.config.get("roles", {})
        
        # Resources and permissions
        self.resources: Dict[str, Dict[str, Any]] = self.config.get("resources", {})
        
        # Token blacklist (for logout)
        self.token_blacklist: List[str] = []
        
        # Audit log
        self.audit_log_path = self.config.get("audit_log_path", "logs/audit.log")
        if self.enable_audit_log:
            os.makedirs(os.path.dirname(self.audit_log_path), exist_ok=True)
        
        logger.info("TensorSecurity initialized")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load security configuration from JSON file or use defaults."""
        default_config = {
            "users": {
                "admin": {
                    "password_hash": self._hash_password("admin"),
                    "roles": ["admin"],
                    "active": True
                }
            },
            "roles": {
                "admin": {
                    "permissions": ["*"],
                    "description": "Administrator with full access"
                },
                "reader": {
                    "permissions": ["tensor:read", "tensor:search"],
                    "description": "Read-only access to tensors"
                },
                "writer": {
                    "permissions": ["tensor:read", "tensor:write", "tensor:search", "tensor:process"],
                    "description": "Read-write access to tensors"
                }
            },
            "resources": {
                "tensor": {
                    "operations": ["read", "write", "delete", "search", "process"]
                },
                "user": {
                    "operations": ["read", "create", "update", "delete"]
                },
                "system": {
                    "operations": ["configure", "monitor", "backup"]
                }
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                logger.info(f"Loaded security configuration from {config_path}")
                return {**default_config, **config}  # Merge with defaults
            except Exception as e:
                logger.warning(f"Error loading security config: {e}. Using defaults.")
        
        logger.info("Using default security configuration")
        return default_config
    
    def _hash_password(self, password: str) -> str:
        """
        Hash a password using SHA-256.
        
        Args:
            password: Plain text password
            
        Returns:
            Hashed password
        """
        return hashlib.sha256(password.encode()).hexdigest()
    
    def _generate_token(self, user_id: str, expiry: Optional[int] = None) -> str:
        """
        Generate a JWT token for a user.
        
        Args:
            user_id: User ID
            expiry: Token expiration time in seconds
            
        Returns:
            JWT token
        """
        expiry = expiry or self.token_expiry
        payload = {
            "user_id": user_id,
            "exp": datetime.utcnow() + timedelta(seconds=expiry),
            "iat": datetime.utcnow(),
            "jti": str(uuid.uuid4())
        }
        
        # Add user roles to the token
        if user_id in self.users:
            payload["roles"] = self.users[user_id].get("roles", [])
        
        return jwt.encode(payload, self.secret_key, algorithm="HS256")
    
    def _verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verify and decode a JWT token.
        
        Args:
            token: JWT token
            
        Returns:
            Decoded payload or None if invalid
        """
        if token in self.token_blacklist:
            return None
        
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            return payload
        except jwt.PyJWTError as e:
            logger.warning(f"Token verification failed: {e}")
            return None
    
    def authenticate(self, username: str, password: str) -> Optional[str]:
        """
        Authenticate a user and generate a token.
        
        Args:
            username: Username
            password: Password
            
        Returns:
            JWT token or None if authentication fails
        """
        if username not in self.users:
            logger.warning(f"Authentication failed: User {username} not found")
            return None
        
        user = self.users[username]
        
        if not user.get("active", True):
            logger.warning(f"Authentication failed: User {username} is inactive")
            return None
        
        if self._hash_password(password) != user.get("password_hash"):
            logger.warning(f"Authentication failed: Invalid password for user {username}")
            return None
        
        token = self._generate_token(username)
        
        # Log the successful authentication
        self.log_audit_event(username, "authentication", "success", 
                          {"ip": "system"})
        
        return token
    
    def logout(self, token: str) -> bool:
        """
        Invalidate a token by adding it to the blacklist.
        
        Args:
            token: JWT token
            
        Returns:
            True if logout was successful
        """
        payload = self._verify_token(token)
        if payload:
            self.token_blacklist.append(token)
            self.log_audit_event(payload.get("user_id", "unknown"), "logout", "success", 
                             {"token_id": payload.get("jti")})
            return True
        
        return False
    
    def authorize(self, token: str, resource: str, operation: str, resource_id: Optional[str] = None) -> bool:
        """
        Check if a user has permission to perform an operation on a resource.
        
        Args:
            token: JWT token
            resource: Resource type (e.g., "tensor")
            operation: Operation (e.g., "read")
            resource_id: Optional specific resource ID
            
        Returns:
            True if authorized
        """
        payload = self._verify_token(token)
        if not payload:
            return False
        
        user_id = payload.get("user_id")
        roles = payload.get("roles", [])
        
        # Check permissions for each role
        for role_name in roles:
            if role_name not in self.roles:
                continue
            
            role = self.roles[role_name]
            permissions = role.get("permissions", [])
            
            # Check for wildcard permission
            if "*" in permissions:
                return True
            
            # Check for resource:operation permission
            resource_permission = f"{resource}:{operation}"
            if resource_permission in permissions:
                return True
            
            # Check for resource:* permission
            resource_wildcard = f"{resource}:*"
            if resource_wildcard in permissions:
                return True
        
        self.log_audit_event(user_id, "authorization", "failure", 
                         {"resource": resource, "operation": operation, "resource_id": resource_id})
        
        return False
    
    def create_user(self, admin_token: str, username: str, password: str, roles: List[str]) -> bool:
        """
        Create a new user.
        
        Args:
            admin_token: Admin JWT token
            username: New user's username
            password: New user's password
            roles: List of roles to assign
            
        Returns:
            True if user was created successfully
        """
        # Check admin authorization
        if not self.authorize(admin_token, "user", "create"):
            return False
        
        # Check if user already exists
        if username in self.users:
            logger.warning(f"User creation failed: User {username} already exists")
            return False
        
        # Check if roles exist
        for role in roles:
            if role not in self.roles:
                logger.warning(f"User creation failed: Role {role} does not exist")
                return False
        
        # Create user
        self.users[username] = {
            "password_hash": self._hash_password(password),
            "roles": roles,
            "active": True,
            "created_at": time.time()
        }
        
        # Log the user creation
        admin_payload = self._verify_token(admin_token)
        admin_user = admin_payload.get("user_id") if admin_payload else "unknown"
        
        self.log_audit_event(admin_user, "user_creation", "success", 
                         {"created_user": username, "roles": roles})
        
        return True
    
    def update_user(self, admin_token: str, username: str, 
                   password: Optional[str] = None, 
                   roles: Optional[List[str]] = None,
                   active: Optional[bool] = None) -> bool:
        """
        Update an existing user.
        
        Args:
            admin_token: Admin JWT token
            username: Username to update
            password: New password (if changing)
            roles: New roles (if changing)
            active: New active status (if changing)
            
        Returns:
            True if user was updated successfully
        """
        # Check admin authorization
        if not self.authorize(admin_token, "user", "update"):
            return False
        
        # Check if user exists
        if username not in self.users:
            logger.warning(f"User update failed: User {username} does not exist")
            return False
        
        # Update user
        if password:
            self.users[username]["password_hash"] = self._hash_password(password)
        
        if roles is not None:
            # Check if roles exist
            for role in roles:
                if role not in self.roles:
                    logger.warning(f"User update failed: Role {role} does not exist")
                    return False
            
            self.users[username]["roles"] = roles
        
        if active is not None:
            self.users[username]["active"] = active
        
        # Log the user update
        admin_payload = self._verify_token(admin_token)
        admin_user = admin_payload.get("user_id") if admin_payload else "unknown"
        
        update_info = {}
        if password:
            update_info["password_changed"] = True
        if roles is not None:
            update_info["roles"] = roles
        if active is not None:
            update_info["active"] = active
        
        self.log_audit_event(admin_user, "user_update", "success", 
                         {"updated_user": username, **update_info})
        
        return True
    
    def delete_user(self, admin_token: str, username: str) -> bool:
        """
        Delete a user.
        
        Args:
            admin_token: Admin JWT token
            username: Username to delete
            
        Returns:
            True if user was deleted successfully
        """
        # Check admin authorization
        if not self.authorize(admin_token, "user", "delete"):
            return False
        
        # Check if user exists
        if username not in self.users:
            logger.warning(f"User deletion failed: User {username} does not exist")
            return False
        
        # Delete user
        del self.users[username]
        
        # Log the user deletion
        admin_payload = self._verify_token(admin_token)
        admin_user = admin_payload.get("user_id") if admin_payload else "unknown"
        
        self.log_audit_event(admin_user, "user_deletion", "success", 
                         {"deleted_user": username})
        
        return True
    
    def create_role(self, admin_token: str, role_name: str, permissions: List[str], description: str) -> bool:
        """
        Create a new role.
        
        Args:
            admin_token: Admin JWT token
            role_name: Name of the new role
            permissions: List of permissions
            description: Role description
            
        Returns:
            True if role was created successfully
        """
        # Check admin authorization
        if not self.authorize(admin_token, "user", "create"):
            return False
        
        # Check if role already exists
        if role_name in self.roles:
            logger.warning(f"Role creation failed: Role {role_name} already exists")
            return False
        
        # Create role
        self.roles[role_name] = {
            "permissions": permissions,
            "description": description,
            "created_at": time.time()
        }
        
        # Log the role creation
        admin_payload = self._verify_token(admin_token)
        admin_user = admin_payload.get("user_id") if admin_payload else "unknown"
        
        self.log_audit_event(admin_user, "role_creation", "success", 
                         {"role": role_name, "permissions": permissions})
        
        return True
    
    def log_audit_event(self, user_id: str, action: str, outcome: str, details: Dict[str, Any] = None):
        """
        Log an audit event.
        
        Args:
            user_id: ID of the user performing the action
            action: Action being performed
            outcome: Outcome (success/failure)
            details: Additional details
        """
        if not self.enable_audit_log:
            return
        
        timestamp = datetime.utcnow().isoformat()
        event = {
            "timestamp": timestamp,
            "user_id": user_id,
            "action": action,
            "outcome": outcome,
            "details": details or {}
        }
        
        try:
            with open(self.audit_log_path, 'a') as f:
                f.write(json.dumps(event) + "\n")
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")
    
    def save_config(self, config_path: Optional[str] = None):
        """
        Save the current configuration to a file.
        
        Args:
            config_path: Path to save the configuration (defaults to original path)
        """
        config_path = config_path or self.config.get("config_path")
        if not config_path:
            logger.warning("Cannot save configuration: No path specified")
            return
        
        # Create config directory if it doesn't exist
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # Prepare configuration
        config = {
            "users": self.users,
            "roles": self.roles,
            "resources": self.resources,
            "token_expiry": self.token_expiry,
            "audit_log_path": self.audit_log_path
        }
        
        try:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"Saved security configuration to {config_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")

class TensorDifferentialPrivacy:
    """
    Implements differential privacy mechanisms for tensor operations.
    """
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5, mechanism: str = "laplace"):
        """
        Initialize differential privacy.
        
        Args:
            epsilon: Privacy budget (smaller = more privacy)
            delta: Probability of privacy violation
            mechanism: Privacy mechanism ("laplace", "gaussian", "exponential")
        """
        self.epsilon = epsilon
        self.delta = delta
        self.mechanism = mechanism
        self.remaining_budget = epsilon
        
        logger.info(f"TensorDifferentialPrivacy initialized with epsilon={epsilon}, delta={delta}, mechanism={mechanism}")
    
    def privatize_tensor(self, tensor: np.ndarray, sensitivity: float, 
                        mechanism: Optional[str] = None) -> np.ndarray:
        """
        Apply differential privacy to a tensor.
        
        Args:
            tensor: Input tensor
            sensitivity: Maximum difference that a single record can make
            mechanism: Privacy mechanism (overrides instance default)
            
        Returns:
            Privatized tensor
        """
        mechanism = mechanism or self.mechanism
        
        if self.remaining_budget <= 0:
            logger.warning("Privacy budget exhausted, cannot privatize tensor")
            return tensor
        
        # Determine how much budget to use for this operation
        # Simple strategy: use 10% of remaining budget
        budget_to_use = min(0.1 * self.remaining_budget, self.remaining_budget)
        
        if mechanism == "laplace":
            private_tensor = self._laplace_mechanism(tensor, sensitivity, budget_to_use)
        elif mechanism == "gaussian":
            private_tensor = self._gaussian_mechanism(tensor, sensitivity, budget_to_use)
        elif mechanism == "exponential":
            private_tensor = self._exponential_mechanism(tensor, sensitivity, budget_to_use)
        else:
            logger.warning(f"Unknown privacy mechanism: {mechanism}. Using Laplace.")
            private_tensor = self._laplace_mechanism(tensor, sensitivity, budget_to_use)
        
        # Update remaining budget
        self.remaining_budget -= budget_to_use
        
        return private_tensor
    
    def _laplace_mechanism(self, tensor: np.ndarray, sensitivity: float, epsilon: float) -> np.ndarray:
        """
        Apply Laplace mechanism.
        
        Args:
            tensor: Input tensor
            sensitivity: Maximum difference that a single record can make
            epsilon: Privacy budget for this operation
            
        Returns:
            Privatized tensor
        """
        scale = sensitivity / epsilon
        noise = np.random.laplace(0, scale, tensor.shape)
        return tensor + noise
    
    def _gaussian_mechanism(self, tensor: np.ndarray, sensitivity: float, epsilon: float) -> np.ndarray:
        """
        Apply Gaussian mechanism.
        
        Args:
            tensor: Input tensor
            sensitivity: Maximum difference that a single record can make
            epsilon: Privacy budget for this operation
            
        Returns:
            Privatized tensor
        """
        # Calculate sigma based on epsilon, delta
        sigma = np.sqrt(2 * np.log(1.25 / self.delta)) * sensitivity / epsilon
        noise = np.random.normal(0, sigma, tensor.shape)
        return tensor + noise
    
    def _exponential_mechanism(self, tensor: np.ndarray, sensitivity: float, epsilon: float) -> np.ndarray:
        """
        Apply Exponential mechanism.
        Note: This is a simplified version for numerical data.
        
        Args:
            tensor: Input tensor
            sensitivity: Maximum difference that a single record can make
            epsilon: Privacy budget for this operation
            
        Returns:
            Privatized tensor
        """
        # For simplicity, we'll treat this as a variant of Laplace mechanism
        # with a different scale
        scale = 2 * sensitivity / epsilon
        noise = np.random.laplace(0, scale, tensor.shape)
        return tensor + noise
    
    def privatize_query(self, database_ref, query_func: Callable, 
                       sensitivity: float, *args, **kwargs) -> Any:
        """
        Execute a query with differential privacy.
        
        Args:
            database_ref: Reference to the database
            query_func: Function that executes the query
            sensitivity: Maximum difference that a single record can make
            *args, **kwargs: Arguments for the query function
            
        Returns:
            Query result with differential privacy applied
        """
        # Execute the query
        result = query_func(*args, **kwargs)
        
        # Apply privacy mechanism to the result
        if isinstance(result, np.ndarray):
            return self.privatize_tensor(result, sensitivity)
        elif isinstance(result, (list, tuple)) and all(isinstance(r, np.ndarray) for r in result):
            return [self.privatize_tensor(r, sensitivity) for r in result]
        else:
            logger.warning("Cannot privatize non-tensor result")
            return result
    
    def reset_budget(self, epsilon: Optional[float] = None):
        """
        Reset the privacy budget.
        
        Args:
            epsilon: New epsilon value (or use existing)
        """
        self.epsilon = epsilon or self.epsilon
        self.remaining_budget = self.epsilon
        logger.info(f"Reset privacy budget to {self.epsilon}")
    
    def get_budget_status(self) -> Dict[str, float]:
        """
        Get the current privacy budget status.
        
        Returns:
            Dictionary with budget information
        """
        return {
            "epsilon": self.epsilon,
            "delta": self.delta,
            "remaining_budget": self.remaining_budget,
            "used_budget": self.epsilon - self.remaining_budget,
            "percent_remaining": (self.remaining_budget / self.epsilon) * 100
        }

def secure_operation(resource: str, operation: str):
    """
    Decorator for securing operations with authentication and authorization.
    
    Args:
        resource: Resource type (e.g., "tensor")
        operation: Operation (e.g., "read")
    
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Extract token from the first argument or from kwargs
            token = kwargs.pop("token", None)
            if token is None and len(args) > 0 and isinstance(args[0], str):
                token = args[0]
                args = args[1:]
            
            if not hasattr(self, "security") or not self.security:
                logger.warning("No security layer found, operation not secured")
                return func(self, *args, **kwargs)
            
            # Check authorization
            if not self.security.authorize(token, resource, operation):
                raise PermissionError(f"Not authorized for {resource}:{operation}")
            
            # Extract user info from token
            payload = self.security._verify_token(token)
            user_id = payload.get("user_id") if payload else "unknown"
            
            # Log the operation
            self.security.log_audit_event(
                user_id, 
                f"{resource}_{operation}", 
                "attempt", 
                {"args": str(args), "kwargs": str(kwargs)}
            )
            
            # Execute the function
            try:
                result = func(self, *args, **kwargs)
                
                # Log success
                self.security.log_audit_event(
                    user_id, 
                    f"{resource}_{operation}", 
                    "success", 
                    {}
                )
                
                return result
            except Exception as e:
                # Log failure
                self.security.log_audit_event(
                    user_id, 
                    f"{resource}_{operation}", 
                    "failure", 
                    {"error": str(e)}
                )
                
                raise
        
        return wrapper
    
    return decorator 