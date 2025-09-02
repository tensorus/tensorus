"""
Tensorus API Key Generator - Industry standard API key generation and validation.

Following patterns from OpenAI, Pinecone, and other major AI/ML services.
"""

import secrets
import string
from typing import Optional


class TensorusAPIKey:
    """Industry-standard API key format for Tensorus"""
    
    PREFIX = "tsr_"  # Short, memorable prefix like OpenAI's "sk-"
    LENGTH = 48      # Standard length for security (excluding prefix)
    
    @classmethod
    def generate(cls) -> str:
        """
        Generate a secure API key following industry standards.
        
        Returns:
            str: A secure API key in format 'tsr_<48-random-chars>'
            
        Example:
            tsr_abc123XYZ789-_def456GHI012jkl345MNO678pqr901STU234
        """
        # Use URL-safe characters (like Pinecone/OpenAI)
        alphabet = string.ascii_letters + string.digits + "-_"
        random_part = ''.join(secrets.choice(alphabet) for _ in range(cls.LENGTH))
        return f"{cls.PREFIX}{random_part}"
    
    @classmethod
    def validate_format(cls, key: str) -> bool:
        """
        Validate API key format without checking if it's authorized.
        
        Args:
            key: The API key to validate
            
        Returns:
            bool: True if format is valid, False otherwise
        """
        if not key or not isinstance(key, str):
            return False
            
        if not key.startswith(cls.PREFIX):
            return False
            
        if len(key) != len(cls.PREFIX) + cls.LENGTH:
            return False
            
        # Check that the part after prefix contains only valid characters
        key_body = key[len(cls.PREFIX):]
        valid_chars = string.ascii_letters + string.digits + "-_"
        return all(c in valid_chars for c in key_body)
    
    @classmethod
    def mask_key(cls, key: str) -> str:
        """
        Mask an API key for safe logging/display.
        
        Args:
            key: The API key to mask
            
        Returns:
            str: Masked key showing only prefix and last 4 characters
            
        Example:
            tsr_abc123...STU234
        """
        if not cls.validate_format(key):
            return "invalid_key"
        
        if len(key) <= 8:
            return key  # Too short to mask meaningfully
            
        return f"{key[:7]}...{key[-4:]}"


def generate_api_key() -> str:
    """
    Convenience function to generate a new API key.
    
    Returns:
        str: A new secure API key
    """
    return TensorusAPIKey.generate()


if __name__ == "__main__":
    # CLI utility for development
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--count":
        try:
            count = int(sys.argv[2])
        except (IndexError, ValueError):
            count = 1
        
        print(f"Generated {count} API key(s):")
        for i in range(count):
            key = generate_api_key()
            print(f"{i+1}. {key}")
    else:
        key = generate_api_key()
        print(f"Generated API key: {key}")
        print(f"Masked format: {TensorusAPIKey.mask_key(key)}")
        print(f"Valid format: {TensorusAPIKey.validate_format(key)}")