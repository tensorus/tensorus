import hashlib
from tensorus.arrays import canonical_json
MAX_REPORT_BYTES = 1024**2
MAX_QUALITY_RELEASE_RETIREMENTS_PER_TENANT = 1000
def _hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()
