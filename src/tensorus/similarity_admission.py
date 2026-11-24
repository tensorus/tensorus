"""Single-process admission for HTTP similarity searches."""
from tensorus.request_work_admission import MAX_REQUEST_WORK_SLOTS
MAX_SLOTS = MAX_REQUEST_WORK_SLOTS
ROUTES = ("/v1/search", "/v1/search/snapshot", "/v1/search/snapshot/validity")
