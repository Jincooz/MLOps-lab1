from prometheus_client import Counter, Histogram, Gauge

REQUEST_LATENCY = Histogram(
    "db_request_latency_seconds",
    "Request latency",
    ["service", "operation"],
    buckets=(0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 30, 60, 120)
)

REQUESTS_TOTAL = Counter(
    "db_requests_total",
    "Total requests",
    ["service", "operation"]
)

REQUEST_ERRORS = Counter(
    "db_request_errors_total",
    "Request errors",
    ["service", "error_type"]
)

NODE_UP = Gauge(
    "node_up",
    "Node health",
    ["service", "role"]
)
