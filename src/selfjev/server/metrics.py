"""A few Prometheus counters in the text exposition format, without a dependency."""

from collections import Counter, defaultdict


class Metrics:
    def __init__(self):
        self.requests, self.seconds, self.count = Counter(), defaultdict(float), Counter()
        self.input_tokens = self.questions = 0

    def observe(self, route: str, status: int, seconds: float):
        self.requests[route, status] += 1
        self.seconds[route] += seconds
        self.count[route] += 1

    def add_usage(self, input_tokens: int, questions: int):
        self.input_tokens += input_tokens
        self.questions += questions

    def render(self, queue_depth: int) -> str:
        lines = ["# TYPE selfjev_requests_total counter"]
        lines += [f'selfjev_requests_total{{route="{r}",status="{s}"}} {n}' for (r, s), n in sorted(self.requests.items())]
        lines += ["# TYPE selfjev_request_seconds summary"]
        for r in sorted(self.count):
            lines += [
                f'selfjev_request_seconds_sum{{route="{r}"}} {self.seconds[r]:.6f}',
                f'selfjev_request_seconds_count{{route="{r}"}} {self.count[r]}',
            ]
        lines += ["# TYPE selfjev_input_tokens_total counter", f"selfjev_input_tokens_total {self.input_tokens}"]
        lines += ["# TYPE selfjev_questions_total counter", f"selfjev_questions_total {self.questions}"]
        lines += ["# TYPE selfjev_queue_depth gauge", f"selfjev_queue_depth {queue_depth}"]
        return "\n".join(lines) + "\n"
