"""Local HTTP server. Two routes, one model instance:

  POST /classify               our native schema (see examples/request.json)
  POST /api/alpha/decisions    request/response *shape* compatible with the decisions API used via OpenRouter
                               (questions: {id: {type: noul | choice | score, instructions, criteria}}).

Shape compatibility only: this serves our own model, so numbers differ from any hosted service and must not
be mixed with them. Mapping (ours, documented, not a claim about how the hosted model works):
  choice  criteria {label: description} -> multiclass over "label: description"; returns choice + probabilities
  noul    criteria {"true": d1, "false": d2} -> 2-way choice between d1 and d2, noul = P(true);
          without criteria -> our binary yes/no, noul = p_yes
  score   criteria [level_0, ..., level_k] -> distribution over ordered levels,
          score = sum_i p_i * i / k  (expected level scaled to [0, 1])
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .classify import classify, run_meta
from .options import with_options
from .formatting import DEFAULT_PROMPT
from .schemas import ValidationError

COMPAT_TYPES = ("noul", "choice", "score")


def _label_text(label: str, description: str) -> str:
    return f"{label.replace('_', ' ')}: {description}"


def compat_to_request(body: dict) -> tuple[dict, dict]:
    """-> (native request, per-question decoding info). Raises ValidationError with a client-facing message."""
    if not isinstance(body, dict):
        raise ValidationError("body must be a JSON object")
    qs = body.get("questions")
    if not isinstance(qs, dict) or not qs:
        raise ValidationError("'questions' must be a non-empty object keyed by question id")
    native, decode = [], {}
    for qid, q in qs.items():
        where = f"question '{qid}'"
        if not isinstance(q, dict) or q.get("type") not in COMPAT_TYPES:
            raise ValidationError(f"{where}: 'type' must be one of {COMPAT_TYPES}")
        instr, crit = q.get("instructions"), q.get("criteria")
        if not isinstance(instr, str) or not instr.strip():
            raise ValidationError(f"{where}: 'instructions' must be a non-empty string")
        if q["type"] == "choice":
            if not isinstance(crit, dict) or len(crit) < 2 or not all(isinstance(v, str) and v.strip() for v in crit.values()):
                raise ValidationError(f"{where}: choice 'criteria' must map at least 2 labels to non-empty descriptions")
            native.append({"id": qid, "type": "multiclass", "instruction": instr,
                           "candidates": [{"id": k, "description": _label_text(k, v)} for k, v in crit.items()]})
        elif q["type"] == "score":
            if not isinstance(crit, list) or len(crit) < 2 or not all(isinstance(v, str) and v.strip() for v in crit):
                raise ValidationError(f"{where}: score 'criteria' must be a list of at least 2 ordered level descriptions")
            if len(set(crit)) != len(crit):
                raise ValidationError(f"{where}: score levels must be distinct")
            native.append({"id": qid, "type": "multiclass", "instruction": instr,
                           "candidates": [{"id": str(i), "description": v} for i, v in enumerate(crit)]})
        elif crit is None:
            native.append({"id": qid, "type": "binary", "instruction": instr})
        else:
            if not isinstance(crit, dict) or set(crit) != {"true", "false"} or not all(isinstance(v, str) and v.strip() for v in crit.values()):
                raise ValidationError(f"{where}: noul 'criteria' must be {{'true': ..., 'false': ...}} with non-empty descriptions")
            native.append({"id": qid, "type": "multiclass", "instruction": instr,
                           "candidates": [{"id": "true", "description": crit["true"]}, {"id": "false", "description": crit["false"]}]})
        decode[qid] = (q["type"], crit)
    return {"state": body.get("state"), "questions": native}, decode


def answers_from(result: dict, decode: dict) -> dict:
    out = {}
    for r in result["questions"]:
        kind, crit = decode[r["id"]]
        if kind == "noul":
            p = r["p_yes"] if r["type"] == "binary" else next(c["probability"] for c in r["candidates"] if c["id"] == "true")
            out[r["id"]] = {"noul": p}
        elif kind == "choice":
            out[r["id"]] = {"choice": r["selected"], "probabilities": {c["id"]: c["probability"] for c in r["candidates"]}}
        else:
            probs = [c["probability"] for c in r["candidates"]]
            out[r["id"]] = {"score": sum(i * p for i, p in enumerate(probs)) / (len(probs) - 1),
                            "probabilities": dict(zip(crit, probs))}
    return out


def make_handler(scorer, calibration=None, prompt=DEFAULT_PROMPT, options_in_question=False):
    def prepare(req):  # adapters trained on data/ova/ see every option in the question text (options.py)
        if not options_in_question or not isinstance(req, dict) or not isinstance(req.get("questions"), list):
            return req
        return req | {"questions": [with_options(q, str(q.get("id"))) if isinstance(q, dict) and q.get("type") in ("multiclass", "multilabel")
                                     and isinstance(q.get("candidates"), list) else q for q in req["questions"]]}

    lock = threading.Lock()  # ponytail: one model, one request at a time; a batching queue if throughput matters

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"  # keep-alive: every response sets Content-Length

        def _send(self, code, obj):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):  # noqa: N802
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"null")
                if self.path == "/classify":
                    with lock:
                        return self._send(200, classify(scorer, prepare(body), calibration, prompt))
                if self.path == "/api/alpha/decisions":
                    req, decode = compat_to_request(body)
                    with lock:
                        result = classify(scorer, prepare(req), calibration, prompt)
                    meta = result["meta"] | {"note": "shape-compatible endpoint serving personal-jev, not the hosted model"}
                    return self._send(200, {"model": f"personal-jev/{run_meta(scorer)['model']}", "answers": answers_from(result, decode),
                                            "meta": meta})
                self._send(404, {"error": {"message": f"unknown route {self.path}"}})
            except (ValidationError, json.JSONDecodeError) as e:
                self._send(400, {"error": {"message": str(e)}})
            except ValueError as e:  # InputTooLong and friends
                self._send(422, {"error": {"message": str(e)}})
            except Exception as e:  # never leave the client with an empty reply
                self._send(500, {"error": {"message": f"{type(e).__name__}: {e}"}})

        def log_message(self, *args):
            pass

    return Handler


def serve(scorer, host="127.0.0.1", port=8000, calibration=None, prompt=DEFAULT_PROMPT, options_in_question=False):
    httpd = ThreadingHTTPServer((host, port), make_handler(scorer, calibration, prompt, options_in_question))
    print(f"serving on http://{host}:{httpd.server_port}  (POST /classify, POST /api/alpha/decisions)", flush=True)
    httpd.serve_forever()
