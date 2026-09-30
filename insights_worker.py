"""One Insights worker process (spec 2026-09-30 §10.1).

insights_jobs starts a few of these with `python insights_worker.py`: run
by path, from the folder the app's own files are in, so it needs nothing
but those files and the app's Python, in the packaged app too. Each one
uses one torch thread and has its own torch random state, so nothing it
does can reach the suggestions made in the app's process.

Protocol: one JSON request per line on stdin, one JSON answer per line on
stdout, in order. The first line out is {"ready": true}, once the imports
are done. A request is
    {"id": ..., "kind": "main", "cfg": ..., "X": [[...]], "Y": [[...]]}
        -> {"id": ..., "state": insights.hyper_state(gp)}
    {"id": ..., "kind": "fold", "cfg": ..., "X", "Y", "Xt", "Yt"}
        -> {"id": ..., "z": insights.fold_z(...)}
and a fit that fails answers {"id": ..., "error": "..."}.
{"id": ..., "kind": "ping"} answers the worker's torch threads and pid. The worker exits
when stdin closes, which is also what happens when the app goes away."""
import json
import os
import sys


def answer(request):
    import torch

    import insights

    if request["kind"] == "ping":
        return {"id": request["id"], "threads": torch.get_num_threads(), "pid": os.getpid()}
    tensor = lambda key: torch.tensor(request[key], dtype=torch.double)
    build = insights.builder(request["cfg"])
    try:
        if request["kind"] == "main":
            gp = insights.fit_xy(build, tensor("X"), tensor("Y"))
            return {"id": request["id"], "state": insights.hyper_state(gp)}
        z = insights.fold_z(build, tensor("X"), tensor("Y"), tensor("Xt"), tensor("Yt"))
        return {"id": request["id"], "z": z}
    except Exception as exc:
        return {"id": request["id"], "error": str(exc) or type(exc).__name__}


def main():
    # Answers go to the real stdout alone: anything a library prints lands
    # on stderr instead of in the middle of an answer.
    out = sys.stdout
    sys.stdout = sys.stderr
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import warnings

    warnings.filterwarnings("ignore")
    import torch

    torch.set_num_threads(1)
    import insights  # noqa: F401  (torch, botorch and the app's code, imported once)

    out.write(json.dumps({"ready": True}) + "\n")
    out.flush()
    for line in sys.stdin:
        if not line.strip():
            continue
        out.write(json.dumps(answer(json.loads(line))) + "\n")
        out.flush()


if __name__ == "__main__":
    main()
