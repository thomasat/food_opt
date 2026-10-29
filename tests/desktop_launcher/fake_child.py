"""A stand-in for launcher/child.py in the behaviour suite: the same CLI,
none of the weight. Stdlib only and standalone (it runs under -I).

    fake_child.py warmup               sleep, then exit (see the env below)
    fake_child.py serve <app> <flags>  a tiny HTTP server on --server.port

Environment:
    FOODOPT_FAKE_RECORD              append one JSON line per child here
    FOODOPT_FAKE_WARMUP_SECS         warm-up duration (default 0)
    FOODOPT_FAKE_WARMUP_EXIT         warm-up exit code (default 0)
    FOODOPT_FAKE_SERVER_EXIT         serve exits at once with this code
    FOODOPT_FAKE_PORT_TAKEN_ONCE     a flag file: when absent, serve creates it
                                     and fails as Streamlit does on a taken port
    FOODOPT_FAKE_HEALTH_DELAY_SECS   health answers 503 until this has passed
    FOODOPT_FAKE_CHATTER             serve writes a numbered line to stderr
                                     every 50 ms (as a real server logs)
"""
import http.server
import json
import os
import signal
import sys
import sysconfig
import time


def _record(kind):
    path = os.environ.get("FOODOPT_FAKE_RECORD")
    if not path:
        return
    env = {k: v for k, v in os.environ.items()
           if k.startswith(("PYTHON", "STREAMLIT_")) or k.lower().endswith("_proxy")
           or k == "GIT_PYTHON_REFRESH"}
    console = None
    if sys.platform == "win32":
        import ctypes
        console = ctypes.windll.kernel32.GetConsoleWindow() or 0
    line = json.dumps({
        "kind": kind,
        "pid": os.getpid(),
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "env": env,
        "flags": {"isolated": sys.flags.isolated, "utf8_mode": sys.flags.utf8_mode,
                  "dont_write_bytecode": sys.flags.dont_write_bytecode},
        "userbase": sysconfig.get_config_var("userbase"),
        "sys_path": sys.path,
        "console_window": console,
    })
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def warmup():
    _record("warmup")
    time.sleep(float(os.environ.get("FOODOPT_FAKE_WARMUP_SECS", "0")))
    return int(os.environ.get("FOODOPT_FAKE_WARMUP_EXIT", "0"))


def _flag(flags, name):
    prefix = f"--{name}="
    return next((f[len(prefix):] for f in flags if f.startswith(prefix)), None)


def serve(app, flags):
    _record("serve")
    if os.environ.get("FOODOPT_FAKE_SERVER_EXIT"):
        return int(os.environ["FOODOPT_FAKE_SERVER_EXIT"])
    port = int(_flag(flags, "server.port"))
    once = os.environ.get("FOODOPT_FAKE_PORT_TAKEN_ONCE")
    if once and not os.path.exists(once):
        with open(once, "w", encoding="utf-8") as fh:
            fh.write("taken\n")
        sys.stderr.write(f"Port {port} is not available\n")
        return 1
    base = os.environ.get("STREAMLIT_SERVER_BASE_URL_PATH")
    health = f"/{base}/_stcore/health" if base else "/_stcore/health"
    ready_at = time.monotonic() + float(os.environ.get("FOODOPT_FAKE_HEALTH_DELAY_SECS", "0"))

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != health:
                self.send_error(404)
                return
            ok = time.monotonic() >= ready_at
            body = b"ok" if ok else b"starting"
            self.send_response(200 if ok else 503)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    if os.environ.get("FOODOPT_FAKE_CHATTER"):
        import threading

        def chatter():
            n = 0
            while True:
                n += 1
                sys.stderr.write(f"fake server chatter {n}\n")
                sys.stderr.flush()
                time.sleep(0.05)
        threading.Thread(target=chatter, daemon=True).start()
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, lambda *a: os._exit(0))
    server.serve_forever()
    return 0


def main(argv):
    if argv[:1] == ["warmup"]:
        return warmup()
    if argv[:1] == ["serve"] and len(argv) >= 2:
        return serve(argv[1], argv[2:])
    sys.stderr.write("usage: fake_child.py warmup | serve <app> <flags...>\n")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
