"""Food Optimizer's shared launcher: the run-time half of desktop/launcher.sh,
in Python (stdlib only). The native shell starts it and reads the files it
writes; this docstring is the contract between them.

Start (every path is an argument; nothing is found by looking around):

    python -I -X utf8 -B -m launcher
        --state-dir DIR       app state: the files below (created if missing)
        --projects-dir DIR    the user's projects; the server's cwd
        --app PATH            the Streamlit app (app.py)
        [--python PATH]       interpreter for the children (default: this one)
        [--child-script PATH] child bootstrap (default: launcher/child.py)
        [--stdin-lifeline]    the shell holds stdin open; EOF = shut down,
                              and the idle watchdog is off
        [--lock-held]         the caller (launcher.sh) already holds launch.lock
        [--base-path auto|none]   auto (default): a new secret base path per launch
        [--warmup-timeout S]  default 120
        [--health-timeout S]  default 180

Environment: FOODOPT_IDLE_TIMEOUT_SECS (default 900, counted in 5 s ticks),
FOODOPT_PORT_BASE (first port to try; tests), FOODOPT_TEST_ARCH (tests; honoured only
when FOODOPT_TEST_HOOKS=1). The server's base path is handed to it as
STREAMLIT_SERVER_BASE_URL_PATH, never on its command line.

Files in the state dir, byte for byte as launcher.sh writes them:
    status.txt      "<text>|<percent>\\n" (percent empty when unknown), written
                    via status.txt.tmp and an atomic rename; removed once healthy
    server.base     "<base>\\n" — written BEFORE server.port, removed with it
                    (Python launcher only; absent with --base-path none)
    server.port     "<PORT> <PID>\\n"
    launch.lock/    held while a launch is in progress; launch.lock/pid = "<PID>\\n"
    launcher.log    "YYYY-MM-DD HH:MM:SS <message>\\n" lines, local time;
                    the previous launch's log is launcher.log.1
In the projects dir: backups/YYYY-MM-DD_HHMMSS/ (a copy of every top-level
*.pkl, the newest 10 kept) and backups/.last_sum.

Health: GET http://127.0.0.1:<port>/<base>/_stcore/health -> 200 "ok". The
server answers only Host 127.0.0.1:<port>; load http://127.0.0.1:<port>/<base>/.

Exit codes: 0 another launch is in progress, or a normal shutdown; 1 fatal
error (a bad argument too); 2 unsupported machine; 143 SIGTERM during the warm-up.
"""
