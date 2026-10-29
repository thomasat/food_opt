"""What only the server child installs, after Streamlit and Tornado are
importable: the Host check and the isolation from the user's own
.streamlit/config.toml files. Loaded by path from child.py."""


def install_host_check(allowed_host: str) -> None:
    """Serve a request only when its Host header is `allowed_host`
    ("127.0.0.1:<port>"), ignoring case; anything else gets 403. Blocks DNS
    rebinding and other users' browsers on a shared machine."""
    import tornado.web

    allowed = allowed_host.lower()
    app = tornado.web.Application
    original = getattr(app.find_handler, "_foodopt_original", app.find_handler)

    def find_handler(self, request, **kwargs):
        if (request.host or "").lower() != allowed:
            return self.get_handler_delegate(request, tornado.web.ErrorHandler,
                                             {"status_code": 403})
        return original(self, request, **kwargs)

    find_handler._foodopt_original = original
    app.find_handler = find_handler


def ignore_user_config_files() -> None:
    """Streamlit reads no config.toml or secrets.toml (~/.streamlit,
    <cwd>/.streamlit, the app's folder): every option that matters is passed as a flag."""
    import streamlit.config as config

    original = getattr(config.get_config_files, "_foodopt_original", config.get_config_files)

    def get_config_files(file_name):
        if file_name in ("config.toml", "secrets.toml"):
            return []
        return original(file_name)

    get_config_files._foodopt_original = original
    config.get_config_files = get_config_files
