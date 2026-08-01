from oldapp import (
    app,
    ensure_archive_retention_thread,
    ensure_probe_daemon_thread,
    initialize_runtime_services,
)


initialize_runtime_services()
ensure_probe_daemon_thread()
ensure_archive_retention_thread()


if __name__ == "__main__":
    app.run()
