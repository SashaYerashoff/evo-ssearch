from oldapp import app, ensure_probe_daemon_thread


ensure_probe_daemon_thread()


if __name__ == "__main__":
    app.run()
