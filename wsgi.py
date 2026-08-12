from oldapp import (
    app,
    ensure_archive_retention_thread,
    ensure_incident_maintenance_worker,
    ensure_probe_daemon_thread,
    initialize_runtime_services,
    runtime_background_services_allowed,
    runtime_handover_pending,
)


initialize_runtime_services()
if not runtime_handover_pending():
    if runtime_background_services_allowed():
        ensure_probe_daemon_thread()
        ensure_incident_maintenance_worker()
    ensure_archive_retention_thread()


if __name__ == "__main__":
    app.run()
