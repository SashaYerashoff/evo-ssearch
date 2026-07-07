"""Gunicorn logging helpers for local self-signed HTTPS deployments."""

from __future__ import annotations

import logging

from gunicorn.glogging import Logger


class _TLSCertificateNoiseFilter(logging.Filter):
    """Suppress noisy client-side certificate rejection warnings."""

    NEEDLES = (
        "SSLV3_ALERT_CERTIFICATE_UNKNOWN",
        "ssl/tls alert certificate unknown",
        "TLSV1_ALERT_UNKNOWN_CA",
        "tlsv1 alert unknown ca",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        folded = message.lower()
        return not any(needle.lower() in folded for needle in self.NEEDLES)


class TLSNoiseFilteredLogger(Logger):
    """Gunicorn logger that keeps useful warnings and drops known TLS cert noise."""

    def setup(self, cfg) -> None:  # type: ignore[no-untyped-def]
        super().setup(cfg)
        noise_filter = _TLSCertificateNoiseFilter()
        self.error_log.addFilter(noise_filter)
        for handler in self.error_log.handlers:
            handler.addFilter(noise_filter)
