import logging
import unittest

from gunicorn_tls_filter import _TLSCertificateNoiseFilter


class GunicornTLSFilterTests(unittest.TestCase):
    def test_suppresses_self_signed_certificate_handshake_noise(self) -> None:
        filt = _TLSCertificateNoiseFilter()
        record = logging.LogRecord(
            name="gunicorn.error",
            level=logging.WARNING,
            pathname=__file__,
            lineno=1,
            msg=(
                "Invalid request from ip=192.168.1.104: "
                "[SSL: SSLV3_ALERT_CERTIFICATE_UNKNOWN] "
                "ssl/tls alert certificate unknown"
            ),
            args=(),
            exc_info=None,
        )

        self.assertFalse(filt.filter(record))

    def test_keeps_other_gunicorn_warnings(self) -> None:
        filt = _TLSCertificateNoiseFilter()
        record = logging.LogRecord(
            name="gunicorn.error",
            level=logging.WARNING,
            pathname=__file__,
            lineno=1,
            msg="Worker timeout (pid: 1234)",
            args=(),
            exc_info=None,
        )

        self.assertTrue(filt.filter(record))


if __name__ == "__main__":
    unittest.main()
