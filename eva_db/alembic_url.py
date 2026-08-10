"""Convert supported PostgreSQL DSNs into SQLAlchemy migration URLs."""

from __future__ import annotations

from sqlalchemy.engine import URL


def sqlalchemy_database_url(dsn: str) -> str:
    """Return an unredacted SQLAlchemy URL for a PostgreSQL DSN.

    EVA accepts both PostgreSQL URI syntax and libpq conninfo.  Alembic uses
    SQLAlchemy, which cannot consume conninfo directly, so keyword/value DSNs
    must be parsed before they are passed to ``engine_from_config``.
    """

    value = str(dsn or "").strip()
    if not value:
        raise ValueError("database DSN is empty")

    if "://" in value:
        if value.startswith("postgres://"):
            value = "postgresql://" + value.removeprefix("postgres://")
        if value.startswith("postgresql://"):
            value = "postgresql+psycopg://" + value.removeprefix("postgresql://")
        if not value.startswith("postgresql+psycopg://"):
            raise ValueError("database DSN must use PostgreSQL")
        return value

    try:
        from psycopg.conninfo import conninfo_to_dict

        parameters = conninfo_to_dict(value)
    except Exception as exc:
        raise ValueError("database DSN is not valid libpq conninfo") from exc

    username = parameters.pop("user", None)
    password = parameters.pop("password", None)
    host = parameters.pop("host", None)
    database = parameters.pop("dbname", None)
    raw_port = parameters.pop("port", None)
    try:
        port = int(raw_port) if raw_port else None
    except (TypeError, ValueError) as exc:
        raise ValueError("database DSN contains an invalid port") from exc

    query = {
        str(key): str(item)
        for key, item in parameters.items()
        if item is not None and str(item) != ""
    }
    url = URL.create(
        "postgresql+psycopg",
        username=username,
        password=password,
        host=host,
        port=port,
        database=database,
        query=query,
    )
    return url.render_as_string(hide_password=False)
