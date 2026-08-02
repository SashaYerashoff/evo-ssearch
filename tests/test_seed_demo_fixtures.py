from datetime import datetime, timezone

from scripts.seed_demo_fixtures import SEED_TAG, _records


def test_live_smoke_seed_is_fresh_and_idempotent_within_utc_day():
    records = _records(112, lambda text: [float(len(text))])
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    assert len(records) == 13
    assert len({row["dedupe_key"] for row in records}) == len(records)
    assert all(f"{SEED_TAG}:{today}:" in row["dedupe_key"] for row in records)
    assert all(row["channel_id"] == 112 for row in records)
    assert sum(row["payload"].get("role") == "positive" for row in records) == 4
    assert sum(row["payload"].get("role") == "negative" for row in records) == 8
