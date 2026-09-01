from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from src.preprocessing.temporal import filter_recent_posts


def test_recent_12_month_filter_is_inclusive_and_dynamic() -> None:
    frame = pd.DataFrame({"post_id": ["old", "boundary", "recent", "future"], "publish_time": [
        "2025-08-14T23:59:59+00:00", "2025-08-15T00:00:00+00:00",
        "2026-01-01T00:00:00+00:00", "2026-08-15T00:00:01+00:00",
    ]})
    result = filter_recent_posts(frame, as_of=datetime(2026, 8, 15, tzinfo=timezone.utc))
    assert result["post_id"].tolist() == ["boundary", "recent"]
