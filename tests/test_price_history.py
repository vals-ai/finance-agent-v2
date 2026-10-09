import logging
import unittest
from unittest.mock import AsyncMock, patch

from finance_agent.tools import MAX_END_DATE, PriceHistory


class PriceHistoryTest(unittest.IsolatedAsyncioTestCase):
    async def test_reversed_ranges_are_rejected_before_fetching(self) -> None:
        tool = PriceHistory(api_key="test-key")
        for start_date, end_date in [
            ("2026-02-02", "2026-02-01"),
            ("2026-04-01", "2026-03-20"),
            ("2026-04-01", "2026-02-01"),
        ]:
            with self.subTest(start_date=start_date, end_date=end_date):
                with patch.object(tool, "_fetch", new_callable=AsyncMock) as fetch:
                    result = await tool.execute(
                        {
                            "ticker": "TEST",
                            "start_date": start_date,
                            "end_date": end_date,
                            "asset_class": "equity",
                        },
                        {},
                        logging.getLogger(__name__),
                    )

                self.assertEqual(
                    result.error,
                    f"start_date '{start_date}' is later than end_date '{end_date}'.",
                )
                fetch.assert_not_awaited()

    async def test_valid_ranges_keep_the_benchmark_cutoff(self) -> None:
        tool = PriceHistory(api_key="test-key")
        for start_date, end_date, expected_start, expected_end in [
            ("2026-02-01", "2026-02-02", "2026-02-01", "2026-02-02"),
            ("2026-02-01", "2026-04-01", "2026-02-01", MAX_END_DATE),
            ("2026-03-20", "2026-04-01", MAX_END_DATE, MAX_END_DATE),
            (MAX_END_DATE, MAX_END_DATE, MAX_END_DATE, MAX_END_DATE),
        ]:
            with self.subTest(start_date=start_date, end_date=end_date):
                with patch.object(
                    tool,
                    "_fetch",
                    new_callable=AsyncMock,
                    return_value=[{"date": f"{expected_start}T00:00:00Z", "close": 12.5}],
                ) as fetch:
                    result = await tool.execute(
                        {
                            "ticker": "TEST",
                            "start_date": start_date,
                            "end_date": end_date,
                            "asset_class": "equity",
                        },
                        {},
                        logging.getLogger(__name__),
                    )

                self.assertIsNone(result.error)
                self.assertEqual(result.output, f"date,close\n{expected_start},12.5")
                fetch.assert_awaited_once_with(
                    "equity", "TEST", expected_start, expected_end
                )


if __name__ == "__main__":
    unittest.main()
