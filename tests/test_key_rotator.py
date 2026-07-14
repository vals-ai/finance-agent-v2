import unittest
from unittest.mock import patch

from finance_agent.key_rotator import KeyRotator


async def _take_keys(rotator: KeyRotator, count: int) -> list[str]:
    keys: list[str] = []
    for _ in range(count):
        async with rotator.acquire() as key:
            keys.append(key)
    return keys


class KeyRotatorTest(unittest.IsolatedAsyncioTestCase):
    async def test_shuffles_a_copy_then_cycles_it(self) -> None:
        keys = ["key-a", "key-b", "key-c"]
        expected = list(reversed(keys))

        with patch(
            "finance_agent.key_rotator.random.shuffle",
            side_effect=lambda values: values.reverse(),
        ) as shuffle:
            actual = await _take_keys(KeyRotator(keys), len(keys) * 2)

        self.assertEqual(keys, ["key-a", "key-b", "key-c"])
        self.assertEqual(actual, expected * 2)
        self.assertEqual(shuffle.call_count, 1)

    async def test_each_rotator_shuffles_independently(self) -> None:
        keys = ["key-a", "key-b", "key-c"]
        orders = iter([list(reversed(keys)), ["key-b", "key-a", "key-c"]])

        def use_next_order(values: list[str]) -> None:
            values[:] = next(orders)

        with patch(
            "finance_agent.key_rotator.random.shuffle", side_effect=use_next_order
        ) as shuffle:
            first = await _take_keys(KeyRotator(keys), len(keys))
            second = await _take_keys(KeyRotator(keys), len(keys))

        self.assertNotEqual(first, second)
        self.assertCountEqual(first, keys)
        self.assertCountEqual(second, keys)
        self.assertEqual(shuffle.call_count, 2)


if __name__ == "__main__":
    unittest.main()
