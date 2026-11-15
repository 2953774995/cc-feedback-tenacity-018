#!/usr/bin/env python3
"""重试统计装饰器验收探针。

契约：
  from tenacity.stats import tracked
  @tracked
  def fn(): ...
  fn.calls     调用总数
  fn.attempts  进入重试的总次数
  fn.failures  异常记录列表（每项 .attempt 与 .type）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FAILED = 0
PASSED = 0


def check(code, desc, fn):
    global FAILED, PASSED
    try:
        fn()
        print(f"[PASS] {code} {desc}")
        PASSED += 1
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] {code} {desc}")
        print(f"        {type(exc).__name__}: {exc}")
        FAILED += 1


def main():
    try:
        from tenacity.stats import tracked  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] P0 导入 tenacity.stats 失败：{type(exc).__name__}: {exc}")
        print("-" * 60)
        print("探针汇总：0/5 通过")
        return 1

    from tenacity import retry, stop_after_attempt, wait_none

    def p1():
        @tracked
        @retry(stop=stop_after_attempt(3), wait=wait_none())
        def ok():
            return 1
        assert ok() == 1
        assert ok.calls == 1 and ok.attempts == 1, (ok.calls, ok.attempts)

    def p2():
        state = {"n": 0}

        @tracked
        @retry(stop=stop_after_attempt(5), wait=wait_none())
        def flaky():
            state["n"] += 1
            if state["n"] < 3:
                raise ValueError("boom")
            return "done"
        assert flaky() == "done"
        assert flaky.attempts == 3, flaky.attempts
        assert len(flaky.failures) == 2, len(flaky.failures)

    def p3():
        @tracked
        @retry(stop=stop_after_attempt(2), wait=wait_none())
        def bad():
            raise RuntimeError("always")
        try:
            bad()
        except RuntimeError:
            pass
        assert bad.attempts == 2, bad.attempts
        assert all(f.type == "RuntimeError" for f in bad.failures), bad.failures

    def p4():
        calls = {"n": 0}

        @tracked
        @retry(stop=stop_after_attempt(2), wait=wait_none())
        def once():
            calls["n"] += 1
            return calls["n"]
        once(); once()
        assert once.calls == 2, once.calls

    def p5():
        from tenacity import retry, stop_after_attempt
        tries = {"n": 0}

        @retry(stop=stop_after_attempt(2))
        def raw():
            tries["n"] += 1
            raise ValueError()
        try:
            raw()
        except ValueError:
            pass
        assert tries["n"] == 2

    check("P1", "一次成功计数为一", p1)
    check("P2", "重试后成功", p2)
    check("P3", "最终失败记满次数", p3)
    check("P4", "多次调用累计", p4)
    check("P5", "既有 retry 不变", p5)

    print("-" * 60)
    print(f"探针汇总：{PASSED}/{PASSED + FAILED} 通过")
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
