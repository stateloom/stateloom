"""Tests for the Signal dispatcher."""

from stateloom.core.signals import Signal


def test_connect_and_emit():
    calls = []
    sig = Signal[str]("test")
    sig.connect(lambda s: calls.append(s))
    sig.emit("hello")
    assert calls == ["hello"]


def test_multiple_receivers():
    order = []
    sig = Signal[int]("multi")
    sig.connect(lambda s: order.append(("a", s)))
    sig.connect(lambda s: order.append(("b", s)))
    sig.emit(42)
    assert order == [("a", 42), ("b", 42)]


def test_disconnect():
    calls = []

    def receiver(s):
        calls.append(s)

    sig = Signal[str]("disc")
    sig.connect(receiver)
    sig.emit("before")
    sig.disconnect(receiver)
    sig.emit("after")
    assert calls == ["before"]


def test_emit_swallows_exceptions():
    calls = []

    def bad_receiver(s):
        raise RuntimeError("boom")

    def good_receiver(s):
        calls.append(s)

    sig = Signal[str]("swallow")
    sig.connect(bad_receiver)
    sig.connect(good_receiver)
    sig.emit("ok")
    assert calls == ["ok"]


def test_clear():
    calls = []
    sig = Signal[str]("clear")
    sig.connect(lambda s: calls.append(s))
    sig.clear()
    sig.emit("nope")
    assert calls == []
    assert len(sig) == 0


def test_len_and_bool():
    sig = Signal[str]("intro")
    assert len(sig) == 0
    assert not sig
    sig.connect(lambda s: None)
    assert len(sig) == 1
    assert sig


def test_duplicate_connect_ignored():
    def receiver(s):
        pass

    sig = Signal[str]("dup")
    sig.connect(receiver)
    sig.connect(receiver)
    assert len(sig) == 1


def test_receivers_returns_copy():
    def receiver(s):
        pass

    sig = Signal[str]("copy")
    sig.connect(receiver)
    receivers = sig.receivers
    receivers.clear()
    assert len(sig) == 1


def test_repr():
    sig = Signal[str]("my_signal")
    assert "my_signal" in repr(sig)
    assert "0" in repr(sig)
