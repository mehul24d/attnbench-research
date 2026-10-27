"""Readers for the sampler-process tests. They live in a module of their own
because the sampler process is spawned, and a spawned process imports its
reader factory by name: a function defined inside a test file would not be
importable there. The shared values are multiprocessing values, so the test
sets the card's state and the child process reads it."""

from __future__ import annotations

import time

from attnbench.clock_sampling import ClockSample


class SharedBitsReader:
    observes_transients = False

    def __init__(self, bits, sm):
        self.bits, self.sm = bits, sm

    def read(self) -> ClockSample:
        return ClockSample(sm_mhz=int(self.sm.value), mem_mhz=6250,
                           throttle_bits=int(self.bits.value))


def shared_bits_reader(bits, sm):
    return SharedBitsReader(bits, sm)


def unbuildable_reader():
    raise RuntimeError("no such device")


class _SticksAfterFirstRead:
    observes_transients = False

    def __init__(self):
        self.n = 0

    def read(self) -> ClockSample:
        self.n += 1
        if self.n > 1:
            time.sleep(60)
        return ClockSample(sm_mhz=2040, mem_mhz=6250, throttle_bits=0)


def sticks_after_first_read():
    return _SticksAfterFirstRead()
