from __future__ import annotations

from dataclasses import dataclass

SURGE_START_CHAIN = 4


@dataclass(frozen=True, slots=True)
class B2BOutcome:
    active: bool
    chain: int
    charge: int
    attack_bonus: int
    released: int


def split_surge(lines: int) -> tuple[int, ...]:
    """Split Surge into three ordered attacks, giving remainders to early packets."""

    total = max(0, int(lines))
    if total <= 0:
        return ()
    quotient, remainder = divmod(total, 3)
    packets = tuple(quotient + int(index < remainder) for index in range(3))
    return tuple(packet for packet in packets if packet > 0)


def resolve_b2b_charging(
    *,
    active: bool,
    chain: int,
    difficult: bool,
    lines: int,
    perfect_clear: bool = False,
) -> B2BOutcome:
    """Advance TETR.IO-style B2B Charging for one lock.

    ``chain`` is the displayed B2B count: the first difficult clear activates
    B2B but remains x0, the second difficult clear becomes x1. A difficult
    attack while already active receives +1 attack. At x4 and above, Surge is
    equal to the displayed chain in standard multiplayer rules.

    A no-clear placement preserves the chain. A non-difficult line clear breaks
    it and releases the stored Surge. Perfect Clears preserve/start B2B and add
    two displayed levels, matching the current multiplayer rule.
    """

    was_active = bool(active)
    current_chain = max(0, int(chain)) if was_active else 0
    cleared = max(0, int(lines)) > 0

    if not cleared:
        charge = current_chain if was_active and current_chain >= SURGE_START_CHAIN else 0
        return B2BOutcome(was_active, current_chain, charge, 0, 0)

    if difficult or perfect_clear:
        attack_bonus = 1 if was_active else 0
        if was_active:
            next_chain = current_chain + (1 if difficult else 0)
        else:
            next_chain = 0
        if perfect_clear:
            next_chain += 2
        charge = next_chain if next_chain >= SURGE_START_CHAIN else 0
        return B2BOutcome(True, next_chain, charge, attack_bonus, 0)

    released = current_chain if was_active and current_chain >= SURGE_START_CHAIN else 0
    return B2BOutcome(False, 0, 0, 0, released)
