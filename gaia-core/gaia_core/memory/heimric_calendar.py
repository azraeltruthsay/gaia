"""Heimric calendar for the in-world journal context.

Schema source: /knowledge/dnd_campaign/docs/heimr_general_info.md.
  • 12 months × 28 days = 336-day year
  • 6 seasons (2 months each): Amora, Verda, Siderea, Autnu, Multum, Negu
  • 7-day week: Pecar, Manak, Tauray, Fieten, Bocos, Librim, Saggit
  • Two moons:
      - Luna: 28-day cycle, new on the 1st, full on the 15th of every month
      - Chriton: 33.6-day cycle, aligns to Luna on 15th of Erom and 15th of Harmen
  • Eras: BC (Before the Cull) and CE (Current Era)

Day-count anchor: day 0 = CE 1, Thame 1. BC dates have negative day counts
(BC 1 Thame 1 = day -336, BC 1 Froso 28 = day -1).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

DAYS_PER_MONTH = 28
MONTHS_PER_YEAR = 12
DAYS_PER_YEAR = DAYS_PER_MONTH * MONTHS_PER_YEAR  # 336

# (month_name, season_name) in calendar order — index+1 = month number.
MONTHS = (
    ("Thame", "Amora"),
    ("Erom", "Amora"),
    ("Renna", "Verda"),
    ("Sedjem", "Verda"),
    ("Gromi", "Siderea"),
    ("Suhmer", "Siderea"),
    ("Ripia", "Autnu"),
    ("Harmen", "Autnu"),
    ("Stoma", "Multum"),
    ("Plenia", "Multum"),
    ("Chima", "Negu"),
    ("Froso", "Negu"),
)
MONTH_NAMES = tuple(m[0] for m in MONTHS)
SEASON_OF_MONTH = {i + 1: m[1] for i, m in enumerate(MONTHS)}
_MONTH_NAME_TO_INDEX = {name.lower(): i + 1 for i, name in enumerate(MONTH_NAMES)}

WEEKDAYS = ("Pecar", "Manak", "Tauray", "Fieten", "Bocos", "Librim", "Saggit")

LUNA_CYCLE = 28.0
CHRITON_CYCLE = 33.6
# Both moons align to "full" at CE 1 Erom 15. Erom is month 2;
# day_count = 1*28 + 14 = 42. We use this as Chriton's full-phase anchor.
CHRITON_FULL_ANCHOR_DAY = 42

ERAS = ("BC", "CE")


@dataclass(frozen=True, order=True)
class HeimricDate:
    """An in-world Heimric date, stored canonically as an integer day_count.

    day_count is days since CE 1 Thame 1 (day 0). BC dates are negative.
    All derived properties (era/year/month/day/weekday/season/moon phases)
    are computed from day_count, so arithmetic stays simple.
    """
    day_count: int

    # ── Components ──────────────────────────────────────────────────────

    @property
    def era(self) -> str:
        return "CE" if self.day_count >= 0 else "BC"

    @property
    def year(self) -> int:
        if self.day_count >= 0:
            return self.day_count // DAYS_PER_YEAR + 1
        # BC: day -1 → BC 1, day -336 → BC 1, day -337 → BC 2.
        return ((-self.day_count - 1) // DAYS_PER_YEAR) + 1

    def _year_start_day(self) -> int:
        """day_count of Thame 1 of this entry's year."""
        if self.day_count >= 0:
            return (self.year - 1) * DAYS_PER_YEAR
        return -self.year * DAYS_PER_YEAR

    def _day_in_year(self) -> int:
        """0-indexed day within the current year (0 = Thame 1, 335 = Froso 28)."""
        return self.day_count - self._year_start_day()

    @property
    def month(self) -> int:
        return self._day_in_year() // DAYS_PER_MONTH + 1

    @property
    def day(self) -> int:
        return self._day_in_year() % DAYS_PER_MONTH + 1

    @property
    def month_name(self) -> str:
        return MONTH_NAMES[self.month - 1]

    @property
    def season(self) -> str:
        return SEASON_OF_MONTH[self.month]

    @property
    def weekday(self) -> str:
        # 7-day cycle. CE 1 Thame 1 (day 0) = Pecar. Negative day_counts
        # use Python's %, which returns a non-negative remainder, so this
        # works for BC dates too.
        return WEEKDAYS[self.day_count % 7]

    # ── Moons ───────────────────────────────────────────────────────────

    @property
    def luna_phase(self) -> float:
        """Luna phase as fraction in [0, 1). 0 = new, 0.5 = full."""
        return (self.day_count % LUNA_CYCLE) / LUNA_CYCLE

    @property
    def chriton_phase(self) -> float:
        """Chriton phase as fraction in [0, 1). 0 = new, 0.5 = full.
        Anchored: phase(CE 1 Erom 15) = 0.5 (full)."""
        offset = (self.day_count - CHRITON_FULL_ANCHOR_DAY) % CHRITON_CYCLE
        return (offset / CHRITON_CYCLE + 0.5) % 1.0

    @property
    def luna_phase_name(self) -> str:
        return phase_name(self.luna_phase)

    @property
    def chriton_phase_name(self) -> str:
        return phase_name(self.chriton_phase)

    # ── Construction / arithmetic ───────────────────────────────────────

    @staticmethod
    def from_components(era: str, year: int, month: int, day: int) -> "HeimricDate":
        era_u = (era or "").upper()
        if era_u not in ERAS:
            raise ValueError(f"era must be 'CE' or 'BC', got {era!r}")
        if not (1 <= month <= 12):
            raise ValueError(f"month {month} out of range 1..12")
        if not (1 <= day <= 28):
            raise ValueError(f"day {day} out of range 1..28")
        if year < 1:
            raise ValueError(f"year must be >= 1, got {year}")
        if era_u == "CE":
            day_count = (year - 1) * DAYS_PER_YEAR + (month - 1) * DAYS_PER_MONTH + (day - 1)
        else:
            day_count = -year * DAYS_PER_YEAR + (month - 1) * DAYS_PER_MONTH + (day - 1)
        return HeimricDate(day_count=day_count)

    def add_days(self, n: int) -> "HeimricDate":
        return HeimricDate(day_count=self.day_count + n)

    # ── Formatting / parsing ────────────────────────────────────────────

    def format(self, *, long: bool = False) -> str:
        base = f"{self.era} {self.year}, {self.month_name} {self.day}"
        if long:
            base += (
                f" ({self.weekday}, {self.season}; "
                f"Luna {self.luna_phase_name}, Chriton {self.chriton_phase_name})"
            )
        return base

    def __str__(self) -> str:
        return self.format()


_DATE_RE = re.compile(
    r"^\s*(?P<era>CE|BC)\s+(?P<year>\d+)\s*[,/]?\s*"
    r"(?P<month>[A-Za-z]+)\s+(?P<day>\d+)\s*$",
    re.IGNORECASE,
)


def parse(text: str) -> HeimricDate:
    """Parse a Heimric date. Accepts:
        'CE 157, Erom 15'
        'CE 157 Erom 15'
        'CE 157 / Erom / 15'
        'BC 12 Thame 3'
    """
    m = _DATE_RE.match(text or "")
    if not m:
        raise ValueError(f"Cannot parse Heimric date: {text!r}")
    era = m.group("era").upper()
    year = int(m.group("year"))
    month_name = m.group("month").lower()
    month = _MONTH_NAME_TO_INDEX.get(month_name)
    if month is None:
        raise ValueError(f"Unknown Heimric month: {m.group('month')!r}")
    day = int(m.group("day"))
    return HeimricDate.from_components(era, year, month, day)


def try_parse(text: str) -> Optional[HeimricDate]:
    """Like parse() but returns None on failure rather than raising."""
    try:
        return parse(text)
    except (ValueError, AttributeError, TypeError):
        return None


# ── Moon phase naming ───────────────────────────────────────────────────

_PHASE_BANDS = (
    (0.0625, "new"),
    (0.1875, "waxing crescent"),
    (0.3125, "first quarter"),
    (0.4375, "waxing gibbous"),
    (0.5625, "full"),
    (0.6875, "waning gibbous"),
    (0.8125, "last quarter"),
    (0.9375, "waning crescent"),
    (1.0625, "new"),  # wrap
)


def phase_name(phase: float) -> str:
    """Map a numeric phase in [0, 1) to a human-readable name."""
    p = phase % 1.0
    for upper, name in _PHASE_BANDS:
        if p < upper:
            return name
    return "new"
