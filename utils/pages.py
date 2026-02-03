"""
Centralized page registry for GPE Lab.

Single source of truth for all page paths, icons, and i18n keys.
This avoids hardcoding emoji filenames throughout the codebase.
"""
from typing import NamedTuple


class PageDef(NamedTuple):
    """Definition of a navigation page."""
    id: str                # Unique identifier for i18n key lookup
    path: str              # File path relative to project root
    icon: str              # Display icon (emoji)
    section: str           # Section grouping: 'a', 'b', 'c', 'changelog'


# === Page Registry ===
# All pages defined in one place. Paths use actual filenames (with emojis).
# Future: rename files to ASCII and update paths here only.

NAV_PAGES: tuple[PageDef, ...] = (
    # Section A: GPE Electrochem Calculator
    PageDef(
        id="import",
        path="pages/1_📊_Import_Data.py",
        icon="📊",
        section="a",
    ),
    PageDef(
        id="eis",
        path="pages/2_⚡_EIS_Conductivity.py",
        icon="⚡",
        section="a",
    ),
    PageDef(
        id="temp_fits",
        path="pages/3_🌡️_Temperature_Fits.py",
        icon="🌡️",
        section="a",
    ),
    PageDef(
        id="transference",
        path="pages/4_🔋_Transference.py",
        icon="🔋",
        section="a",
    ),
    PageDef(
        id="stability",
        path="pages/5_📈_Stability_Window.py",
        icon="📈",
        section="a",
    ),
    # Section B: Smart Window
    PageDef(
        id="smart_window",
        path="pages/6_🪟_Smart_Window.py",
        icon="🪟",
        section="b",
    ),
    # Section C: Lab Database
    PageDef(
        id="database",
        path="pages/7_🗃️_Lab_Database.py",
        icon="🗃️",
        section="c",
    ),
    PageDef(
        id="analytics",
        path="pages/8_📉_Analytics.py",
        icon="📉",
        section="c",
    ),
    PageDef(
        id="reports",
        path="pages/9_📝_Reports.py",
        icon="📝",
        section="c",
    ),
    # Changelog
    PageDef(
        id="changelog",
        path="pages/10_📋_Update_Report.py",
        icon="📋",
        section="changelog",
    ),
)


def get_pages_by_section(section: str) -> list[PageDef]:
    """Get all pages in a given section."""
    return [p for p in NAV_PAGES if p.section == section]


def get_page_by_id(page_id: str) -> PageDef | None:
    """Get a page by its ID."""
    for p in NAV_PAGES:
        if p.id == page_id:
            return p
    return None
