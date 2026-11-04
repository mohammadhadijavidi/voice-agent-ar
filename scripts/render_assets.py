"""Render README images.

assets/hero.png : a real screenshot of the dashboard after running the bundled simulated
                  calls (mock providers, fixed clock: Wed 4 Nov 2026, 09:00 Dubai).
assets/flow.png : scripts/flow.html (architecture diagram) rendered with Chromium.

Usage:  pip install playwright && python -m playwright install chromium
        python scripts/render_assets.py
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voice_agent_ar.cli import bundled_scenarios, running_server  # noqa: E402
from voice_agent_ar.config import Settings  # noqa: E402
from voice_agent_ar.profile import load_profile  # noqa: E402
from voice_agent_ar.server import create_app  # noqa: E402
from voice_agent_ar.simulator import Scenario, TwilioSimulator  # noqa: E402
from voice_agent_ar.storage import Store  # noqa: E402

NOW = datetime(2026, 11, 4, 9, 0, tzinfo=ZoneInfo("Asia/Dubai"))
# oldest first; the barge-in scenario seeds a full 8 pm sitting, so it runs after the 8:30 pm booking
CALLS = ["restaurant_booking", "english_caller", "landline_caller", "restaurant_taken_bargein"]


async def hero() -> None:
    db = str(Path(tempfile.mkdtemp()) / "hero.db")
    store = Store(db)
    profile = load_profile("restaurant")
    profile.now_fn = lambda: NOW
    app = create_app(Settings.from_env(db_path=db), profile=profile, store=store)
    async with running_server(app) as base:
        for name in CALLS:
            sc = Scenario.load(bundled_scenarios()[name])
            sc.seed(store, profile)
            await TwilioSimulator(base, speed=40).run(sc)
            await asyncio.sleep(0.6)
        calls = store.list_calls(10)
        target = next(c for c in calls if c["barge_ins"])   # the barge-in + SMS call
        async with async_playwright() as p:
            b = await p.chromium.launch()
            page = await b.new_page(viewport={"width": 1600, "height": 900}, device_scale_factor=2)
            await page.goto(f"{base}/?call={target['id']}")
            await page.wait_for_selector(".msg")
            await page.wait_for_timeout(800)
            await page.screenshot(path=str(ROOT / "assets" / "hero.png"))
            await b.close()
    store.close()


async def flow() -> None:
    async with async_playwright() as p:
        b = await p.chromium.launch()
        page = await b.new_page(viewport={"width": 1600, "height": 900}, device_scale_factor=2)
        await page.goto((ROOT / "scripts" / "flow.html").as_uri())
        await page.wait_for_timeout(500)
        await page.screenshot(path=str(ROOT / "assets" / "flow.png"))
        await b.close()


if __name__ == "__main__":
    (ROOT / "assets").mkdir(exist_ok=True)
    asyncio.run(hero())
    asyncio.run(flow())
    print("wrote assets/hero.png and assets/flow.png")
