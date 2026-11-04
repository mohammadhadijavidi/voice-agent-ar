"""Add your own tool: e.g. tell callers today's special.

Run:  python examples/custom_tool.py   (starts the server with the extra tool on :8080)
"""

from voice_agent_ar.config import Settings
from voice_agent_ar.server import create_app
from voice_agent_ar.profile import load_profile
from voice_agent_ar.tools import Tool, ToolContext, default_tools

SPECIALS = {"fri": "مجبوس لحم", "sat": "هامور مشوي"}   # sample data


async def todays_special(ctx: ToolContext) -> dict:
    day = ctx.profile.today().strftime("%a").lower()
    dish = SPECIALS.get(day)
    return {"has_special": bool(dish), "spoken_special": dish or ""}


def build_app():
    settings = Settings.from_env()
    profile = load_profile(settings.profile)
    tools = default_tools(profile)
    tools.add(Tool("todays_special", "Today's special dish, if any.", {"type": "object", "properties": {}},
                   todays_special))
    return create_app(settings, profile=profile, tools=tools)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(build_app(), host="0.0.0.0", port=8080, ws="websockets-sansio")
