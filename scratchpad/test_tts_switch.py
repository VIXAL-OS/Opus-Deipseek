"""Offline checks for the AZURE_TTS_ENABLED off-switch (2026-09-21): with it
set to false the bot must make NO Azure calls and NO G2P LLM calls, and
inline [[speak:..]] / [[french:..]] markers must collapse to plain text.
Builds a real ClaudeBot (no network at construction) inside a temp dir so the
live config.json / memories aren't read or written. Run from the repo root:

    PYTHONIOENCODING=utf-8 python scratchpad/test_tts_switch.py
"""
import asyncio
import os
import sys
import tempfile

sys.path.insert(0, os.getcwd())
os.environ["AZURE_TTS_KEY"] = "fake-key"          # set BEFORE import so the
os.environ["AZURE_TTS_REGION"] = "eastus"         # real .env can't override them
os.environ.setdefault("DISCORD_TOKEN", "x")
import bot  # noqa: E402

ok = fail = 0


def check(name, cond):
    global ok, fail
    print(("  ok   " if cond else "  FAIL ") + name)
    ok += bool(cond)
    fail += (not cond)


def build(flag):
    if flag is None:
        os.environ.pop("AZURE_TTS_ENABLED", None)
    else:
        os.environ["AZURE_TTS_ENABLED"] = flag
    here = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            b = bot.ClaudeBot()
        finally:
            os.chdir(here)
    return b


print("flag parsing")
for flag in ("false", "FALSE", " off ", "0", "no"):
    b = build(flag)
    check(f"AZURE_TTS_ENABLED={flag!r} → switched off, key blanked",
          b.azure_tts_switched_off and b.azure_tts_key is None)
for flag in (None, "true", "1", "yes", ""):
    b = build(flag)
    check(f"AZURE_TTS_ENABLED={flag!r} → on, key kept",
          not b.azure_tts_switched_off and b.azure_tts_key == "fake-key")

print("switched off → no Azure / G2P calls, markers collapse")
b = build("false")
calls = []


async def boom(*a, **k):
    calls.append(a)
    raise AssertionError("TTS path should not run while switched off")

b._azure_tts = boom
b._synthesize_mandarin = boom
b._synthesize_french = boom
text, files = asyncio.run(b._render_speak_attachments(0, "Say [[speak:你好|nǐ hǎo]] please", 4))
check("[[speak:..]] → plain text, no files", text == "Say 你好 please" and files == [])
text, files = asyncio.run(b._render_speak_attachments(0, "try !speak 谢谢", 4))
check("!speak 汉字 inline → plain text, no files", "谢谢" in text and "[[" not in text and files == [])
text, files = asyncio.run(b._render_french_attachments(0, "Dis [[french: bonjour]] !", 4))
check("[[french:..]] → plain text, no files", "bonjour" in text and "[[" not in text and files == [])
check("no Azure or G2P call was attempted", calls == [])
check("_azure_tts itself refuses with the key blanked",
      asyncio.run(bot.ClaudeBot._azure_tts(b, "<speak/>")) is None)

print(f"\n{ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
