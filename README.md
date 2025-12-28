# Claude Opus 4.5 Discord Bot

A cost-effective Discord bot powered by Claude Opus 4.5 with smart context management.

## Features

- 🧵 **Thread-based conversations** - keeps channels clean
- 📷 **Image understanding** - upload images and Claude can see them
- 😀 **Emoji reactions** - Claude can react to your messages
- 📎 **File attachments** - long code becomes downloadable files
- 🧠 **Two-tier memory** - just like an actual brain!
- 🔍 **Web search** - opt-in current info with citations
- 💰 **Cost tracking** - see exactly what you're spending

## The Memory System (The Cool Part)

This bot has two types of memory, like an actual brain:

### Working Memory (📝 Auto-managed)

Claude automatically jots down notes during conversation:
```
[note: sarah_deadline: grant due late January]
[note: xandra_working_on: RhizomeDB GraphQL layer]
```

- **Fades after ~48h** if not referenced
- **Sticks around longer** if Claude keeps finding it relevant
- **Max 10 notes** - stalest ones get pushed out
- **You can see them** with `!memories`
- **Promote to permanent** with `!keep <key>`

### Long-Term Memory (🧠 Permanent)

Explicit facts that never decay:
```
!remember sarah_cats Oscar and Reina
!remember project_name three-girlbosses-and-greg
```

- **Created by users** with `!remember`
- **Never decays** until you `!forget`
- **Max 25 entries** - you manage what's important

### Why This Works

| The problem | Bad solutions | This solution |
|-------------|---------------|---------------|
| "Remember my cats" | 72h arbitrary decay 💀 | Long-term memory (permanent) |
| "Claude should notice things" | Myk's 500-entry scratchpad 💸 | Working memory (10 max, auto-decay) |
| "Context gets huge" | Load everything every time | Bounded: 20 msgs + 10 notes + 25 facts |

## Cost Comparison

| Feature | Myk's Approach | This Bot |
|---------|---------------|----------|
| Message storage | Duplicate everything locally | Use Discord's own history |
| Memory | 500+ entries, all loaded every time 💸 | 10 working + 25 long-term |
| Context per message | 50,000+ tokens | ~3,000-8,000 tokens ✨ |

**Estimated costs:**
- ~$0.02-0.05 per message with Opus 4.5
- $20 prepaid → 400-1000 messages
- Low-traffic dev server = weeks/months of usage

## Setup

### Architecture: One Channel, Many Threads

```
#claude-dev (allowed channel from config.json)
├── Thread: "Chat with Xandra" ← Xandra's conversation
├── Thread: "Chat with Lauren" ← Lauren's conversation  
├── Thread: "Chat with Sarah"  ← Sarah's conversation
└── All threads share guild-level memory!
```

**Why this design:**
- **Clean channels** - conversations don't overlap
- **Shared context** - `!remember` in one thread is visible in all
- **Thread awareness** - Claude sees other threads for context
- **Per-person history** - your thread = your conversation history

**Good for dev teams:** Xandra's `!remember api_endpoint http://...` is visible to Lauren's conversations too.

### 1. Create Discord Bot

1. Go to [Discord Developer Portal](https://discord.com/developers/applications)
2. Click "New Application", name it whatever
3. Go to "Bot" section
4. Click "Reset Token" and copy it (you'll need this)
5. Enable these **Privileged Gateway Intents**:
   - ✅ MESSAGE CONTENT INTENT
6. Go to "OAuth2" > "URL Generator"
   - Scopes: `bot`
   - Bot Permissions: `Send Messages`, `Read Message History`, `Create Public Threads`, `Send Messages in Threads`, `Add Reactions`, `Attach Files`, `Embed Links`
7. Copy the generated URL and open it to invite bot to your server

### 2. Get Anthropic API Key

1. Go to [Anthropic Console](https://console.anthropic.com/)
2. Add credits (start with $20, seriously that's plenty for testing)
3. Create an API key

### 3. Configure the Bot

```bash
# Clone/download this folder
cd claude-discord-bot

# Install dependencies
pip install -r requirements.txt

# Create .env file
cp .env.example .env
# Edit .env with your tokens

# Create config.json
cp config.example.json config.json
# Edit config.json with your channel IDs
```

**.env file:**
```
DISCORD_TOKEN=your_discord_bot_token_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

**config.json:**
```json
{
  "allowed_channels": [
    123456789012345678
  ]
}
```

To get channel IDs: Discord Settings → Advanced → Enable Developer Mode → Right-click channel → Copy ID

### 4. Run It

```bash
python bot.py
```

## Usage

The bot will only respond in channels listed in `config.json`. When someone sends a message, it creates a thread for the conversation.

### Commands

| Command | Description |
|---------|-------------|
| `!help` | Show all commands |
| `!context` | Show current context size and cost estimate |
| `!cost` | Show total API usage and cost since bot started |
| `!memories` | List all memories (both working and long-term) |
| `!threads` | Show other recent threads in this channel |
| `!search <query>` | 🔍 Web search with citations (costs extra!) |
| `!remember <key> <value>` | Store a permanent memory |
| `!forget <key>` | Remove a memory (works for both types) |
| `!keep <key>` | Promote a working note to permanent memory |
| `!summarize <key>` | Auto-summarize this thread and save to long-term |
| `!summarize <key> <text>` | Save your own thread summary |

### Web Search

Need current information? Use `!search`:

```
!search latest Claude API updates
!search RhizomeDB github stars
!search Pittsburgh weather
```

**How it works:**
- Uses Claude's built-in web search tool
- Returns answer with source citations
- Sources displayed as Discord embed with clickable links

**Cost warning:** Web search adds ~$0.01-0.03 per search (extra tokens for search results). That's why it's opt-in via command rather than automatic.

### Thread Awareness

Claude can see other recent threads in the channel:

```
**Other recent threads in this channel** (for context):
- **Chat with Xandra** (2h ago) - "hey can we work on the GraphQL layer..."
- **RhizomeDB planning** (1d ago) - "ok let's figure out priorities..."
```

This is **read-only** - Claude can reference what others are working on but won't write notes about threads it's not in. This prevents feedback loops where Claude notes-about-notes-about-notes.

**Why this matters:** In your thread, you can ask "what's Lauren working on?" and Claude knows from the thread index.

### Saving Thread Summaries

When a thread has important decisions/outcomes, save it:

```
!summarize graphql_design
```

Claude will auto-generate a 1-2 sentence summary and save to long-term memory as `thread_graphql_design`. Or write your own:

```
!summarize graphql_design Decided to prioritize read queries, use Discord snowflake IDs
```

### Memory Indicators

When you run `!memories`, working notes show freshness:
- 🟢 Fresh (>70% life remaining)
- 🟡 Fading (30-70% life)  
- 🔴 Almost gone (<30% life)

If you see something important going 🔴, use `!keep <key>` to make it permanent!

### Special Features

**Image Understanding:**
Just upload an image with your message and Claude will see it.

**Emoji Reactions:**
Claude can react to your messages! It includes `[react: 👍]` in its response (which gets stripped from the visible text).

**File Attachments:**
If Claude generates a long code block with a filename like:
````
```script.py
# ... lots of code ...
```
````
It automatically becomes a downloadable file attachment.

### How Claude Writes Working Notes

Claude will automatically include `[note: key: value]` tags in responses when it notices something worth remembering:

```
User: My grant deadline is January 28th, I'm stressed
Claude: [note: deadline: grant due Jan 28] I hear you - grant deadlines are rough...
```

The `[note: ...]` part is stripped from the visible message, but the note gets saved to working memory. If Claude keeps referencing the deadline in later messages, the note stays fresh. If it never comes up again, it fades after ~48h.

### Promoting Notes

See a working note that should be permanent?
```
!keep deadline
```
Moves it from working memory (temporary) to long-term memory (permanent).

## Customization

Edit the `BotConfig` class in `bot.py`:

```python
@dataclass
class BotConfig:
    model: str = "claude-opus-4-5-20250514"  # or claude-sonnet-4-20250514 for cheaper
    max_tokens: int = 4096
    max_messages_to_fetch: int = 20          # Discord history to include
    max_longterm_memories: int = 25          # Permanent memory slots
    max_working_notes: int = 10              # Auto-managed notes
    working_memory_decay_hours: float = 48.0 # How long notes last
    system_prompt: str = """..."""           # Customize personality here
```

### Tuning Memory Behavior

**More aggressive decay** (notes fade faster):
```python
working_memory_decay_hours: float = 24.0  # Fade after 1 day
```

**More working memory** (for busy servers):
```python
max_working_notes: int = 20  # Double the slots
```

**Want it cheaper? Use Sonnet instead:**
```python
model: str = "claude-sonnet-4-20250514"  # ~10x cheaper than Opus
```

## File Structure

```
claude-discord-bot/
├── bot.py              # Main bot code
├── config.json         # Allowed channels (create from example)
├── config.example.json # Example config
├── requirements.txt    # Python dependencies
├── .env               # API keys (create this, don't commit!)
├── .env.example       # Example env file
├── memories.json      # Auto-generated memory persistence
└── README.md          # You are here
```

## Architecture Notes

**Context sources (in order):**
```
1. System prompt       (personality, instructions)
2. Thread index        (READ-ONLY list of other threads)
3. Long-term memory    (permanent !remember facts)
4. Working memory      (Claude's auto-notes, with decay)
5. Current thread      (last 20 messages from Discord)
```

**Why thread index is read-only:**

If Claude could write notes about other threads, you'd get feedback loops:
```
Thread A: Claude sees Thread B, notes "B discussed X"
Thread B: Claude sees note about itself, notes "someone noted I discussed X"
Thread A: Claude sees new note, notes "B knows someone noted..."
→ INFINITE LOOP 💀
```

By making thread index read-only (fetched fresh from Discord each time), Claude can reference other threads but can't create recursive note chains.

**Why two-tier memory?**

The original "72h decay on everything" was dumb because:
- "My cats are Oscar and Reina" shouldn't expire
- But Claude SHOULD be able to notice things without explicit commands

Solution:
```
Working Memory          Long-Term Memory
(Claude notices)   →    (User decides)
     ↓ !keep ↑
Fades if unused         Permanent until !forget
Max 10 notes            Max 25 entries
```

**Memory scope:**
- Memory is **per-guild** (server), not per-thread
- All threads in a guild share the same memory stores
- Thread summaries can be saved with `!summarize`

## Troubleshooting

**Bot doesn't respond:**
- Check channel ID is in config.json
- Check MESSAGE CONTENT INTENT is enabled in Discord Developer Portal
- Check bot has permissions in the channel

**"API Error" messages:**
- Check ANTHROPIC_API_KEY is correct
- Check you have credits in your Anthropic account

**Images not working:**
- Check file is under 20MB
- Check it's a supported format (png, jpg, jpeg, gif, webp)

**High costs:**
- Use `!cost` to check usage
- Switch to Sonnet for cheaper operation
- Reduce `max_messages_to_fetch` in config

## License

Do whatever you want with this. It's a Discord bot, not a spaceship.
