# Intro

Alpha's metacognitive layer. Watches conversations and notices what's memorable.

## What It Does

1. Subscribes to `transcript:*` Redis pubsub (published by the Loom's watcher)
2. Buffers user/assistant messages per session
3. After each assistant response, asks OLMo: "what's memorable here?"
4. Writes memorables to Redis for the Loom to inject

## Redis Keys

- `intro:conversation:{session_id}` - buffered conversation (list)
- `intro:memorables:{session_id}` - things worth remembering (list)

Both cleared when Alpha stores to Cortex. Both have 24h TTL as safety.

## Configuration

Environment variables:
- `REDIS_URL` - Redis connection (required)
- `OLLAMA_URL` - Ollama API endpoint (required)
- `OLLAMA_MODEL` - Model to use (default: olmo3:7b)
- `OTEL_EXPORTER_OTLP_ENDPOINT` - Parallax for observability

## Running

```bash
# Dev (with source mounted)
docker compose up

# Watch the logs
docker compose logs -f

# Or run directly
REDIS_URL=redis://alpha-pi:6379 OLLAMA_URL=http://primer:11434 python -m intro.main
```

## Observability

All OLMo calls are instrumented with OpenTelemetry spans:
- `intro.process_turn` - the full turn processing
- `intro.call_ollama` - the LLM call (routed to Phoenix via Parallax)

Watch in Phoenix at http://alpha-pi:6006

## The Prompt

`prompt.md` contains Intro's soul - who Alpha is, what matters, what to notice.
Written by Alpha on January 16, 2026.
