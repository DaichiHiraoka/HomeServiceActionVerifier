# Legacy VLM Model Selection

## Decision

Use `gemma3:4b` through Ollama as the minimum local VLM for optional helper checks.

Use `gemma3:12b` as the higher-quality local check when the machine has enough memory and latency is acceptable. Keep `mock` only for pipeline smoke tests. The core research evaluation should still run with `rule_based`, `token_only`, and `proposed`.

## Rationale

Legacy visual helper checks need repeated comparisons across frame sampling and visual masking conditions. The default model therefore needs to be:

- runnable locally, so raw video does not leave the machine
- able to accept image input, because this PoC sends only selected/masked `grid.jpg`
- small enough for repeated experimental runs
- capable of structured JSON-like reasoning in Japanese prompts

Ollama's `gemma3` library lists multimodal 4B, 12B, and 27B variants with text and image input. The 4B variant is listed at about 3.3GB and the 12B variant at about 8.1GB. The model page describes Gemma 3 as multimodal with support for over 140 languages, which fits this repository's Japanese prompt and report workflow. Local testing in this repository also showed more stable JSON behavior with `gemma3:4b` than with `qwen2.5vl:3b` under repeated matrix runs.

Ollama's `qwen2.5vl` library lists 3B, 7B, 32B, and 72B variants with text and image input. The 3B variant is listed at about 3.2GB and remains a useful cross-model check, but it is not the default because repeated local matrix runs produced malformed JSON more often.

`openbmb/minicpm-v4` is a useful second-family robustness check after Gemma 3 is running. Its Ollama page describes it as an efficient 4.1B-parameter MiniCPM-V model with single-image, multi-image, and video understanding capability, but this repository still sends only selected/masked grid images.

## Commands

Install Ollama, then pull the selected model:

```bash
ollama pull gemma3:4b
```

Optional stronger local model:

```bash
ollama pull gemma3:12b
```

Enable the backend:

```bash
copy .env.example .env
```

Set:

```env
OLLAMA_ENABLED=true
OLLAMA_MODEL=gemma3:4b
```

Check readiness:

```bash
uv run python -m home_service_action_verifier.cli doctor
```

## Sources

- Ollama vision capability docs: https://docs.ollama.com/capabilities/vision
- Ollama `gemma3` model page and tags: https://registry.ollama.ai/library/gemma3 and https://registry.ollama.ai/library/gemma3/tags
- Ollama `qwen2.5vl` model page and tags: https://registry.ollama.ai/library/qwen2.5vl and https://registry.ollama.ai/library/qwen2.5vl/tags
- Ollama `llama3.2-vision` model page: https://registry.ollama.ai/library/llama3.2-vision
- Ollama `openbmb/minicpm-v4` model page: https://registry.ollama.com/openbmb/minicpm-v4
