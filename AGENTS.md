# LLM budget checks

Use `python3 llm_budget.py deepseek` for DeepSeek, `python3 llm_budget.py openrouter`
for OpenRouter, or `python3 llm_budget.py both` to check both. Read provider
credentials from the shell environment:

- `DEEPSEEK_API_KEY` holds the DeepSeek Platform API key for balance checks.
- `OPENROUTER_API_KEY` holds the OpenRouter Management Key for usage and credit checks.

Keep key values out of source files, command-line arguments, logs, and chat. Use the variable names when documenting setup.
