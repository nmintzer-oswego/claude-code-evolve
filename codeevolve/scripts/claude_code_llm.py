"""
ClaudeCodeLLM - OpenEvolve LLM provider that routes through Claude Code API.

This implements OpenEvolve's LLMInterface by calling `claude -p` (print mode)
as a subprocess for each generation request. This means:
- All LLM calls go through your Claude Code subscription (Max plan)
- No direct Anthropic API key needed
- Cost is covered by your subscription, not pay-per-token
- Each call is stateless (fresh context window per invocation)

CLI reference (claude v2.1.33):
- Prompt is always a positional CLI argument to `-p`
- Stdin piping feeds additional context (e.g. `cat file | claude -p "analyze"`)
- `--output-format json` returns {"result": "...", "session_id": "...", "usage": {...}, ...}
- `--system-prompt` replaces default system prompt entirely
- `--append-system-prompt` appends to default (keeps Claude Code defaults)
- `--tools ""` disables all tools; more reliable than `--allowedTools ""`
- `--no-session-persistence` makes calls stateless (print mode only)
- `--model` accepts aliases (sonnet, opus, haiku) or full names
- No --temperature or --max-tokens flags exist
- On Windows, must resolve full path via shutil.which() for subprocess calls

Usage in OpenEvolve config:
    from claude_code_llm import create_claude_code_llm

    model_cfg = LLMModelConfig(
        name="sonnet",
        init_client=create_claude_code_llm,
        ...
    )

Source: Promoted from Research/R6_EndToEnd/claude_code_llm.py (validated R1-R6).
Changes from R6 version:
- Updated sys.path to point from codeevolve/scripts/ to lib/openevolve_pkg/
"""

import asyncio
import json
import logging
import shutil
import subprocess
import sys
from typing import Dict, List, Optional

# Add the local openevolve install to path if present (vendored copy).
# Falls back to system-installed openevolve (pip install openevolve==0.2.26).
_lib_path = __import__('pathlib').Path(__file__).resolve().parent.parent.parent / 'lib' / 'openevolve_pkg'
if _lib_path.exists():
    sys.path.insert(0, str(_lib_path))

from openevolve.llm.base import LLMInterface

logger = logging.getLogger(__name__)


class ClaudeCodeLLM(LLMInterface):
    """LLM interface that routes through Claude Code's `claude -p` CLI."""

    def __init__(self, model_cfg=None):
        self.model = getattr(model_cfg, 'name', 'sonnet') or 'sonnet'
        self.timeout = getattr(model_cfg, 'timeout', 120) or 120
        self.retries = getattr(model_cfg, 'retries', 3) or 3
        self.retry_delay = getattr(model_cfg, 'retry_delay', 5) or 5
        self.system_message = getattr(model_cfg, 'system_message', None)

        # Budget control per invocation (in USD)
        self.max_budget_usd = getattr(model_cfg, 'max_budget_usd', None)

        # Cost tracking — accumulates total_cost_usd from each claude -p response
        self._total_cost_usd = 0.0
        self._call_count = 0

        # Resolve full path to claude CLI (needed on Windows where .cmd isn't auto-resolved)
        self.claude_path = shutil.which('claude')
        if not self.claude_path:
            raise FileNotFoundError(
                "Could not find 'claude' CLI. Ensure it's installed and on PATH."
            )

        logger.info(f"Initialized ClaudeCodeLLM with model: {self.model}, claude: {self.claude_path}")

    @property
    def total_cost_usd(self) -> float:
        """Total accumulated cost across all calls."""
        return self._total_cost_usd

    @property
    def call_count(self) -> int:
        """Total number of successful calls."""
        return self._call_count

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt by calling claude -p."""
        system_msg = kwargs.get('system_message', self.system_message)
        return await self._call_claude(prompt, system_message=system_msg, **kwargs)

    async def generate_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        """Generate text using a system message and conversational context.

        OpenEvolve calls this with system_message + messages=[{"role":"user","content":"..."}].
        We use --system-prompt for the system message and pass the user content
        as the -p prompt argument.
        """
        # Flatten messages into a single prompt
        parts = []
        for msg in messages:
            role = msg.get('role', 'user').lower()
            content = msg.get('content', '')
            if role == 'user':
                parts.append(content)
            elif role == 'assistant':
                parts.append(f"[Previous assistant response]:\n{content}")
            elif role == 'system':
                parts.append(f"[System]:\n{content}")

        user_prompt = '\n\n'.join(parts)

        return await self._call_claude(
            user_prompt,
            system_message=system_message,
            **kwargs
        )

    async def _call_claude(self, prompt: str, **kwargs) -> str:
        """Execute a claude -p subprocess call with retries."""
        retries = kwargs.get('retries', self.retries)
        retry_delay = kwargs.get('retry_delay', self.retry_delay)
        timeout = kwargs.get('timeout', self.timeout)

        last_error = None
        for attempt in range(retries + 1):
            try:
                result = await asyncio.wait_for(
                    self._run_subprocess(prompt, **kwargs),
                    timeout=timeout
                )
                return result
            except asyncio.TimeoutError:
                last_error = f"Timeout after {timeout}s"
                if attempt < retries:
                    delay = retry_delay * (2 ** attempt)  # exponential backoff
                    logger.warning(
                        f"Claude -p timeout on attempt {attempt + 1}/{retries + 1}. "
                        f"Retrying in {delay}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All {retries + 1} attempts timed out")
                    raise
            except Exception as e:
                last_error = str(e)
                if attempt < retries:
                    delay = retry_delay * (2 ** attempt)
                    logger.warning(
                        f"Error on attempt {attempt + 1}/{retries + 1}: {last_error}. "
                        f"Retrying in {delay}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All {retries + 1} attempts failed: {last_error}")
                    raise

    async def _run_subprocess(self, prompt: str, **kwargs) -> str:
        """Run claude -p as an async subprocess.

        The prompt is passed as the -p argument for short prompts, or piped
        via stdin for long prompts (Windows CLI arg limit is ~32K chars).
        """
        # Always pipe the full prompt via stdin to avoid Windows CLI arg escaping
        # issues with special characters (backticks, braces, newlines, etc.).
        # claude -p "instruction" reads stdin as additional context.
        cmd = [self.claude_path, '-p', 'Follow the instructions provided in the piped input below. Respond exactly as instructed.']
        stdin_data = prompt
        use_stdin = True

        # Output format
        cmd.extend(['--output-format', 'json'])

        # Model selection
        model = kwargs.get('model', self.model)
        if model:
            cmd.extend(['--model', model])

        # System prompt — use --system-prompt to replace defaults entirely
        # (we want pure text generation, not Claude Code's file-editing defaults)
        system_message = kwargs.get('system_message')
        if system_message:
            cmd.extend(['--system-prompt', system_message])

        # Budget control
        max_budget = kwargs.get('max_budget_usd', self.max_budget_usd)
        if max_budget:
            cmd.extend(['--max-budget-usd', str(max_budget)])

        # Stateless — don't persist sessions to disk
        cmd.append('--no-session-persistence')

        # Disable all tools — pure text generation only
        cmd.extend(['--tools', ''])

        # Log command details for debugging
        prompt_len = len(prompt)
        cmd_str_len = sum(len(c) for c in cmd)
        logger.info(
            f"claude -p call: prompt_len={prompt_len}, cmd_args={len(cmd)}, "
            f"cmd_str_len={cmd_str_len}, use_stdin={use_stdin}, "
            f"has_system_prompt={'--system-prompt' in cmd}"
        )

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: subprocess.run(
            cmd,
            input=stdin_data,
            capture_output=True,
            text=True,
            timeout=self.timeout,
        ))

        if result.returncode != 0:
            error_msg = result.stderr[:500] if result.stderr else 'Unknown error'
            logger.warning(f"claude -p exited with code {result.returncode}: {error_msg}")
            raise subprocess.CalledProcessError(
                result.returncode, cmd[0], result.stdout, result.stderr
            )

        # Parse JSON output — schema: {"result": "...", "session_id": "...", "usage": {...}}
        stdout = result.stdout.strip()
        try:
            response = json.loads(stdout)
            text = response.get('result', '')

            # Track cost from response
            cost = response.get('total_cost_usd', response.get('cost_usd', 0.0))
            if cost:
                self._total_cost_usd += cost
            self._call_count += 1

            # Sanitize Unicode minus sign (R4 finding: Claude sometimes emits U+2212)
            text = text.replace('\u2212', '-')

            if not text and stdout:
                # Fallback: return raw stdout if 'result' key is empty/missing
                logger.warning(f"JSON parsed but 'result' empty. Keys: {list(response.keys())}")
                text = stdout
            return text
        except json.JSONDecodeError:
            # If JSON parsing fails, return raw stdout (likely text format)
            logger.warning(
                f"Failed to parse claude -p JSON output (len={len(stdout)}). "
                f"First 200 chars: {stdout[:200]!r}"
            )
            return stdout


def create_claude_code_llm(model_cfg) -> ClaudeCodeLLM:
    """Factory function for use as init_client in OpenEvolve LLMModelConfig."""
    return ClaudeCodeLLM(model_cfg)
