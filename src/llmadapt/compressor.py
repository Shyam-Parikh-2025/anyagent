import ast
import functools
import itertools
import logging
import re
from dataclasses import dataclass
from typing import Callable, Optional

logger = logging.getLogger(__name__)

_TOKEN_CHUNK = re.compile(r"[A-Za-z0-9]+|\s+|[^\sA-Za-z0-9]")


class ContextCompressor:
    """Uses standard library AST parsing to compress code down to pure interface stubs,
    plus helpers for truncating/deduping noisy tool output and splitting one shared
    context budget across multiple things that need compressing.

    Note: comments live outside the AST, so code_to_stub always drops them - only
    docstrings survive. See the bottom of this file for notes on alternatives (e.g. a
    CST-based approach) that don't have this limitation.
    """

    @staticmethod
    def _chunk_tokens(chunk: str) -> int:
        """Token cost of one already-split chunk (see _TOKEN_CHUNK). A single non-space
        punctuation/symbol char is ~1 token in real BPE tokenizers, so that's free. Runs of
        letters/digits or of multiple whitespace chars get the same ~4-chars/token guess
        chars/4 uses globally, just scoped to the run instead of the whole string. A single
        whitespace char is ~free since BPE usually folds it into the next word's token rather
        than spending a token on it by itself."""
        if chunk.isspace():
            return max(1, -(-len(chunk) // 4)) if len(chunk) > 1 else 0
        if chunk[0].isalnum():
            return max(1, -(-len(chunk) // 4))  # ceil division
        return 1

    @staticmethod
    def _token_len(text: str) -> int:
        """Best-effort token count with no external tokenizer - see _TOKEN_CHUNK's comment
        for the reasoning. Not exact, just noticeably closer than a flat chars/4 guess."""
        if not text:
            return 0
        return max(1, sum(ContextCompressor._chunk_tokens(c) for c in _TOKEN_CHUNK.findall(text)))

    @staticmethod
    def _char_offset_for_token_budget(text: str, token_budget: int, from_end: bool = False) -> int:
        """Finds a character offset such that the estimated token count of the kept slice is
        as close to token_budget as possible without going over. Walks the same chunks
        _token_len counts, front-to-back (or back-to-front when from_end=True), stopping the
        moment adding another chunk would exceed the budget. Returns an offset for
        `text[:offset]` (from_end=False) or `text[offset:]` (from_end=True).

        Doing this chunk-by-chunk (instead of e.g. scaling token_budget by an average
        chars/token ratio) means we're using the exact same accounting as _token_len itself,
        so "budget N tokens" and "the estimated size of what we sliced" always agree - and as
        a side effect we only ever stop between chunks, never mid-word, so this alone mostly
        keeps a head/tail cut from landing inside a token or a word."""
        if token_budget <= 0:
            return len(text) if from_end else 0
        chunks = _TOKEN_CHUNK.findall(text)
        if from_end:
            chunks = reversed(chunks)
        used = 0
        kept_chars = 0
        for chunk in chunks:
            cost = ContextCompressor._chunk_tokens(chunk)
            if used + cost > token_budget:
                break
            used += cost
            kept_chars += len(chunk)
        return len(text) - kept_chars if from_end else kept_chars

    @staticmethod
    def _snap_to_newline(text: str, index: int, look: int = 200) -> int:
        """Nudge a raw slice index onto the nearest newline within `look` chars, so head/tail
        splits don't cut a line (or a multi-char sequence) in half. Falls back to the raw
        index if there's no newline nearby."""
        window_start = max(0, index - look)
        window_end = min(len(text), index + look)
        before = text.rfind("\n", window_start, index)
        after = text.find("\n", index, window_end)
        if before == -1 and after == -1:
            return index
        if before == -1:
            return after + 1
        if after == -1:
            return before + 1
        return before + 1 if (index - before) <= (after - index) else after + 1

    @staticmethod
    def _dedupe_repeated_lines(text: str, min_repeats: int = 3) -> str:
        """Collapses runs of the same line repeated min_repeats+ times in a row - logs and
        stack traces love to repeat one line dozens of times and that's budget we'd rather
        spend elsewhere."""
        lines = text.split("\n")
        out = []
        for line, group in itertools.groupby(lines):
            count = sum(1 for _ in group)
            if count >= min_repeats:
                out.append(line)
                out.append(f"... [line repeated {count} times] ...")
            else:
                out.extend([line] * count)
        return "\n".join(out)

    @staticmethod
    def _collapse_literal(node: ast.expr, max_items: int) -> ast.expr:
        """Replace a too-big list/tuple/set/dict literal with a small placeholder showing the
        count, so a stub doesn't drag in a 500-entry constant table along with it."""
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)) and len(node.elts) > max_items:
            placeholder = ast.Constant(value=f"... {len(node.elts)} items ...")
            new_node = ast.Set(elts=[placeholder]) if isinstance(node, ast.Set) \
                else type(node)(elts=[placeholder], ctx=ast.Load())
            ast.copy_location(new_node, node)
            ast.fix_missing_locations(new_node)
            return new_node
        if isinstance(node, ast.Dict) and len(node.keys) > max_items:
            new_node = ast.Dict(keys=[ast.Constant(value="...")],
                                 values=[ast.Constant(value=f"{len(node.keys)} items")])
            ast.copy_location(new_node, node)
            ast.fix_missing_locations(new_node)
            return new_node
        return node

    @staticmethod
    @functools.lru_cache(maxsize=64)
    def _code_to_stub_cached(source_code: str, max_short_body_lines: int, max_short_body_chars: int,
                              max_literal_items: int) -> str:
        """Does the actual stubbing. Cached (keyed on the exact source + settings) so re-stubbing
        the same file across multiple turns doesn't re-parse it from scratch every time."""
        try:
            tree = ast.parse(source_code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Extract docstring if present
                    docstring = ast.get_docstring(node)

                    # Filter out docstring statement to measure ONLY executable logic size
                    logic_stmts = [
                        stmt for stmt in node.body
                        if not (
                            isinstance(stmt, ast.Expr)
                            and isinstance(stmt.value, ast.Constant)
                            and isinstance(stmt.value.value, str)
                        )
                    ] if docstring else node.body

                    # Calculate actual character & line size of implementation logic
                    unparsed_logic = "\n".join(ast.unparse(s) for s in logic_stmts)
                    logic_chars = len(unparsed_logic)
                    logic_lines = unparsed_logic.count("\n") + 1 if unparsed_logic else 0

                    # Explicitly say the list type as list[ast.stmt] bc Pylance may think it is not
                    new_body: list[ast.stmt] = []

                    # Replace function body with docstring + Ellipsis (...) if docstring exists,
                    # else if function is small, then keep it, else Ellipsis
                    if logic_chars <= max_short_body_chars and logic_lines <= max_short_body_lines:
                        new_body.extend(node.body)
                    else:
                        if docstring:
                            new_body.append(ast.Expr(value=ast.Constant(value=docstring)))
                        new_body.append(ast.Expr(value=ast.Constant(value=Ellipsis)))

                    # Now Pylance knows this is a valid list[ast.stmt]
                    node.body = new_body

                elif isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
                    # Big module/class-level constant tables (config dicts, lookup lists, ...) bloat
                    # a stub just as much as a long function body does - collapse those too.
                    node.value = ContextCompressor._collapse_literal(node.value, max_literal_items)

            stubbed = ast.unparse(tree)
            before, after = len(source_code), len(stubbed)
            reduction = 100 * (1 - after / before) if before else 0
            logger.debug("code_to_stub: %d -> %d chars (%.0f%% reduction)", before, after, reduction)
            return stubbed
        except Exception as e:
            # Log it instead of failing silently - otherwise a bad compression just looks like
            # "the file didn't shrink" and nobody notices why.
            logger.warning("code_to_stub: couldn't parse/compress source (%r), returning as-is", e)
            return source_code + "\n... [Code could not be compressed] ..."

    @staticmethod
    def code_to_stub(source_code: str, max_short_body_lines: int = 5, max_short_body_chars: int = 120,
                      max_literal_items: int = 20) -> str:
        """Strips inner function/class bodies while preserving signatures, docstrings, and type hints.
        Also collapses oversized module/class-level literals (max_literal_items) down to a placeholder
        showing how many items were dropped."""
        return ContextCompressor._code_to_stub_cached(source_code, max_short_body_lines, max_short_body_chars,
                                                        max_literal_items)

    @staticmethod
    def compress_tool_output(output: str, max_chars: int = 500, max_tokens: int = None, dedupe: bool = True,
                              summarizer: Callable[[str, int], str] = None) -> str:
        """Truncates massive tool logs to save context tokens for high-tier supervisors.

        max_chars is the default (character) budget, kept for backwards compatibility. Pass
        max_tokens instead to budget by estimated token count (see _token_len - no external
        tokenizer, just a chunk-based heuristic that's noticeably closer than flat chars/4 for
        code/logs) - chars are only a rough proxy and the rest of this codebase already thinks
        in tokens. dedupe collapses repeated consecutive lines before truncating, since that's
        usually free space to reclaim before we start cutting real content.

        summarizer is optional and deliberately dumb on our end: it's just any
        (text, budget) -> str callable. We do NOT construct or hold an Agent/local model in
        here - that would mean importing core/router into compressor.py, which core.py would
        then import right back (circular import), and it would let this "just a truncation
        utility" silently trigger a local model load/swap behind the LocalModelSingleton's
        back. Whoever has the actual model (router.py, later company.py) builds a small
        closure/partial around whichever agent it already decided to use and hands that in -
        compressor never picks or owns a model itself. Only called when we'd truncate anyway,
        and if it throws or still comes back over budget we fall back to plain truncation
        rather than failing the whole compression.
        """
        if dedupe:
            output = ContextCompressor._dedupe_repeated_lines(output)

        use_tokens = max_tokens is not None
        budget = max_tokens if use_tokens else max_chars
        unit = "tokens" if use_tokens else "characters"
        total = ContextCompressor._token_len(output) if use_tokens else len(output)

        if budget <= 0:  # TODO decide if we want to raise an exception or just return the original output or nothing
            return f"... [Truncated {total} {unit}] ..."

        if total <= budget:
            return output

        if summarizer is not None:
            try:
                summary = summarizer(output, budget)
                summary_size = ContextCompressor._token_len(summary) if use_tokens else len(summary)
                if summary_size <= budget:
                    logger.debug("compress_tool_output: summarizer got it to %d/%d %s", summary_size, budget, unit)
                    return summary
                # Summarizer tried but still overshot - clamp it with plain truncation, and
                # explicitly no summarizer here so this can never recurse.
                logger.debug("compress_tool_output: summary still over budget (%d/%d %s), truncating it",
                             summary_size, budget, unit)
                return ContextCompressor.compress_tool_output(summary, max_chars=max_chars, max_tokens=max_tokens,
                                                                dedupe=False)
            except Exception as e:
                logger.warning("compress_tool_output: summarizer failed (%r), falling back to truncation", e)

        marker = f"... [Truncated {total - budget} {unit}] ..."
        marker_size = ContextCompressor._token_len(marker) if use_tokens else len(marker)

        # Guard against small budgets to prevent negative string slicing
        if budget <= marker_size:
            return marker

        half_allowed = (budget - marker_size) // 2

        if use_tokens:
            # Walk chunks to find the char offsets worth ~half_allowed estimated tokens each,
            # rather than a flat chars/token scale factor - see _char_offset_for_token_budget.
            head = output[:ContextCompressor._char_offset_for_token_budget(output, half_allowed)]
            tail = output[ContextCompressor._char_offset_for_token_budget(output, half_allowed, from_end=True):]
        else:
            head = output[:half_allowed]
            tail = output[-half_allowed:]

        # Snap both cuts onto line boundaries so we don't chop a line in half
        head = head[:ContextCompressor._snap_to_newline(head, len(head))]
        tail = tail[ContextCompressor._snap_to_newline(tail, 0):]

        return f"{head}\n{marker}\n{tail}"

    @staticmethod
    def compress_batch(outputs: list[str], total_budget: int, min_chars: int = 200,
                        summarizer: Callable[[str, int], str] = None) -> list[str]:
        """Water-fills one shared character budget across multiple tool outputs - small outputs
        pass through untouched, and whatever budget they don't use gets handed to the biggest
        offenders, instead of giving every item the same flat cutoff regardless of size.
        summarizer (see compress_tool_output) gets reused for every item that ends up needing it."""
        n = len(outputs)
        if n == 0 or total_budget <= 0:
            return outputs

        caps = [0] * n
        remaining_budget = total_budget
        remaining_count = n
        for i in sorted(range(n), key=lambda i: len(outputs[i])):
            fair_share = max(remaining_budget // remaining_count, min_chars)
            caps[i] = min(len(outputs[i]), fair_share)
            remaining_budget -= caps[i]
            remaining_count -= 1

        return [ContextCompressor.compress_tool_output(o, max_chars=caps[i], dedupe=True, summarizer=summarizer)
                for i, o in enumerate(outputs)]


@dataclass
class CompressionPolicy:
    """This is what "installing the compressor on an agent" actually means in this codebase.

    Long version, because this is the piece that decides how compression plugs into the rest
    of llmadapt and it's worth writing down the reasoning, not just the code:

    ContextCompressor itself is 100% @staticmethod - it has no per-agent state, it's just a
    bag of pure functions. So there's nothing to "instantiate one per agent" - the compressor
    is the same shared thing everywhere. What actually differs from agent to agent is the
    POLICY around when/how those functions get called: does this particular agent even bother
    compressing its tool output, how big a budget does it get, and is it allowed to spend a
    model call on summarizing instead of just truncating. That's all this class is - a small,
    plain-data config object that a router/company-tier setup can build once per agent rank
    and hand to the agent, instead of the agent (or the compressor) hardcoding any of this.

    Why compression should be opt-in per agent rather than global:
    Per the roadmap, C-Suite/Manager agents run on frontier APIs with big context windows and
    are more likely consuming already-delegated/summarized results than raw tool spew, so they
    often don't need this at all (or need a much bigger budget). Junior/Assistant/local-tier
    "worker" agents are the ones directly executing tools and are the ones with the tightest
    context windows, so they're the ones that actually need this switched on. Hence
    enabled=False by default - a plain Agent with no policy attached, or an explicitly
    disabled one, behaves exactly like it does today (no compression at all).

    Why summarizer defaults to None here specifically (re: the model-thrash discussion):
    A worker agent compressing ITS OWN tool output is usually the local model itself. Handing
    it a summarizer would mean that agent's own turn triggers a second local model
    load/generate call just to shrink its own scratch output - that's exactly the kind of
    thrash LocalModelSingleton exists to prevent, and it's rarely worth spending a model call
    on. So the "give a worker a CompressionPolicy" case should almost always leave summarizer
    unset (truncation/dedupe only, fast and free). Where a summarizer earns its keep is a tier
    ABOVE workers: e.g. a manager agent building a CompressionPolicy for itself to run
    compress_batch(..., summarizer=...) over several workers' results before passing one
    digested summary up the chain - that agent isn't competing with the workers for the same
    GPU slot, and condensing N outputs into one coherent summary is a much better use of a
    model call than shrinking a single blob of text.

    Fields:
        enabled              Master on/off switch. False = compress() is a no-op passthrough.
                              This is what lets "only worker agents use this" be a one-line
                              flip per agent instead of an if-rank-check scattered everywhere
                              tool output gets handled.
        max_chars             Same meaning as compress_tool_output's max_chars - the character
                              budget to truncate down to when max_tokens isn't set.
        max_tokens             Optional - if set, budget is measured in estimated tokens (no
                              external tokenizer, just _token_len's chunk-based heuristic)
                              instead of raw characters. See compress_tool_output's docstring
                              for why that's usually the better unit to budget in.
        summarizer            Optional (text, budget) -> str callable, forwarded straight
                              into compress_tool_output/compress_batch. Left as None for most
                              worker policies - see the thrash discussion above for why.
                              Whoever builds the CompressionPolicy (today: you by hand, later:
                              router.py by rank) is responsible for constructing this closure
                              around whichever agent/model it already decided to use -
                              CompressionPolicy, like ContextCompressor, never imports or
                              constructs an Agent itself. That keeps this file free of any
                              dependency on core.py/router.py, so there's no circular import
                              and no hidden model selection happening behind an agent's back.
        min_chars_to_bother    Skip the whole pipeline (dedupe + truncate/summarize) if the
                              output is already shorter than this. Avoids paying the cost of
                              a dedupe pass and length check on every tiny tool result when
                              most of them are already well under budget.

    Usage (illustrative - this is how an agent would use one, not code that runs today):

        worker_policy = CompressionPolicy(enabled=True, max_chars=1500, min_chars_to_bother=300)
        manager_policy = CompressionPolicy(enabled=True, max_chars=3000, summarizer=some_closure)

        # wherever a tool result is about to be added to conversation history:
        result = self.compression_policy.compress(raw_tool_result)
        self.conversation.history.append({"role": "tool", "content": result, ...})
    """
    enabled: bool = False
    max_chars: int = 500
    max_tokens: Optional[int] = None
    summarizer: Optional[Callable[[str, int], str]] = None
    min_chars_to_bother: int = 0

    def compress(self, output: str) -> str:
        """The one method an agent actually calls. Does nothing if this policy is disabled or
        the output's too small to bother with - otherwise delegates straight to
        ContextCompressor.compress_tool_output using this policy's settings."""
        if not self.enabled or len(output) <= self.min_chars_to_bother:
            return output
        return ContextCompressor.compress_tool_output(
            output, max_chars=self.max_chars, max_tokens=self.max_tokens, summarizer=self.summarizer)


if __name__ == '__main__':
    sample_code = '''class Agent:
    async def chat_async(self, user_input: str, custom_format_func: Callable[[list], list] = None, max_tokens: int = None,
                         thinking_visible: bool = True, spinner: bool = True) -> str:
        """DOCTSTRING TRY 1"""
        self.conversation.add_user_msg(user_input)

        return await self.generate_async(custom_format_func, max_tokens=max_tokens, thinking_visible=thinking_visible)

SUPPORTED_MODELS = [f"model-{i}" for i in range(50)]
'''
    print(ContextCompressor.code_to_stub(sample_code))

    print("\n--- dedupe + truncate demo ---")
    noisy_log = "connecting...\n" + "retrying...\n" * 40 + "connected!\n" + "x" * 2000
    print(ContextCompressor.compress_tool_output(noisy_log, max_chars=200))

    print("\n--- batch budget demo ---")
    outputs = ["short output", "y" * 5000, "z" * 1000]
    for c in ContextCompressor.compress_batch(outputs, total_budget=1000):
        print(len(c), repr(c[:60]))

    print("\n--- CompressionPolicy demo ---")
    no_policy = CompressionPolicy()  # disabled by default - e.g. a C-Suite agent
    worker_policy = CompressionPolicy(enabled=True, max_chars=200, min_chars_to_bother=50)
    big_result = "line\n" * 300
    print("no_policy   :", len(no_policy.compress(big_result)), "chars (untouched)")
    print("worker_policy:", len(worker_policy.compress(big_result)), "chars (compressed)")
    print("tiny result :", len(worker_policy.compress("ok")), "chars (below min_chars_to_bother, untouched)")
