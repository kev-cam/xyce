"""Verilog-A preprocessor: ``\\`include``, ``\\`define`` (object- and
function-like), ``\\`ifdef``/``\\`ifndef``/``\\`else``/``\\`endif``,
``\\`undef``.

The parser in parser.py used to silently skip every `\\`-directive. That
worked for toy models whose params lived in the .va itself, but every
real compact model in /usr/local/share/xyce/verilog-a uses either:

  * an external macro-definition file (Common103_macrodefs.include,
    bsimcmg_main.va, etc.) brought in via `\\`include, OR
  * macros for "model parameter real" / "instance parameter real"
    declarations (`\\`IPRnb(W, ...)`, `\\`MPRcc(...)`, etc.)

so the parser ended up with zero parameters on the resulting Module —
which broke device-card param dispatch on every M/Q/D card in the
regression suite.

This module emits a single source string ready to hand to tokenize().
It is intentionally textual: no AST, no lex-token-aware substitution.
That's good enough for the compact-model macro libraries shipped with
Xyce, which all expand to flat parameter declarations.

Search path for `\\`include:
  1. directory of the file currently being preprocessed
  2. /usr/local/src/xyce/utils/PyMS/vae  (constants.vams etc.)
  3. directory of the original top-level file
  4. /usr/local/share/xyce/PyMS/vae  (installed)

A missing include is a warning, not an error — disciplines.vams in
particular is referenced everywhere but PyMS doesn't ship one.
"""

from __future__ import annotations

import os
import re
import sys
from typing import Optional


# Directive regexes — trailing content after the directive (e.g.
# `` `endif // foo ``) is ignored; without that, .include files that
# decorate every `` `endif `` with a comment would never close their
# conditionals and the rest of the file would be silently dropped.
_DEFINE_RE = re.compile(
    r'^\s*`define\s+(\w+)\s*(\([^)]*\))?\s*(.*)$')
_INCLUDE_RE = re.compile(r'^\s*`include\s+"([^"]+)".*$')
_IFDEF_RE = re.compile(r'^\s*`(ifdef|ifndef)\s+(\w+)\b.*$')
_ELSE_RE = re.compile(r'^\s*`else\b.*$')
_ELSIF_RE = re.compile(r'^\s*`elsif\s+(\w+)\b.*$')
_ENDIF_RE = re.compile(r'^\s*`endif\b.*$')
_UNDEF_RE = re.compile(r'^\s*`undef\s+(\w+)\b.*$')

# Strip a trailing ``// ...`` line comment, respecting string literals.
_LINE_COMMENT_RE = re.compile(r'//')

# Token: a backtick-name optionally followed by an arg list.
# Captures NAME after the backtick.
_MACRO_USE_RE = re.compile(r'`([A-Za-z_]\w*)')

_DEFAULT_FALLBACKS = [
    os.path.dirname(os.path.abspath(__file__)),
    '/usr/local/share/xyce/PyMS/vae',
]


class PreprocessError(Exception):
    pass


def _unfold_continuations(source: str) -> str:
    # `\` immediately before a newline joins lines (common in macro bodies).
    return re.sub(r'\\\n', ' ', source)


def _strip_line_comment(line: str) -> str:
    """Strip ``// ...`` trailing comment from a line, respecting "..."
    string literals. Leaves /* ... */ alone (they're rare in compact
    models and span lines — the tokenizer handles them)."""
    out = []
    i = 0
    in_str = False
    while i < len(line):
        c = line[i]
        if in_str:
            out.append(c)
            if c == '\\' and i + 1 < len(line):
                out.append(line[i + 1])
                i += 2
                continue
            if c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
                out.append(c)
            elif c == '/' and i + 1 < len(line) and line[i + 1] == '/':
                break
            else:
                out.append(c)
        i += 1
    return ''.join(out).rstrip()


def _split_macro_args(arglist: str) -> list[str]:
    """Split a macro argument list (without surrounding parens) on
    top-level commas — commas inside nested parens / braces / brackets
    / string literals belong to the argument."""
    out, depth, buf = [], 0, []
    in_str = False
    i = 0
    while i < len(arglist):
        ch = arglist[i]
        if in_str:
            buf.append(ch)
            if ch == '\\' and i + 1 < len(arglist):
                buf.append(arglist[i + 1])
                i += 2
                continue
            if ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
            buf.append(ch)
        elif ch in '([{':
            depth += 1
            buf.append(ch)
        elif ch in ')]}':
            depth -= 1
            buf.append(ch)
        elif ch == ',' and depth == 0:
            out.append(''.join(buf).strip())
            buf = []
        else:
            buf.append(ch)
        i += 1
    if buf:
        out.append(''.join(buf).strip())
    return out


def _match_balanced_args(text: str, start: int) -> Optional[tuple[str, int]]:
    """text[start] must be '(' — return (arglist_without_parens, end_idx_just_past_close).
    Returns None if the parens never close. Respects "..." strings so
    a ')' inside a string literal doesn't close the arg list early."""
    assert text[start] == '('
    depth, i = 0, start
    in_str = False
    while i < len(text):
        c = text[i]
        if in_str:
            if c == '\\' and i + 1 < len(text):
                i += 2
                continue
            if c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
            if depth == 0:
                return text[start + 1:i], i + 1
        i += 1
    return None


class _Preprocessor:
    def __init__(self, top_filename: str, include_dirs: Optional[list[str]] = None,
                 max_include_depth: int = 32, max_expansions: int = 100000):
        self.top_dir = os.path.dirname(os.path.abspath(top_filename)) \
            if top_filename else os.getcwd()
        # ADMS-style compatibility: many compact-model .va files gate
        # their xyce* attribute decorations behind `ifdef insideADMS`
        # (HICUM L2, FBH HBT, BSIM-CMG branches with $simparam, etc.).
        # When insideADMS is unset the `ATTR macros expand to nothing,
        # so xyceModelGroup vanishes — predefining it makes those
        # blocks visible. We act as a faithful ADMS replacement, so
        # this is the correct posture.
        self.defines: dict[str, tuple[Optional[list[str]], str]] = {
            'insideADMS': (None, '1'),
            # VBIC 1.3 and similar compact-model trees gate the
            # XYCE_ATTR macro definition behind `ifdef
            # __XYCE_COMPACT_MODELING__`. Predefining it picks up
            # the variant that actually expands to (* xyceModelGroup
            # ... *) rather than the empty no-op.
            '__XYCE_COMPACT_MODELING__': (None, '1'),
        }
        # Cond stack entries: (taking, any_taken)
        # taking=True means we emit lines for the current arm; any_taken
        # tracks whether any prior arm of this if/elsif chain was taken
        # (so subsequent `else doesn't re-emit).
        self.cond_stack: list[tuple[bool, bool]] = []
        self.include_stack: list[str] = []
        self.max_include_depth = max_include_depth
        self.expansions = 0
        self.max_expansions = max_expansions
        # Fallback search dirs (in order)
        # Search path: caller-supplied dirs, then PYMS_INCLUDE_PATH
        # (colon-separated, recurse one level so an "IHP-Open-PDK" root
        # finds includes under any of its module subdirectories), then
        # compile-time fallbacks.
        env_dirs = []
        env_var = os.environ.get('PYMS_INCLUDE_PATH', '')
        for entry in env_var.split(os.pathsep):
            entry = entry.strip()
            if not entry:
                continue
            env_dirs.append(entry)
            # Auto-expand subdirectories one level deep so a single
            # ``PYMS_INCLUDE_PATH=/foo/IHP-Open-PDK`` picks up
            # IHP-Open-PDK/ihp-sg13g2/libs.tech/verilog-a/r3_cmc/, etc.
            if os.path.isdir(entry):
                for root, dirs, _ in os.walk(entry):
                    for d in dirs:
                        env_dirs.append(os.path.join(root, d))
                    # Cap depth to avoid scanning all of /usr — 4
                    # levels covers IHP's deepest nest.
                    cur_depth = root[len(entry):].count(os.sep)
                    if cur_depth >= 4:
                        dirs.clear()
        self.search_dirs = (list(include_dirs or [])
                            + env_dirs + _DEFAULT_FALLBACKS)
        self.missing_includes: set[str] = set()

    def _emitting(self) -> bool:
        return all(taking for taking, _ in self.cond_stack)

    def _resolve_include(self, rel: str, current_dir: str) -> Optional[str]:
        candidates = [os.path.join(current_dir, rel), os.path.join(self.top_dir, rel)]
        for d in self.search_dirs:
            candidates.append(os.path.join(d, rel))
        for c in candidates:
            if os.path.exists(c):
                return c
        return None

    def _expand_macros(self, text: str) -> str:
        """Repeatedly expand `name and `name(args) tokens until none of
        the defined names remain (or we exceed max_expansions)."""
        for _ in range(64):
            new_text, changed = self._expand_once(text)
            if not changed:
                return new_text
            text = new_text
        # Reached the cap — return what we have (best-effort).
        return text

    def _expand_once(self, text: str) -> tuple[str, bool]:
        changed = False
        out = []
        i = 0
        while i < len(text):
            m = _MACRO_USE_RE.search(text, i)
            if not m:
                out.append(text[i:])
                break
            out.append(text[i:m.start()])
            name = m.group(1)
            after = m.end()
            if name not in self.defines:
                # Leave alone — may be a directive (`ifdef etc.) or an
                # unknown identifier we mustn't mangle.
                out.append(m.group(0))
                i = after
                continue
            params, body = self.defines[name]
            if params is not None:
                # Function-like — need '(' next (whitespace allowed).
                j = after
                while j < len(text) and text[j] in ' \t':
                    j += 1
                if j >= len(text) or text[j] != '(':
                    # Defined as function-like but used without args:
                    # treat as literal (rare; safer than dropping).
                    out.append(m.group(0))
                    i = after
                    continue
                got = _match_balanced_args(text, j)
                if got is None:
                    out.append(m.group(0))
                    i = after
                    continue
                arglist_text, end_idx = got
                args = _split_macro_args(arglist_text)
                substituted = self._substitute(body, params, args)
                out.append(substituted)
                i = end_idx
                changed = True
                self.expansions += 1
                if self.expansions > self.max_expansions:
                    raise PreprocessError(
                        'preprocess: macro expansion limit exceeded — '
                        'likely recursive macro')
            else:
                out.append(body)
                i = after
                changed = True
                self.expansions += 1
        return ''.join(out), changed

    def _substitute(self, body: str, params: list[str], args: list[str]) -> str:
        """Substitute each param name in body with the corresponding arg.
        Uses whole-word matching so a param `n` doesn't replace inside
        `name`."""
        # Pad args to match params length (empty if missing).
        while len(args) < len(params):
            args.append('')
        repl = dict(zip(params, args))
        # Build one combined regex with named groups would be tidier,
        # but params are usually <10 — simple loop is fine.
        for p, a in repl.items():
            if not p:
                continue
            body = re.sub(rf'\b{re.escape(p)}\b', a, body)
        return body

    def preprocess(self, filename: str) -> str:
        """Top-level entry. Returns the expanded source text for filename."""
        if filename in self.include_stack:
            raise PreprocessError(
                f'preprocess: recursive include detected: '
                f'{" -> ".join(self.include_stack + [filename])}')
        if len(self.include_stack) >= self.max_include_depth:
            raise PreprocessError(
                f'preprocess: include depth exceeded at {filename}')
        with open(filename, 'r', errors='replace') as f:
            src = f.read()
        src = _unfold_continuations(src)
        self.include_stack.append(filename)
        try:
            return self._process(src, os.path.dirname(os.path.abspath(filename)))
        finally:
            self.include_stack.pop()

    def _process(self, src: str, current_dir: str) -> str:
        out: list[str] = []
        for raw_line in src.split('\n'):
            line = _strip_line_comment(raw_line)
            stripped = line.strip()
            if not stripped:
                if self._emitting():
                    out.append('')
                continue

            # Conditional directives — checked first, since they affect
            # whether subsequent directives even apply.
            m = _IFDEF_RE.match(stripped)
            if m:
                kind, name = m.group(1), m.group(2)
                outer_emit = self._emitting()
                cond = (name in self.defines)
                if kind == 'ifndef':
                    cond = not cond
                taking = outer_emit and cond
                self.cond_stack.append((taking, taking))
                continue

            m = _ELSIF_RE.match(stripped)
            if m:
                if not self.cond_stack:
                    raise PreprocessError('preprocess: `elsif outside `ifdef')
                _, any_taken = self.cond_stack[-1]
                outer_emit = all(t for t, _ in self.cond_stack[:-1])
                cond = m.group(1) in self.defines
                taking = outer_emit and not any_taken and cond
                self.cond_stack[-1] = (taking, any_taken or taking)
                continue

            if _ELSE_RE.match(stripped):
                if not self.cond_stack:
                    raise PreprocessError('preprocess: `else outside `ifdef')
                _, any_taken = self.cond_stack[-1]
                outer_emit = all(t for t, _ in self.cond_stack[:-1])
                taking = outer_emit and not any_taken
                self.cond_stack[-1] = (taking, any_taken or taking)
                continue

            if _ENDIF_RE.match(stripped):
                if not self.cond_stack:
                    raise PreprocessError('preprocess: `endif without `ifdef')
                self.cond_stack.pop()
                continue

            if not self._emitting():
                # Skipped branch — but still need to count nested ifs
                # (handled above via cond_stack push/pop on directives).
                continue

            m = _DEFINE_RE.match(stripped)
            if m:
                name = m.group(1)
                paren = m.group(2)
                body = m.group(3)
                params = None
                if paren is not None:
                    inner = paren[1:-1]
                    params = [p.strip() for p in inner.split(',')] \
                        if inner.strip() else []
                self.defines[name] = (params, body)
                continue

            m = _UNDEF_RE.match(stripped)
            if m:
                self.defines.pop(m.group(1), None)
                continue

            m = _INCLUDE_RE.match(stripped)
            if m:
                rel = m.group(1)
                path = self._resolve_include(rel, current_dir)
                if path is None:
                    if rel not in self.missing_includes:
                        self.missing_includes.add(rel)
                        sys.stderr.write(
                            f'preprocess: warning: {rel!r} not found '
                            f'(skipped)\n')
                    continue
                included_text = self.preprocess(path)
                out.append(included_text)
                continue

            # Ordinary line — emit raw. Macro expansion happens once at
            # the end, on the joined output, so multi-line macro
            # invocations (PSP102's ``\\`P(\n  info=…\n  ...\n);``) and
            # nested expansions across line boundaries both work.
            out.append(line)
        joined = '\n'.join(out)
        return self._expand_macros(joined)


def preprocess_file(path: str, include_dirs: Optional[list[str]] = None) -> str:
    """Preprocess a Verilog-A file at path. Returns the fully-expanded source."""
    pp = _Preprocessor(path, include_dirs=include_dirs)
    return pp.preprocess(path)


def preprocess_source(source: str, filename: str = '<source>',
                      include_dirs: Optional[list[str]] = None) -> str:
    """Preprocess a Verilog-A source string. `filename` is used to
    resolve relative `\\`include paths."""
    pp = _Preprocessor(filename, include_dirs=include_dirs)
    src = _unfold_continuations(source)
    pp.include_stack.append(filename or '<source>')
    try:
        return pp._process(src, os.path.dirname(os.path.abspath(filename))
                           if filename else os.getcwd())
    finally:
        pp.include_stack.pop()
