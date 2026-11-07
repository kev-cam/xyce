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
  2. directory of the original top-level file
  3. caller-supplied include_dirs, then PYMS_INCLUDE_PATH
  4. this directory (vae/), which ships the Verilog-AMS standard headers
     disciplines.vams and constants.vams (Accellera Verilog-AMS 2.4.0,
     Annex D and E, verbatim: the notice permits verbatim copies; the
     same text openvaf-r carries built in), then
     /usr/local/share/xyce/PyMS/vae (installed)
The legacy names discipline.h, disciplines.h, disciplines.va, constants.h
and constants.va resolve to the standard headers when no file of that
name is on the search path (openvaf-r knows the same names).

A missing include is a warning, not an error (``preprocess: warning:
'name' not found (skipped)``; vamos reads that form).

Macros are expanded where they are used, with the definitions in force
at that point (a later `\\`define or `\\`undef of the name does not reach
back), and directives inside /* */ comments are comment text.

Command-line macros (``xyce_device_gen.py --define NAME[=VALUE]``, the
``PYMS_DEFINES`` environment variable, ``build_vae_so.py --define``) are
the equivalent of a ``\\`define`` line ahead of the top-level file, as
openvaf-r's ``-D``: a ``\\`define`` of the same name in the source replaces
it from its line on.  See parse_define_option().
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
# ``\\`define NAME(formals) body`` is function-like only when the '('
# follows the name immediately (IEEE 1364-2005 19.3.1, which Verilog-AMS
# 2.4 inherits: "the left parenthesis shall follow the text macro name
# immediately, with no space in between").  With a blank before it the
# macro is object-like and the parenthesis opens its body:
# ``\\`define TWO_Q (2 * 1.6021918e-19)``, BSIM-CMG's ``\\`define IPM
# (*units="" ...*) parameter real m=1.0 from(0.0:inf);``.
_DEFINE_RE = re.compile(
    r'^\s*`define\s+(\w+)(\([^)]*\))?\s*(.*)$')
_IDENT_RE = re.compile(r'^[A-Za-z_]\w*$')
_INCLUDE_RE = re.compile(r'^\s*`include\s+"([^"]+)".*$')
_IFDEF_RE = re.compile(r'^\s*`(ifdef|ifndef)\s+(\w+)\b.*$')
_ELSE_RE = re.compile(r'^\s*`else\b.*$')
_ELSIF_RE = re.compile(r'^\s*`elsif\s+(\w+)\b.*$')
_ENDIF_RE = re.compile(r'^\s*`endif\b.*$')
_UNDEF_RE = re.compile(r'^\s*`undef\s+(\w+)\b.*$')
# A conditional directive anywhere in a line (the LRM allows compiler
# directives anywhere in the text): `ifdef / `ifndef / `elsif NAME, `else,
# `endif.  Searched in the line with its string literals blanked.
_COND_ANY_RE = re.compile(r'`(ifdef|ifndef|elsif)\s+(\w+)\b|`(else|endif)\b')
_STR_LIT_RE = re.compile(r'"(?:[^"\\]|\\.)*"')

# Strip a trailing ``// ...`` line comment, respecting string literals.
_LINE_COMMENT_RE = re.compile(r'//')

# Token: a backtick-name optionally followed by an arg list.
# Captures NAME after the backtick.
_MACRO_USE_RE = re.compile(r'`([A-Za-z_]\w*)')

_DEFAULT_FALLBACKS = [
    os.path.dirname(os.path.abspath(__file__)),
    '/usr/local/share/xyce/PyMS/vae',
]

# Compiler directives (IEEE 1364 / Verilog-AMS) that are not macros: a
# `name of these left in a line is passed on (the parser skips the ones
# it does not handle), never reported as an undefined macro.
_DIRECTIVES = frozenset((
    'begin_keywords', 'celldefine', 'default_discipline', 'default_nettype',
    'default_transition', 'define', 'else', 'elsif', 'end_keywords',
    'endcelldefine', 'endif', 'ifdef', 'ifndef', 'include', 'line',
    'nounconnected_drive', 'pragma', 'resetall', 'timescale',
    'unconnected_drive', 'undef', 'undefineall', '__FILE__', '__LINE__',
))

# The legacy names of the standard headers (admsXml and openvaf-r supply
# them itself; HICUM L0 and L2 include "constants.h"): when the search path
# has no file of that name, the standard header PyMS ships stands in.
# Without it the `P_K, `P_Q, `M_PI of those models are undefined macros.
_ADMS_HEADERS = {
    'constants.h': 'constants.vams',
    'constants.va': 'constants.vams',
    'discipline.h': 'disciplines.vams',
    'disciplines.h': 'disciplines.vams',
    'disciplines.va': 'disciplines.vams',
}

# A string, a block-comment delimiter or a `name, for _scan_macro_uses.
_SCAN_RE = re.compile(r'"|/\*|\*/|`([A-Za-z_]\w*)')


class PreprocessError(Exception):
    pass


def parse_define_option(item: str) -> tuple[str, str]:
    """Split a command-line macro definition ``NAME`` or ``NAME=VALUE``
    into (name, body).  A bare NAME has an empty body, exactly like a
    ``\\`define NAME`` line (and openvaf-r's ``-D NAME``); VALUE is the
    body verbatim.  A NAME that is not an identifier is a PreprocessError
    naming the option."""
    name, sep, body = item.partition('=')
    name = name.strip()
    if not _IDENT_RE.match(name) or '\n' in item:
        raise PreprocessError(
            f'preprocess: --define {item!r}: {name!r} is not a macro name '
            f'(expected NAME or NAME=VALUE, NAME an identifier)')
    return name, (body.strip() if sep else '')


class PreprocessedText(str):
    """The preprocessed source text, with ``line_map``: for each of its lines
    (1-based line i is ``line_map[i - 1]``) the (file, line) of the source it
    came from.  Directive lines leave no line, an include brings the lines of
    the included file, and a macro invocation spanning several lines leaves
    its expansion on its first line and blank lines after it, so every line
    keeps its own origin.  The parser reads the map (Parser._loc_at), which is
    how every PyMS error names the true file and line."""

    def __new__(cls, text: str, line_map: list[tuple[str, int]]):
        obj = super().__new__(cls, text)
        obj.line_map = line_map
        return obj


def _fit_map(text: str, line_map: list[tuple[str, int]],
             fallback: tuple[str, int]) -> list[tuple[str, int]]:
    """line_map made to cover exactly the lines of text (an empty included
    text is one empty line in the joined output)."""
    n = text.count('\n') + 1
    if len(line_map) == n:
        return line_map
    if len(line_map) > n:
        return line_map[:n]
    pad = line_map[-1] if line_map else fallback
    return line_map + [pad] * (n - len(line_map))


def _unfold_continuations(source: str) -> str:
    # `\` immediately before a newline joins lines (common in macro bodies).
    return re.sub(r'\\\n', ' ', source)


def _logical_line_numbers(source: str) -> list[int]:
    """The 1-based source line on which each line of
    _unfold_continuations(source) starts."""
    numbers = []
    start = None
    for i, raw in enumerate(source.split('\n'), 1):
        if start is None:
            start = i
        if raw.endswith('\\'):
            continue  # joined with the next line
        numbers.append(start)
        start = None
    if start is not None:
        numbers.append(start)
    return numbers


def _scan_macro_uses(line: str, in_comment: bool) -> tuple[list[tuple[str, int]], bool]:
    """The `name uses of one line outside "..." strings and /* */
    comments, as (name, index just past the name); also whether the line
    ends inside a block comment."""
    uses = []
    i, n = 0, len(line)
    while i < n:
        if in_comment:
            j = line.find('*/', i)
            if j < 0:
                return uses, True
            i, in_comment = j + 2, False
            continue
        m = _SCAN_RE.search(line, i)
        if not m:
            break
        tok = m.group(0)
        if tok == '"':
            j = m.end()
            while j < n and line[j] != '"':
                j += 2 if line[j] == '\\' else 1
            i = j + 1
        elif tok == '/*':
            in_comment, i = True, m.end()
        elif tok == '*/':
            i = m.end()
        else:
            uses.append((m.group(1), m.end()))
            i = m.end()
    return uses, in_comment


def _strip_comments(line: str, in_comment: bool) -> tuple[str, bool]:
    """The code of one line without its ``// ...`` comment and its
    ``/* ... */`` comments (each replaced by a blank), respecting "..."
    string literals; ``in_comment`` says whether the line starts inside a
    block comment, and the second result whether it ends inside one.  A
    directive in a comment is comment text (IEEE 1364-2005 19: compiler
    directives are recognised outside comments only)."""
    out = []
    i, n = 0, len(line)
    while i < n:
        if in_comment:
            j = line.find('*/', i)
            if j < 0:
                break
            out.append(' ')
            i, in_comment = j + 2, False
            continue
        c = line[i]
        if c == '"':
            j = i + 1
            while j < n and line[j] != '"':
                j += 2 if line[j] == '\\' else 1
            out.append(line[i:j + 1])
            i = j + 1
        elif c == '/' and line.startswith('//', i):
            break
        elif c == '/' and line.startswith('/*', i):
            in_comment, i = True, i + 2
        else:
            out.append(c)
            i += 1
    return ''.join(out).rstrip(), in_comment


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
                 max_include_depth: int = 32, max_expansions: int = 100000,
                 defines: Optional[list[str]] = None):
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
        # Where each macro was defined (file, line), for error messages.
        self.define_locs: dict[str, tuple[str, int]] = {}
        # Command-line macros (--define NAME[=VALUE]): object-like, in force
        # from the first line of the top-level file; a `define of the same
        # name in the source replaces it from its line on.
        for item in defines or ():
            name, body = parse_define_option(item)
            self.defines[name] = (None, body)
            self.define_locs[name] = ('<command line>', 0)

    def _emitting(self) -> bool:
        return all(taking for taking, _ in self.cond_stack)

    def _cond_directive(self, kind: str, name: Optional[str], filename: str, lineno: int):
        """Apply one conditional directive to the conditional stack."""
        if kind in ('ifdef', 'ifndef'):
            outer_emit = self._emitting()
            cond = (name in self.defines)
            if kind == 'ifndef':
                cond = not cond
            taking = outer_emit and cond
            self.cond_stack.append((taking, taking))
        elif kind == 'elsif':
            if not self.cond_stack:
                raise PreprocessError(f'{filename}:{lineno}: `elsif outside `ifdef')
            _, any_taken = self.cond_stack[-1]
            outer_emit = all(t for t, _ in self.cond_stack[:-1])
            taking = outer_emit and not any_taken and (name in self.defines)
            self.cond_stack[-1] = (taking, any_taken or taking)
        elif kind == 'else':
            if not self.cond_stack:
                raise PreprocessError(f'{filename}:{lineno}: `else outside `ifdef')
            _, any_taken = self.cond_stack[-1]
            outer_emit = all(t for t, _ in self.cond_stack[:-1])
            taking = outer_emit and not any_taken
            self.cond_stack[-1] = (taking, any_taken or taking)
        else:  # endif
            if not self.cond_stack:
                raise PreprocessError(f'{filename}:{lineno}: `endif without `ifdef')
            self.cond_stack.pop()

    def _split_conditionals(self, line: str, filename: str, lineno: int) -> Optional[str]:
        """A line that holds a conditional directive beside other text
        (``r = `ifdef BIG 2k `else 1k `endif;``): apply its directives in
        order and return the text of the arms that are kept, joined into one
        line (so the line map is unchanged), or '' when nothing is kept.
        None when the line has no such directive or is a directive alone
        (the whole-line forms below handle those)."""
        blank = _STR_LIT_RE.sub(lambda m: '"' + ' ' * (len(m.group(0)) - 2) + '"', line)
        hits = list(_COND_ANY_RE.finditer(blank))
        if not hits:
            return None
        rest = blank[:hits[0].start()] + ''.join(
            blank[a.end():b.start()] for a, b in zip(hits, hits[1:])) + blank[hits[-1].end():]
        if len(hits) == 1 and not rest.strip():
            return None
        if line.lstrip().startswith('`define'):
            return None
        kept = []
        pos = 0
        for m in hits:
            if self._emitting():
                kept.append(line[pos:m.start()])
            self._cond_directive(m.group(1) or m.group(3), m.group(2), filename, lineno)
            pos = m.end()
        if self._emitting():
            kept.append(line[pos:])
        return ' '.join(k.strip() for k in kept if k.strip())

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
                    # Used without its arguments: _check_macro_uses reports
                    # the source-line case with file and line; this is the
                    # use whose arguments should follow a macro body.
                    raise PreprocessError(self._no_args_message(name))
                got = _match_balanced_args(text, j)
                if got is None:
                    dfile, dline = self.define_locs.get(name, ('<predefined>', 0))
                    raise PreprocessError(
                        f'preprocess: the argument list of `{name} (defined at '
                        f'{dfile}:{dline}) does not close: '
                        f'{text[m.start():m.start() + 80]!r}')
                arglist_text, end_idx = got
                args = _split_macro_args(arglist_text)
                substituted = self._substitute(body, params, args)
                # An invocation over several lines (PSP102's `P(\n ...\n);)
                # leaves its expansion on its first line and as many line
                # breaks after it as it spanned: the text after it stays on
                # its own source line (the line map of PreprocessedText).
                spanned = text.count('\n', m.start(), end_idx)
                if spanned or '\n' in substituted:
                    substituted = substituted.replace('\n', ' ') + '\n' * spanned
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

    def _check_macro_uses(self, line: str, filename: str, lineno: int,
                          in_comment: bool) -> bool:
        """Raise PreprocessError for a use of an undefined macro in one
        source line, directly or through the body of a macro the line uses
        (with the macros defined at this point: the LRM substitutes a macro
        where it is used).  Returns whether the line ends inside a block
        comment.  Left in place, an undefined macro was mis-read by the
        parser according to where it stood: dropped as an annotation,
        merged with the next statement and dropped with it, or turned into
        a node or a C++ name."""
        uses, in_comment = _scan_macro_uses(line, in_comment)
        for name, end in uses:
            if name in _DIRECTIVES:
                continue
            if name not in self.defines:
                raise PreprocessError(
                    f'{filename}:{lineno}: `{name} is not defined: no `define '
                    f'{name} precedes this use in the file or the files it '
                    f'includes (an undefined macro is an error in Verilog-A)')
            self._check_body(name, line, end, filename, lineno, {name})
        return in_comment

    def _check_body(self, name: str, text: str, end: int, filename: str,
                    lineno: int, seen: set[str]):
        """The macros a use of `name (in text, ending at end) brings in
        through its body must be defined too."""
        params, body = self.defines[name]
        if params is not None:
            # A function-like macro needs its argument list (IEEE 1364-2005
            # 19.3.1).  In a source line it must follow on the same line; at
            # the end of a macro body it comes from the text after the use,
            # which _expand_once checks.
            j = end
            while j < len(text) and text[j] in ' \t':
                j += 1
            if j >= len(text) and len(seen) > 1:
                return
            if j >= len(text) or text[j] != '(':
                raise PreprocessError(self._no_args_message(name, filename, lineno))
        for inner, inner_end in _scan_macro_uses(body, False)[0]:
            if inner in _DIRECTIVES or inner in seen:
                continue
            if inner not in self.defines:
                dfile, dline = self.define_locs.get(name, ('<predefined>', 0))
                raise PreprocessError(
                    f'{filename}:{lineno}: `{inner} is not defined: the body of '
                    f'`{name} (defined at {dfile}:{dline}) uses it, and no '
                    f'`define {inner} precedes this use of `{name} (an '
                    f'undefined macro is an error in Verilog-A)')
            self._check_body(inner, body, inner_end, filename, lineno, seen | {inner})

    def _no_args_message(self, name: str, filename: Optional[str] = None,
                         lineno: Optional[int] = None) -> str:
        dfile, dline = self.define_locs.get(name, ('<predefined>', 0))
        where = f'{filename}:{lineno}: ' if filename else 'preprocess: '
        return (f'{where}`{name} is a macro with arguments (defined at '
                f'{dfile}:{dline}) used without an argument list "(...)" on '
                f'the same line (a macro defined with a blank between its name '
                f'and "(" has no arguments)')

    def _substitute(self, body: str, params: list[str], args: list[str]) -> str:
        """Substitute each param name in body with the corresponding arg.
        Uses whole-word matching so a param `n` doesn't replace inside
        `name`."""
        # Pad args to match params length (empty if missing).
        while len(args) < len(params):
            args.append('')
        repl = {p: a for p, a in zip(params, args) if p}
        if not repl:
            return body
        # One pass over the body with every parameter in one pattern: an
        # argument is inserted once and never scanned again (substituting
        # one parameter after another replaced a later parameter's name
        # inside an earlier argument: `ADD2(b, 250) with parameters (a, b)
        # gave ((250) + 2*(250))), and a function replacement inserts the
        # argument literally (a backslash in it -- an escaped identifier --
        # is not a re.sub template escape).
        pat = re.compile(r'\b(?:' + '|'.join(re.escape(p) for p in
                                              sorted(repl, key=len, reverse=True)) + r')\b')
        return pat.sub(lambda m: repl[m.group(0)], body)

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
        line_numbers = _logical_line_numbers(src)
        src = _unfold_continuations(src)
        self.include_stack.append(filename)
        try:
            return self._process(src, os.path.dirname(os.path.abspath(filename)),
                                 filename, line_numbers)
        finally:
            self.include_stack.pop()

    def _process(self, src: str, current_dir: str, filename: str = '<source>',
                 line_numbers: Optional[list[int]] = None) -> PreprocessedText:
        # Lines are expanded where the macros are used: the ordinary lines
        # since the last directive that changes the macro table (`define,
        # `undef, `include) are expanded together, with the table as it
        # stands, before the directive applies.  Expanding a run of lines
        # at once keeps multi-line macro invocations (PSP102's
        # ``\\`P(\n  info=…\n  ...\n);``) working.
        out: list[str] = []
        pending: list[str] = []
        # The origin (file, line) of each line of out's chunks and of pending.
        out_maps: list[list[tuple[str, int]]] = []
        pending_locs: list[tuple[str, int]] = []

        def flush():
            if pending:
                text = self._expand_macros('\n'.join(pending))
                out.append(text)
                out_maps.append(_fit_map(text, list(pending_locs), pending_locs[-1]))
                pending.clear()
                pending_locs.clear()

        in_block = False
        for index, raw_line in enumerate(src.split('\n')):
            lineno = line_numbers[index] if line_numbers else index + 1
            line, in_block = _strip_comments(raw_line, in_block)
            stripped = line.strip()
            if not stripped:
                if self._emitting():
                    pending.append('')
                    pending_locs.append((filename, lineno))
                continue

            # A conditional directive in the middle of a line: the kept
            # arms are an ordinary line (one output line: the line map).
            if '`' in stripped:
                kept = self._split_conditionals(line, filename, lineno)
                if kept is not None:
                    if re.match(r'`(define|undef|include)\b', kept):
                        raise PreprocessError(
                            f'{filename}:{lineno}: a `define, `undef or `include after a '
                            f'conditional directive on the same line is not supported by '
                            f'PyMS; put it on a line of its own')
                    if kept:
                        self._check_macro_uses(kept, filename, lineno, False)
                        pending.append(kept)
                        pending_locs.append((filename, lineno))
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
                    raise PreprocessError(f'{filename}:{lineno}: `elsif outside `ifdef')
                _, any_taken = self.cond_stack[-1]
                outer_emit = all(t for t, _ in self.cond_stack[:-1])
                cond = m.group(1) in self.defines
                taking = outer_emit and not any_taken and cond
                self.cond_stack[-1] = (taking, any_taken or taking)
                continue

            if _ELSE_RE.match(stripped):
                if not self.cond_stack:
                    raise PreprocessError(f'{filename}:{lineno}: `else outside `ifdef')
                _, any_taken = self.cond_stack[-1]
                outer_emit = all(t for t, _ in self.cond_stack[:-1])
                taking = outer_emit and not any_taken
                self.cond_stack[-1] = (taking, any_taken or taking)
                continue

            if _ENDIF_RE.match(stripped):
                if not self.cond_stack:
                    raise PreprocessError(f'{filename}:{lineno}: `endif without `ifdef')
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
                    for p in params:
                        if not _IDENT_RE.match(p):
                            raise PreprocessError(
                                f'{filename}:{lineno}: `define {name}: the formal '
                                f'argument {p!r} is not an identifier (a blank '
                                f'between the name and "(" makes the parenthesis '
                                f'part of an object-like macro\'s text)')
                flush()
                self.defines[name] = (params, body)
                self.define_locs[name] = (filename, lineno)
                continue

            m = _UNDEF_RE.match(stripped)
            if m:
                flush()
                self.defines.pop(m.group(1), None)
                continue

            m = _INCLUDE_RE.match(stripped)
            if m:
                flush()
                rel = m.group(1)
                path = self._resolve_include(rel, current_dir)
                if path is None and rel in _ADMS_HEADERS:
                    path = self._resolve_include(_ADMS_HEADERS[rel], current_dir)
                if path is None:
                    if rel not in self.missing_includes:
                        self.missing_includes.add(rel)
                        sys.stderr.write(
                            f'preprocess: warning: {rel!r} not found '
                            f'(skipped)\n')
                    continue
                included_text = self.preprocess(path)
                out.append(included_text)
                out_maps.append(_fit_map(included_text,
                                         list(getattr(included_text, 'line_map', [])),
                                         (filename, lineno)))
                continue

            # Ordinary line — collected for flush().  An undefined macro is
            # an error here, where file and line are known (the expanded
            # text has neither).
            self._check_macro_uses(line, filename, lineno, False)
            pending.append(line)
            pending_locs.append((filename, lineno))
        flush()
        line_map: list[tuple[str, int]] = []
        for chunk_map in out_maps:
            line_map.extend(chunk_map)
        return PreprocessedText('\n'.join(out), line_map)


def preprocess_file(path: str, include_dirs: Optional[list[str]] = None,
                    defines: Optional[list[str]] = None) -> PreprocessedText:
    """Preprocess a Verilog-A file at path. Returns the fully-expanded source
    (a str; its ``line_map`` gives each line's source file and line).
    `defines`: command-line macros, ``NAME`` or ``NAME=VALUE`` each."""
    pp = _Preprocessor(path, include_dirs=include_dirs, defines=defines)
    return pp.preprocess(path)


def preprocess_source(source: str, filename: str = '<source>',
                      include_dirs: Optional[list[str]] = None,
                      defines: Optional[list[str]] = None) -> PreprocessedText:
    """Preprocess a Verilog-A source string. `filename` is used to
    resolve relative `\\`include paths."""
    pp = _Preprocessor(filename, include_dirs=include_dirs, defines=defines)
    line_numbers = _logical_line_numbers(source)
    src = _unfold_continuations(source)
    pp.include_stack.append(filename or '<source>')
    try:
        return pp._process(src, os.path.dirname(os.path.abspath(filename))
                           if filename else os.getcwd(),
                           filename or '<source>', line_numbers)
    finally:
        pp.include_stack.pop()
