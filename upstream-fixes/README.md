# Suggested upstream fixes

Standalone patches from this fork that are candidates for upstream Xyce
(https://github.com/Xyce/Xyce). Each patch is self-contained, touches no
fork-specific code (PyMS, smp-load, oracle), and applies to stock Xyce with:

```
git am upstream-fixes/<patch>          # preserves message + attribution
# or
patch -p1 < upstream-fixes/<patch>
```

| Patch | What | Severity | Status |
|-------|------|----------|--------|
| 0001-IO-parse-indented-netlist-statements-instead-of-sile.patch | Netlist readers treat any leading-whitespace line as a comment and SILENTLY drop it — indented device/directive lines (ubiquitous in decks written for ngspice/HSpice/LTspice, which all ignore leading whitespace) vanish with no diagnostic; first symptom is typically "undefined symbols in .PRINT" on an otherwise-valid deck. Fix makes both readers (`NextChar_`, `skipCommentsAndBlankLines_`) rewind to line start and parse the statement; indented `*`/`;` comments, `+` continuations, and blank lines keep their existing behavior. Both readers must agree byte-for-byte because pass 2 re-reads the deck from recorded stream positions. | silent data loss | proposed |

Verification for 0001: indented R/C/X lines after device, dot, and blank
lines; chained indented instances; indented comments still skipped; column-0
and indented `+` continuations unchanged; whitespace-only lines; tab
indentation; and a fully indented 10112-FET ISCAS-85 C6288 deck that
previously collapsed to 2 devices now parses completely and computes
0xFFFF x 0xFFFF = 0xFFFE0001.
