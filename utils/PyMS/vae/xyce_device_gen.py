#!/usr/bin/env python3
"""
PyMS Xyce device class generator.

Takes a parsed Verilog-A module and produces a complete Xyce device plugin
(.so) that can be loaded via -plugin or .hdl.

The generated device class:
  - Registers with Xyce's device factory
  - Handles node allocation (external + internal)
  - Parses model/instance parameters from netlists
  - At eval time, calls a VAE-compiled .so for model computation
  - Stamps F/Q/dFdV/dQdV into Xyce's DAE system

Usage:
    python3 xyce_device_gen.py input.va [--output dir/] [--define NAME[=VALUE]]...
    python3 xyce_device_gen.py --version
"""

import os
import re
import sys
import subprocess
from pathlib import Path

# Add parent for vae imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from vae.parser import parse_verilog_a, parse_file, NodeKind, ContribKind, ParseError
from vae.ginac_emitter import VAE_STATE_CAP, VAE_STATE_DECL, VAE_STATE_ABI
from vae.preprocess import PreprocessError
from vae.va_const import (ConstExprError, evaluate_param_defaults, dependency_order,
                          to_cxx, param_kind, XYCE_SIMPARAM_DEFAULTS, IPOW_CXX,
                          TEMPERATURE_KEY)


class DeviceGenError(ValueError):
    """A .va PyMS cannot turn into a device; the message names file, line and construct."""


def _param_loc(mod, pname):
    """``file:line`` of the declaration of parameter ``pname`` (the parser's
    Param.loc, from the preprocessor's line map: the true file, an included
    one too), or None."""
    for p in getattr(mod, 'params', None) or ():
        if p.name == pname:
            loc = getattr(p, 'loc', None)
            if loc is not None and loc.line:
                return str(loc)
    return None


def _param_source_line(va_path, pname):
    """The 1-based line of the declaration of parameter ``pname`` in the file
    itself, by a search of its text (for a module without Param.loc), or None
    when it is not declared there."""
    try:
        with open(va_path, errors='replace') as f:
            for i, line in enumerate(f, 1):
                if re.search(r'\b(?:parameter|localparam)\b.*\b' + re.escape(pname)
                             + r'\s*(?:\[|=|;|from\b|exclude\b|\(\*|`)', line):
                    return i
    except OSError:
        pass
    return None


def _locate_param_error(va_path, msg, mod=None):
    """Prefix a va_const message ('parameter X ...') with file:line."""
    m = re.search(r'parameter (\w+)', msg)
    where = _param_loc(mod, m.group(1)) if m else None
    if where:
        return f"{where}: {msg}"
    line = _param_source_line(va_path, m.group(1)) if m else None
    return f"{va_path}:{line}: {msg}" if line else f"{va_path}: {msg}"


def scan_va_attributes(va_path: str, defines=None, module_name=None, siblings=True) -> dict:
    """Extract ``(* key="value" ... *)`` attributes from a Verilog-A
    module declaration.  `module_name`: only the declaration of that module
    (or paramset) counts (1.1.5: the xyceModelGroup of a later module of the
    file no longer lands on the first, m5_second_has_group_attr); None: the
    first one that has them.  `defines`: the command-line macros of the build
    (--define / PYMS_DEFINES), so that attributes behind an ``\\`ifdef``
    are seen as the device build sees them.

    The compact-model .va files in the install tree write these in five
    different shapes — handle them all:

      1. ``module foo(a,b) (* xyceModelGroup="MOSFET" ... *) ;``
         (Accellera-standard, post-decl, comma-separated)
      2. ``(* xyceModelGroup="..." *)  module foo(a,b);``
         (pre-decl — EKV 2.6, toys/capacitor)
      3. ``module foo(a,b) `ATTR(xyceModelGroup="..." ...);``
         (HICUM L2 / Diode CMC / MVS / FBH-HBT — macro that expands
         to ``(* txt *)``)
      4. Module declaration lives behind a ``\\`include`` — preprocess
         first and the include's expansion brings the decl into scope
         (BSIM-CMG 107 / 108).
      5. Attributes declared in a sibling .include file with no module
         on the entry-point line (rare; not handled here).

    To handle 3+4 in one shot we run the preprocessor and search the
    *expanded* source for both pre- and post-decl ``(* ... *)`` blocks.
    Whichever set contains an ``xyceModelGroup`` wins.
    """
    attrs = {}
    if not va_path or not os.path.exists(va_path):
        return attrs
    import re
    try:
        # Late import so the parser/codegen modules can be imported
        # without dragging in the preprocessor when callers don't need it.
        from vae.preprocess import preprocess_file
        text = preprocess_file(va_path, defines=defines)
    except Exception:
        # Fall back to raw source — better to find *some* attrs than
        # to fail outright if preprocess hits something unexpected.
        with open(va_path, 'r', errors='replace') as f:
            text = f.read()

    # Candidate windows: each is a ``(* ... *)`` block adjacent to a
    # ``module`` or ``paramset`` keyword. The block can span multiple
    # lines and may contain comma- or space-separated key=value pairs.
    candidates = []
    # Post-decl: ``module name(ports) (* ... *)``
    def mine(name):
        return module_name is None or name == module_name
    for m in re.finditer(
            r'module\s+(\w+)\s*\([^)]*\)\s*\(\*(.*?)\*\)', text, re.DOTALL):
        if mine(m.group(1)):
            candidates.append(m.group(2))
    # Pre-decl: ``(* ... *) module name(ports)`` (the block right before the
    # keyword: the last ``(*`` of the text matched)
    for m in re.finditer(
            r'\(\*(.*?)\*\)\s*module\s+(\w+)\s*\(', text, re.DOTALL):
        if mine(m.group(2)):
            candidates.append(m.group(1).split('(*')[-1])
    # Pre-decl on paramset: ``(* ... *) paramset NAME UNDERLYING``
    for m in re.finditer(
            r'\(\*(.*?)\*\)\s*paramset\s+(\w+)\s+\w+', text, re.DOTALL):
        if mine(m.group(2)):
            candidates.append(m.group(1).split('(*')[-1])

    # Pick the first candidate whose body actually mentions
    # ``xyceModelGroup`` — generic ``(* desc="..." *)`` blocks on
    # parameter declarations are not what we want here.
    for body in candidates:
        if 'xyceModelGroup' not in body and 'xyceLevelNumber' not in body:
            continue
        attrs = {}
        for am in re.finditer(r'(\w+)\s*=\s*"([^"]*)"', body):
            attrs[am.group(1)] = am.group(2)
        if 'xyceModelGroup' in attrs or 'xyceLevelNumber' in attrs:
            return attrs

    # Sibling-directory fallback. Several entry-point .va files in
    # the install tree are variants of a base model that lack the
    # xyceModelGroup/xyceLevelNumber attributes outright:
    #   - psp102/psp102b.va, psp102e.va are binning/local-model
    #     variants of psp102/psp102.va
    #   - fbh_hbt-2.1/fbhhbt-2.1.va is the plain variant; the
    #     attributes live in fbhhbt-2.1_nonoise_limited_inductors_typed.va
    # The variants share the same device family as the sibling, so
    # if we found nothing on the entry-point itself, scan adjacent
    # .va files and reuse the first xyce* attrs we find.
    sibling_attrs = {}
    try:
        d = os.path.dirname(va_path) if siblings else ''
        if d and os.path.isdir(d):
            entry_name = os.path.basename(va_path)
            for sib in sorted(os.listdir(d)):
                if not sib.endswith('.va') or sib == entry_name:
                    continue
                # Only a variant of the same model inherits: one file
                # name extends the other (psp102b.va / psp102.va,
                # fbhhbt-2.1.va / fbhhbt-2.1_..._typed.va). An unrelated
                # file in the same directory must not lend its group:
                # toys/rlc.va took toys/capacitor.va's Capacitor level 6
                # and ran as a bare capacitor, its inductor lost
                # (BUGS.md CHANGES 1.1.1).
                e_stem = os.path.splitext(entry_name)[0].lower()
                s_stem = os.path.splitext(sib)[0].lower()
                if not (e_stem.startswith(s_stem) or s_stem.startswith(e_stem)):
                    continue
                sib_path = os.path.join(d, sib)
                # Quick raw-text peek: avoid recursive preprocess on
                # every sibling, just grep for the attribute name in
                # source. Good enough — false positives get filtered
                # by the regex below.
                try:
                    with open(sib_path, 'r', errors='replace') as f:
                        sib_text = f.read()
                except OSError:
                    continue
                if 'xyceModelGroup' not in sib_text \
                        and 'xyceLevelNumber' not in sib_text:
                    continue
                # Reuse our own logic to extract — recurse once.
                got = scan_va_attributes(sib_path, defines)
                if got.get('xyceModelGroup') or got.get('xyceLevelNumber'):
                    sibling_attrs = got
                    break
    except Exception:
        pass
    if sibling_attrs:
        return sibling_attrs

    # No xyce* attrs found — return whatever the first generic
    # candidate yielded (still better than {}).
    if candidates:
        for am in re.finditer(r'(\w+)\s*=\s*"([^"]*)"', candidates[0]):
            attrs[am.group(1)] = am.group(2)
    return attrs


# --- behavioral-stamp emitters for the QSPICE C/L loss shorthands ----------
# Emit C++ that adds the RSER (series) / RPAR (parallel) loss branches to the
# simple-R/L/C behavioral stamp. They assume the enclosing stamp has already
# emitted `double V_drop = V(p) - V(n);` and that li_<node> members exist.
_NOISE_RE = re.compile(r'(?<![\w$])(white_noise|flicker_noise|noise_table_log|noise_table)\s*\(')


def noise_sources(mod):
    """The noise source calls of a module (its analog block and analog
    functions), as (function name, source line or 0), first use of each name
    and line, in source order (1.1.8, BUGS.md A19: PyMS drops them)."""
    found = []

    def scan(text, loc):
        for m in _NOISE_RE.finditer(text or ''):
            k = (m.group(1), getattr(loc, 'line', 0) or 0)
            if k not in found:
                found.append(k)

    def walk(node):
        if node is None:
            return
        for t in (node.expr, node.text, node.condition, node.lhs):
            scan(t, node.loc)
        for ch in (node.children or []):
            walk(ch)
        walk(node.else_body)

    walk(mod.analog_block)
    for f in getattr(mod, 'functions', []) or []:
        walk(f.body)
    return found


def _emit_rpar(c, rpar_ref, a, b):
    """Parallel leakage RPAR directly across the +/- terminals (no node)."""
    c.append(f'    double Rpar = {rpar_ref}; if (Rpar < 1e-12) Rpar = 1e-12;')
    c.append('    double Gpar = 1.0 / Rpar;')
    c.append(f'    F_[{a}] += Gpar * V_drop;')
    c.append(f'    F_[{b}] -= Gpar * V_drop;')
    c.append(f'    dFdx_[{a}][{a}] += Gpar;')
    c.append(f'    dFdx_[{a}][{b}] -= Gpar;')
    c.append(f'    dFdx_[{b}][{a}] -= Gpar;')
    c.append(f'    dFdx_[{b}][{b}] += Gpar;')


def _emit_rser(c, rser_ref, p0, mid_name, a, m):
    """Series ESR between the + terminal (p0, slot a) and the internal node
    (mid_name, slot m); the reactive element then sits between mid and -."""
    c.append(f'    double Rser = {rser_ref}; if (Rser < 1e-12) Rser = 1e-12;')
    c.append('    double Gser = 1.0 / Rser;')
    c.append(f'    double Vpm = (*solVec)[li_{p0}] - (*solVec)[li_{mid_name}];')
    c.append(f'    F_[{a}] += Gser * Vpm;')
    c.append(f'    F_[{m}] -= Gser * Vpm;')
    c.append(f'    dFdx_[{a}][{a}] += Gser;')
    c.append(f'    dFdx_[{a}][{m}] -= Gser;')
    c.append(f'    dFdx_[{m}][{a}] -= Gser;')
    c.append(f'    dFdx_[{m}][{m}] += Gser;')


def generate_device_cpp(mod, xyce_src_dir: str, va_path: str = '',
                        defines=None, module_arg: bool = False,
                        primary: str = None) -> tuple[str, str]:
    """Generate Xyce device C++ source (.h and .C) from parsed VA module.

    `defines`: the command-line macros the module was parsed with; the
    shell passes them on to build_vae_so.py and keys the math-library
    cache on them.  `module_arg`: the module is not the first of its file
    (a file with several modules): the shell names it to build_vae_so.py,
    and its Xyce attributes are its own (Module.attributes), never the
    file's first module's (`primary`, its name).  A further module whose own
    declaration carries an xyceModelGroup / xyceLevelNumber registers that
    group and level (1.1.6, B21: m6b_dd_second; 1.1.5 registered every
    further module as a Y device only), unless it is the first module's
    group and level, which the first module registers: then a Y device of
    its own name.  Returns (header_source, impl_source) strings.
    """
    name = mod.name  # e.g. 'rlc', 'PSP103_VA'
    NAME = name.upper()

    # Extract Xyce ADMS-style attributes from the VA source
    if module_arg:
        # a further module: the attributes of its own declaration only (no
        # sibling file lends it a group); the first module's group and level
        # stay the first module's
        va_attrs = scan_va_attributes(va_path, defines, module_name=mod.name, siblings=False)
        if va_attrs.get('xyceModelGroup') or va_attrs.get('xyceLevelNumber'):
            first = scan_va_attributes(va_path, defines, module_name=primary) if primary else {}
            key = lambda a: ((a.get('xyceModelGroup') or '').upper(), str(a.get('xyceLevelNumber') or ''))
            if primary and key(first) == key(va_attrs):
                print(f"xyce_device_gen: note: module {mod.name} of {va_path} declares the model "
                      f"group {va_attrs.get('xyceModelGroup', '')} level "
                      f"{va_attrs.get('xyceLevelNumber', '')} of the file's first module "
                      f"{primary}: it is a Y device of its own name")
                va_attrs = {}
        else:
            va_attrs = {}
    else:
        va_attrs = scan_va_attributes(va_path, defines, module_name=mod.name)
    model_group = va_attrs.get('xyceModelGroup', '')  # e.g. "MOSFET", "Diode"
    level_number = va_attrs.get('xyceLevelNumber', '')  # e.g. "77"
    type_variable = va_attrs.get('xyceTypeVariable', '')  # e.g. "TYPE"

    # An analog operator in a condition: a file:line error now, not g++ later.
    # An undeclared node, branch or assignment target: the same.
    from vae.ginac_emitter import check_conditions as _check_conditions
    from vae.ginac_emitter import check_declarations as _check_declarations
    _check_conditions(mod)
    _check_declarations(mod)

    ports = [(p.name, p.direction.name) for p in mod.ports]
    # Node collapse: drop internal nodes shorted via V(a,b)<+0 (e.g. ge->g when
    # RGATEMOD=0, N when NQSMOD=0). The GiNaC math emitter excludes these from
    # its node list, so the device shell MUST too — otherwise the shell keeps
    # extra (gmin-padded, grounded) nodes and stamps e.g. the gate current
    # I(ge,*) into a dead node instead of g, giving a singular DC Jacobian.
    from vae.ginac_emitter import GiNaCEmitter as _CollapseGE
    try:
        _cge = _CollapseGE(mod, param_values={})
        _shorted_internals = _cge._find_shorted_nodes(mod.analog_block)
    except Exception:
        _shorted_internals = set()
    internals = [n for n in mod.internal_nodes if n not in _shorted_internals]
    all_nodes = [p.name for p in mod.ports] + internals
    n_ext = len(ports)
    n_int = len(internals)
    n_total = n_ext + n_int

    # Xyce's device-letter dispatch counts a fixed number of nodes
    # on the netlist card based on the SPICE convention for the
    # device family — M needs 4 (D G S B), Q needs 4 (C B E S),
    # D needs 2 (A C). Extra ports declared in the .va (thermal,
    # body-pickup, substrate) are *optional* — Xyce only consumes
    # them if the netlist supplies them. If we report all declared
    # ports as required, the parser treats the model name as the
    # last "node" and everything after that as an unrecognised
    # parameter, which is the ``Unrecognized parameter X for
    # device M1`` failure mode that dominates the regression
    # suite.
    _GROUP_REQUIRED_TERMINALS = {
        'MOSFET':    4,
        'BJT':       4,
        'Diode':     2,
        'Resistor':  2,
        'Capacitor': 2,
        'Inductor':  2,
    }
    if model_group in _GROUP_REQUIRED_TERMINALS:
        conv = _GROUP_REQUIRED_TERMINALS[model_group]
        # If the .va declares FEWER ports than the SPICE-letter
        # convention (e.g. MVS 2.0 declares only ``d, g, s`` for a
        # MOSFET, no body), we still tell Xyce ``numNodes() = conv``
        # so the netlist's standard M / Q / D card parses correctly.
        # The extra ports beyond what the .va declared are virtual:
        # registerLIDs will accept their LIDs but no contribution
        # references them, equivalent to the SPICE 3-terminal-MOSFET
        # convention of tying body to source.
        num_nodes_required = conv
        num_nodes_optional = max(0, n_ext - conv)
        if n_ext < conv:
            # Synthesize virtual port names so the rest of the
            # codegen (li_<n>, registerLIDs) has something to bind
            # the extra extLIDVec entries to. Conventional spelling
            # per group keeps stamps and printouts readable.
            _VIRTUAL_PORTS = {
                'MOSFET': ['_b', '_t'],
                'BJT':    ['_s', '_t'],
                'Diode':  ['_t'],
            }
            extras = _VIRTUAL_PORTS.get(model_group, [])
            existing = {p[0].lower() for p in ports}
            i = 0
            while n_ext < conv and i < len(extras):
                vn = 'vp' + extras[i]
                while vn.lower() in existing:
                    vn = vn + '_'
                ports = ports + [(vn, 'INOUT')]
                existing.add(vn.lower())
                n_ext += 1
                i += 1
            while n_ext < conv:
                vn = f'vp_extra{n_ext}'
                ports = ports + [(vn, 'INOUT')]
                n_ext += 1
            # Recompute all_nodes / n_total — downstream code uses
            # both heavily for stamp indices and for the registerLIDs
            # loop. Keeping ports as the prefix of all_nodes is the
            # invariant the rest of the generator expects.
            all_nodes = [p[0] for p in ports] + internals
            n_total = n_ext + n_int
    else:
        # Y-devices and unknown groups: every declared port is required,
        # zero optional — matches what Xyce's generic Y handler expects.
        num_nodes_required = n_ext
        num_nodes_optional = 0

    # Detect "simple R / L / C" pattern. When xyceModelGroup is one of
    # the linear-element families AND the module exposes a standard
    # instance parameter naming the resistance / capacitance /
    # inductance, we don't need a vae_eval .so at all — the wrapper
    # stamps a linear conductance / capacitance / inductance directly,
    # the same way Xyce's built-in R / L / C devices do. The detection
    # is intentionally narrow: only the SPICE-primitive case. Anything
    # more complex (BSIM-CMG, r3_cmc with sheet-resistance + geometry
    # computation, etc.) takes the regular vae_eval path.
    _SIMPLE_RLC_PARAM_NAMES = {
        'Resistor':  ('resistance', 'r'),
        'Capacitor': ('capacitance', 'c'),
        'Inductor':  ('inductance', 'l'),
    }
    # Built-in model-group Traits to join, per xyceModelGroup. Header +
    # qualified Traits name for the DeviceTraits third template arg.
    _GROUP_TRAITS = {
        'RESISTOR':  ('N_DEV_Resistor.h',  'Resistor::Traits'),
        'CAPACITOR': ('N_DEV_Capacitor.h', 'Capacitor::Traits'),
        'INDUCTOR':  ('N_DEV_Inductor.h',  'Inductor::Traits'),
        'MOSFET':    ('N_DEV_MOSFET1.h',   'MOSFET1::Traits'),
        'DIODE':     ('N_DEV_Diode.h',     'Diode::Traits'),
        'BJT':       ('N_DEV_BJT.h',       'BJT::Traits'),
    }
    # simple_rlc is either:
    #   ('R'|'C'|'L', cxx_param_name)              — direct value param
    #   ('R_SHEET', rsh_cxx, l_cxx, w_cxx)         — compact-resistor pattern
    # or None.
    #
    # The point: any module tagged xyceModelGroup="Resistor" (etc.) is
    # treated as a behavioral linear element regardless of how the
    # underlying compact model actually computes its physics. Chip-
    # level simulation just needs "this device acts like a resistor",
    # not "this device matches r3_cmc's body-charge / self-heating
    # equations to 4 sig figs". Compact-model VAE path is preserved
    # as a fallback (``if (!vae_eval_)`` guard inside the stamp), but
    # for a Resistor/Capacitor/Inductor group we'd rather get a clean
    # convergent linear stamp than a singular Jacobian from a half-
    # wired-up compact-model evaluator.
    _CXX_RESERVED_SET = {
        'short', 'long', 'int', 'char', 'float', 'double', 'void', 'bool',
        'auto', 'const', 'static', 'extern', 'register', 'volatile',
        'class', 'struct', 'union', 'enum', 'template', 'typename',
        'new', 'delete', 'this', 'return', 'if', 'else', 'for', 'while',
        'do', 'switch', 'case', 'default', 'break', 'continue', 'goto',
        'try', 'catch', 'throw', 'namespace', 'using', 'public', 'private',
        'protected', 'virtual', 'friend', 'inline', 'operator', 'sizeof',
        'typeid', 'true', 'false', 'nullptr',
    }
    def _cxx_mangle(nm):
        return f'pyms_{nm}' if nm.lower() in _CXX_RESERVED_SET else nm
    simple_rlc = None
    if model_group in _SIMPLE_RLC_PARAM_NAMES and n_ext >= 2:
        wanted = _SIMPLE_RLC_PARAM_NAMES[model_group]
        canonical = wanted[0]
        # Pass 1: direct value parameter (resistance/capacitance/inductance).
        # Instance params take precedence; a plain ``parameter real R``
        # with no type="instance" attr is a MODEL param (set from the
        # .MODEL card, e.g. ``.MODEL m r level=91 R=2k``) — reference
        # it through the model_ member so the stamp sees the card value.
        inst_params = [p for p in mod.params if getattr(p, 'is_instance', False)]
        ordered = sorted(inst_params,
                         key=lambda p: 0 if p.name.lower() == canonical else 1)
        for p in ordered:
            if p.name.lower() in wanted:
                # kind = SPICE letter from the param table (R/C/L), NOT
                # model_group[0] -- "Inductor"[0] is 'I', which matched no
                # stamp branch, leaving the inductor behaviorally unstamped.
                simple_rlc = (wanted[1].upper(), _cxx_mangle(p.name))
                break
        if simple_rlc is None:
            model_params = [p for p in mod.params
                            if not getattr(p, 'is_instance', False)]
            ordered = sorted(model_params,
                             key=lambda p: 0 if p.name.lower() == canonical else 1)
            for p in ordered:
                if p.name.lower() in wanted:
                    simple_rlc = (wanted[1].upper(), 'model_.' + _cxx_mangle(p.name))
                    break
        # Pass 2: compact-resistor sheet-resistance pattern (rsh × l / w).
        # Only applies to Resistor; nothing analogous exists for C/L in
        # the IHP set.
        if simple_rlc is None and model_group == 'Resistor':
            all_param_names = {p.name.lower(): p for p in mod.params}
            inst_param_names = {p.name.lower(): p for p in inst_params}
            # rsh may be either Instance or Model. l, w are instance.
            rsh_p = all_param_names.get('rsh')
            l_p   = inst_param_names.get('l')
            w_p   = inst_param_names.get('w')
            if rsh_p and l_p and w_p:
                # Instance members are bare; Model members are
                # accessed through the ``model_`` reference held by
                # each Instance.
                rsh_ref = (_cxx_mangle(rsh_p.name)
                           if getattr(rsh_p, 'is_instance', False)
                           else 'model_.' + _cxx_mangle(rsh_p.name))
                simple_rlc = ('R_SHEET',
                              rsh_ref,
                              _cxx_mangle(l_p.name),
                              _cxx_mangle(w_p.name))

    # QSPICE-dialect loss shorthands on a C/L value device: RSER in series
    # and RPAR across the element. The behavioral stamp must honor these or
    # a lossy cap/inductor silently simulates as ideal (RSER=0, RPAR=inf) --
    # the simple-R/L/C path skips vae_eval, so loss branches in the .va are
    # never evaluated. RSER sits between the + terminal and the reactive
    # element, so it needs the module's internal node (qspice_cap/ind: `mid`).
    rlc_loss = None
    if simple_rlc and simple_rlc[0] in ('C', 'L'):
        def _loss_ref(nm):
            for p in mod.params:
                if p.name.lower() == nm:
                    return (_cxx_mangle(p.name) if getattr(p, 'is_instance', False)
                            else 'model_.' + _cxx_mangle(p.name))
            return None
        rser_ref = _loss_ref('rser')
        rpar_ref = _loss_ref('rpar')
        if rser_ref or rpar_ref:
            has_mid = bool(rser_ref) and n_int >= 1
            rlc_loss = {
                'rser':     rser_ref,
                'rpar':     rpar_ref,
                'mid_name': internals[0] if has_mid else None,
                'mid_idx':  n_ext        if has_mid else None,
            }

    # Split scalar vs array params. Scalars go into Model/Instance class
    # members via addPar; arrays are emitted once as namespace-scope static
    # const double[].
    #
    # Mangle parameter names that collide with C++ reserved words. The
    # netlist sees the original name (via addPar's first arg, which we
    # uppercase); only the C++ member changes. This is how IHP's
    # ``parameter real short = 0.0;`` survives codegen.
    _CXX_RESERVED = {
        'short', 'long', 'int', 'char', 'float', 'double', 'void', 'bool',
        'auto', 'const', 'static', 'extern', 'register', 'volatile',
        'class', 'struct', 'union', 'enum', 'template', 'typename',
        'new', 'delete', 'this', 'return', 'if', 'else', 'for', 'while',
        'do', 'switch', 'case', 'default', 'break', 'continue', 'goto',
        'try', 'catch', 'throw', 'namespace', 'using', 'public', 'private',
        'protected', 'virtual', 'friend', 'inline', 'operator', 'sizeof',
        'typeid', 'true', 'false', 'nullptr',
    }
    def _cxx_safe(nm):
        return f'pyms_{nm}' if nm.lower() in _CXX_RESERVED else nm

    params = []            # (name, default, is_instance, cxx_name) — scalars only
    array_params = []      # (name, array_size, elements)
    for p in mod.params:
        if getattr(p, 'array_size', None):
            array_params.append((p.name, p.array_size, p.elements or []))
        else:
            params.append((p.name, p.default,
                           getattr(p, 'is_instance', False),
                           _cxx_safe(p.name)))

    # Xyce parameter names are case-insensitive: the shell registers each
    # one in capitals (addPar below) and the netlist parser folds a card's
    # names the same way, which is how build_vae_so.py maps them back onto
    # the module's case-sensitive names. Two parameters that differ only in
    # case (EKV 2.6: TEMP and Temp) would register one Xyce name twice (Xyce
    # aborts with "Parameter TEMP already added to class ...") and no card
    # could address them: refuse, naming them.
    _by_upper = {}
    for pname, _pd, _ii, _cxx in params:
        _by_upper.setdefault(pname.upper(), []).append(pname)
    for _names in _by_upper.values():
        if len(_names) > 1:
            raise ValueError(
                f"{_param_loc(mod, _names[0]) or _decl_line(va_path, _names[0])}: "
                f"module {mod.name}: parameters "
                + " and ".join(repr(n) for n in _names)
                + " differ only in case; Xyce parameter names are case-insensitive, "
                "so a .model card or instance line cannot address them")

    # Every parameter default, evaluated with Verilog-A semantics (va_const):
    # the addPar defaults below, and the check that every form is one PyMS can
    # evaluate -- an unevaluable default is an error here, at .hdl time, naming
    # the file, the line and the construct, never a device with that parameter
    # 0. $simparam reads Xyce's option defaults here; the math library
    # evaluates every default per instance (build_vae_so.py bakes only the
    # given values), and the shell re-evaluates the dependent ones in
    # processParams() for its own members (the .op listing, the R/L/C path).
    try:
        # (a default that reads $temperature / $vt: the shell's reading is its
        # value at 27 C; the math library computes it at the instance's
        # temperature, at run time -- 1.1.6, BUGS.md A17)
        _defaults = evaluate_param_defaults(
            mod, {}, dict(XYCE_SIMPARAM_DEFAULTS, **{TEMPERATURE_KEY: 300.15}))
        _thermal = set()
        evaluate_param_defaults(mod, {}, XYCE_SIMPARAM_DEFAULTS, thermal=_thermal)
        _dep_order = dependency_order(mod)
    except ConstExprError as e:
        raise DeviceGenError(_locate_param_error(va_path, str(e), mod))
    _kinds = {p.name: param_kind(p) for p in mod.params}
    # String parameters (LRM 3.4.6) are std::string members registered with
    # a string default, so a card or instance line can give one
    # (mode=slow); the shell passes the text to the math library, which
    # resolves the string compares at elaboration (ginac_emitter._cond_ns).
    _str_params = {pn for pn, k in _kinds.items() if k == 'str'}

    def _ctype(pname):
        return 'std::string' if pname in _str_params else 'double'

    def _cxx_str(text):
        return '"' + text.replace('\\', '\\\\').replace('"', '\\"') + '"'
    # Instance parameters with xyceAlsoModel="yes" (the Xyce/ADMS convention):
    # registered on the model too, and an instance that does not give one
    # takes the card's value (Instance::processParams). Without it a card
    # value of such a parameter was "No model parameter W found" (PSP103,
    # BSIM-CMG, BSIM-SOI, BSIM6, Mextram, VBIC, EKV carry 270 of them).
    _also_model = {p.name for p in mod.params
                   if getattr(p, 'is_instance', False) and getattr(p, 'also_model', False)}
    _cxx_of = {pn: cx for pn, _d, _ii, cx in params}
    _is_inst_of = {pn: ii for pn, _d, ii, _cx in params}
    # A module parameter named TEMP (EKV 2.6, fbh_hbt: "device temperature in
    # C") is the module's own, as the LRM has it -- the circuit temperature
    # reaches a module through $temperature only.  Xyce's device manager
    # writes the circuit temperature (C) into the TEMP of every device at a
    # temperature change (DeviceMgr::updateTemperature: .step temp), which
    # would change it -- and bake it, a library per temperature.  The shell
    # undoes that write in processParams (1.1.6): a TEMP equal to the circuit
    # temperature that the module's value is not goes back to the module's
    # value and given-ness (a .step of the module's own TEMP to exactly the
    # circuit temperature is the one value this cannot tell; it keeps the
    # previous one).  VACASK never writes a module parameter either.
    _temp_par_inst = next((pn for pn, _d, ii, _cx in params if ii and pn.upper() == 'TEMP'), None)
    _temp_par_model = next((pn for pn, _d, ii, _cx in params
                            if (not ii or pn in _also_model) and pn.upper() == 'TEMP'), None)

    # The instance's own temperature, as Xyce's built-in devices have it
    # (N_DEV_Diode.C, N_DEV_Resistor.C): an instance parameter TEMP (C on the
    # netlist, kelvin inside: Xyce's conversion) is the device temperature,
    # else the circuit temperature plus DTEMP (C = K difference).  Registered
    # by the shell when the module has no parameter of that name itself
    # (BSIM-CMG has its own DTEMP, PSP its DTA, HiCUM its dt: those models
    # apply them to $temperature themselves).  A netlist TEMP stays the
    # instance's at a .step temp (the device manager's write is undone as
    # for a module TEMP, above) and DTEMP stays applied; Xyce's built-in
    # diode takes the circuit temperature there and drops DTEMP.
    _all_upper = {pn.upper() for pn, _d, _ii, _cx in params}
    # ``aliasparam ALIAS = TARGET`` (1.1.8, BUGS.md B26): the alias is another
    # netlist name of the target's value, on the card and the instance line
    # alike (as openvaf-r has it).  Registered with addPar on the target's
    # member; the constructors hand a given alias over to the target's name
    # (pyms_unalias_), so given("TARGET"), the math library's key and
    # $param_given see the target.  An alias of an array parameter is not.
    _aliases = [(a, t) for a, t in getattr(mod, 'aliases', {}).items() if t in _cxx_of]
    for a, t in _aliases:
        if a.upper() == 'XYCEADMSINSTTEMP':
            raise DeviceGenError(f"{va_path}: aliasparam {a}: the name is the device shell's own")
    _all_upper |= {a.upper() for a, _t in _aliases}
    _shell_temp = 'TEMP' not in _all_upper
    _shell_dtemp = 'DTEMP' not in _all_upper

    def _tp_restore(member, val='pyms_tp_val_', giv='pyms_tp_given_', kelvin=False):
        ct = ('getDeviceOptions().temp.getImmutableValue<double>()' if kelvin
              else 'getDeviceOptions().temp.getImmutableValue<double>() - CONSTCtoK')
        return [f'  {{ const double _ct = {ct};',
                f'    if (std::fabs({member} - _ct) <= 1e-9 * (1.0 + std::fabs(_ct))) {{',
                f'      {member} = {val};  // the device manager\'s write (or the same value): the netlist\'s',
                f'      if (!{giv}) pyms_ungive_(*this, "TEMP");',
                f'    }} else if ({member} != {val}) {{ {val} = {member}; {giv} = given("TEMP"); }}',
                '  }']

    # Branch-current unknowns (MNA). The target branch of every voltage
    # contribution and every branch whose current is probed (I(a,b), I(br))
    # get an internal solution variable each: a voltage contribution is the
    # equation V_a - V_b - e = 0 with the branch current in the two KCL rows,
    # a probed current-contribution branch the equation i - sum(e) = 0, and
    # a probed branch without contribution a short carrying the unknown. The
    # math library builds the SAME list on the same module
    # (GiNaCEmitter.collect_current_unknowns) and exports, for each unknown
    # alive in its elaboration, the row label and the row of the branch's
    # equation; updateIntermediateVars matches them by label. A port
    # current probe I(<p>) is refused here, before any C++ is compiled.
    # The simple R/L/C stamp never calls the library, so it has none. (After
    # the defaults' check: the emitter evaluates the defaults too, without
    # the file and line that check reports.)
    cur_records = _CollapseGE(mod, param_values={}).collect_current_unknowns(all_nodes)
    if simple_rlc:
        cur_records = []
    n_cur = len(cur_records)
    n_unk = n_total + n_cur
    if n_total > VAE_STATE_CAP or n_cur > VAE_STATE_CAP:
        raise DeviceGenError(
            f'{va_path or mod.name}: module {mod.name} has {n_total} nodes and {n_cur} '
            f'branch-current unknowns; the PyMS VaeState holds at most {VAE_STATE_CAP} of each')
    _node_idx = {n: i for i, n in enumerate(all_nodes)}
    # Per unknown: node indices of its branch (-1 = ground), the row label
    # shared with the math library, the Xyce name of the internal variable.
    cur_info = []
    for rec in cur_records:
        cur_info.append((_node_idx.get(rec['a'], -1), _node_idx.get(rec['b'], -1),
                         rec['label'],
                         'ibr_' + re.sub(r'\W+', '_', rec['label'][3:-1])))

    def _cxx_default(p, tree, in_instance):
        """C++ that re-evaluates p's default where the shell holds the other
        parameters as double members (model_.x from an Instance)."""
        def _ref(nm):
            if in_instance and not _is_inst_of.get(nm, False):
                return 'model_.' + _cxx_of[nm]
            return _cxx_of[nm]
        try:
            code, kind = to_cxx(tree, _kinds, _ref)
        except ConstExprError as e:
            raise DeviceGenError(_locate_param_error(va_path, f"parameter {p.name}: {e}", mod))
        if _kinds[p.name] == 'int' and kind != 'int':
            code = f'_pyms_rnd({code})'
        return code

    # Find contributions
    contribs = []
    def find_contribs(node):
        if node is None:
            return
        if node.kind == NodeKind.CONTRIB:
            contribs.append((node.contrib_kind, node.branch, node.expr))
        for c in (node.children or []):
            find_contribs(c)
        if node.else_body:
            find_contribs(node.else_body)
    find_contribs(mod.analog_block)

    # (1.1.8, BUGS.md A19) PyMS has no noise: the emitter replaces the noise
    # sources by 0, so a .noise analysis sees none of the module's noise.  Say
    # so (Xyce shows a generator note as a ".HDL:" warning on every run).
    _noise = noise_sources(mod)
    if _noise:
        print(f"xyce_device_gen: note: module {mod.name} of {va_path}: "
              + ", ".join(f"{fn} (line {ln})" if ln else fn for fn, ln in _noise)
              + ": PyMS does not implement noise; these sources contribute nothing (a .noise analysis "
              "shows none of this module's noise)")

    # ================================================================
    # Generate header
    # ================================================================
    h = []
    h.append(f'// PyMS-generated Xyce device: {name}')
    h.append(f'// From Verilog-A module "{name}"')
    h.append(f'#ifndef Xyce_PYMS_{NAME}_h')
    h.append(f'#define Xyce_PYMS_{NAME}_h')
    h.append('')
    h.append('#include <N_DEV_Configuration.h>')
    h.append('#include <N_DEV_DeviceBlock.h>')
    h.append('#include <N_DEV_DeviceInstance.h>')
    h.append('#include <N_DEV_DeviceModel.h>')
    h.append('#include <N_DEV_DeviceMaster.h>')
    h.append('#include <N_UTL_BreakPoint.h>')
    h.append('#include <dlfcn.h>')
    # Join the built-in model group rather than becoming our own.
    # DeviceTraits<M, I> (two-arg) makes the generated device its own
    # model group; Configuration::addDevice then last-one-sticks
    # overrides modelTypeNameModelGroupMap_["r"] (etc.), stealing the
    # device letter from the built-in — after which every PLAIN
    # ``R1 a b 1k`` routes to the .va device and dies with "instance
    # must reference a model". Built-in levels avoid this by passing
    # the group's Traits as the third template arg (cf. Resistor3,
    # MOSFET_B4); do the same for every group we register a level on.
    group_traits = None
    if level_number:
        group_traits = _GROUP_TRAITS.get(model_group.upper())
    if group_traits:
        h.append(f'#include <{group_traits[0]}>')
    h.append('')
    # Xyce_config.h (pulled in transitively above) ``#define``s
    # build-version macros that collide with parameter names from
    # real compact-model .va files — e.g. BSIM-SOI 4.7 declares
    # ``parameter real VERSION = 4.7``, but Xyce_config.h has
    # ``#define VERSION "..."``, so the macro rewrites every
    # ``Model::VERSION`` member as a string literal. Undef the known
    # offenders before any class declaration uses them.
    for offender in ('VERSION', 'PACKAGE_VERSION', 'PACKAGE_NAME',
                     'PACKAGE_STRING', 'MAJOR', 'MINOR'):
        h.append(f'#ifdef {offender}\n#undef {offender}\n#endif')
    h.append('')
    h.append(f'namespace Xyce {{ namespace Device {{ namespace PYMS_{NAME} {{')
    h.append('')

    # Node IDs
    for i, n in enumerate(all_nodes):
        h.append(f'static const int nodeID_{n} = {i};')
    h.append(f'static const int numNodes = {n_total};')
    h.append(f'static const int numExtNodes = {n_ext};')
    h.append(f'static const int numIntNodes = {n_int};')
    h.append(f'static const int numCurrents = {n_cur};  // branch-current unknowns')
    h.append(f'static const int numUnknowns = {n_unk};  // nodes, then branch currents')
    h.append('')

    # VAE function pointer types. V: node voltages; I: the branch-current
    # unknowns in the math library's order (vae_current_label).
    h.append(VAE_STATE_DECL)
    h.append('typedef void (*VaeEvalFn)(VaeState*, double*, double*);')
    h.append('typedef void (*VaeSetCbFn)(double(*)(const char*));')
    h.append('')

    # Traits
    h.append('class Model;')
    h.append('class Instance;')
    h.append('')
    if group_traits:
        h.append(f'struct Traits : public DeviceTraits<Model, Instance, {group_traits[1]}> {{')
    else:
        h.append('struct Traits : public DeviceTraits<Model, Instance> {')
    h.append(f'  static const char *name() {{ return "{name}"; }}')
    h.append(f'  static const char *deviceTypeName() {{ return "{NAME}"; }}')
    h.append(f'  static int numNodes() {{ return {num_nodes_required}; }}')
    h.append(f'  static int numOptionalNodes() {{ return {num_nodes_optional}; }}')
    h.append(f'  static bool isLinearDevice() {{ return false; }}')
    h.append(f'  static bool modelRequired() {{ return true; }}')
    h.append('  static const char **nodeNames();')
    h.append('  static void loadInstanceParameters(ParametricData<Instance> &p);')
    h.append('  static void loadModelParameters(ParametricData<Model> &p);')
    h.append('  static Device *factory(const Configuration &, const FactoryBlock &);')
    h.append('};')
    h.append('')

    # Instance class
    h.append('class Instance : public DeviceInstance {')
    h.append('  friend class Model;')
    h.append('  friend class Traits;')
    h.append('  friend class DeviceMaster<Traits>;')
    h.append('public:')
    h.append('  Instance(const Configuration &config, const InstanceBlock &ib,')
    h.append('           Model &model, const FactoryBlock &fb);')
    h.append('  ~Instance();')
    h.append('  void registerLIDs(const std::vector<int> &intLIDVec,')
    h.append('                    const std::vector<int> &extLIDVec);')
    h.append('  void registerStateLIDs(const std::vector<int> &staLIDVec);')
    h.append('  void registerStoreLIDs(const std::vector<int> &stoLIDVec);')
    h.append('  std::map<int,std::string> &getIntNameMap();')
    h.append('  std::map<int,std::string> &getStoreNameMap();')
    h.append('  const std::vector<std::string> &getDepSolnVars();')
    h.append('  const std::vector< std::vector<int> > &jacobianStamp() const;')
    h.append('  void registerJacLIDs(const std::vector< std::vector<int> > &jacLIDVec);')
    h.append('  bool processParams();')
    h.append('  double getCbParam(const char* nm) const;  // runtime-callback param fetch')
    # The device temperature (1.1.6, BUGS.md A17): every evaluation reads it
    # through this instance's temperature source, a pointer.  By default it
    # points at pyms_tdev_, the instance's Xyce temperature (kelvin: the
    # hidden instance parameter XYCEADMSINSTTEMP, which Xyce's device
    # manager sets at every temperature change -- .step temp -- as it does
    # for its ADMS devices; at construction the circuit temperature, TEMP /
    # .options device temp; or the instance's own TEMP / circuit + DTEMP,
    # as Xyce's built-in devices have them).  A thermal network or a self-heating state
    # re-points it (setTemperatureSource) at a value it updates during the
    # run: nothing is regenerated, no library rebuilt.
    h.append('  void setTemperatureSource(const double* src) { pyms_tsrc_ = src ? src : &pyms_tdev_; }')
    h.append('  const double* getTemperatureSource() const { return pyms_tsrc_; }')
    h.append('  double getDeviceTemperature() const { return *pyms_tsrc_; }')
    h.append('  bool updateIntermediateVars();')
    h.append('  bool updatePrimaryState();')
    h.append('  bool loadDAEFVector();')
    h.append('  bool loadDAEQVector();')
    h.append('  bool loadDAEdFdx();')
    h.append('  bool loadDAEdQdx();')
    h.append('  void loadNodeSymbols(Util::SymbolTable &symbol_table) const;')
    # The held state of the math library (1.1.4): committed at an accepted
    # point (acceptStep) and when an analysis leaves its first-point phase;
    # the cross events of a transient place a breakpoint where they predict
    # the crossing.
    h.append('  void acceptStep();')
    h.append('  bool isConverged();')
    h.append('  bool getInstanceBreakPoints(std::vector<Util::BreakPoint> &breakPointTimes);')
    h.append('')
    h.append('  Model &getModel() { return model_; }')
    h.append('')
    h.append('private:')
    h.append('  Model &model_;')
    # Instance parameters
    for pname, pdefault, is_inst, _cxx in params:
        if is_inst:
            h.append(f'  {_ctype(pname)} {_cxx};')
    h.append('')
    # LID storage
    for n in all_nodes:
        h.append(f'  int li_{n};')
    for k, (_a, _b, label, _nm) in enumerate(cur_info):
        h.append(f'  int li_cur_{k};  // branch current {label}')
    h.append('')
    # Jacobian offsets
    h.append(f'  std::vector<std::vector<int>> jacStamp_;')
    h.append(f'  std::vector<std::vector<int>> jacLIDs_;')
    h.append('')
    # VAE state. F_/Q_ and the Jacobians are indexed by unknown: the nodes
    # first (KCL rows), then the branch currents (their equations).
    h.append('  // VAE compiled model')
    h.append('  double pyms_temp_;          // K: XYCEADMSINSTTEMP, the circuit temperature Xyce gives the instance')
    h.append('  double pyms_tdev_;          // K: the device temperature (TEMP, or the circuit\'s + DTEMP)')
    h.append('  const double* pyms_tsrc_;   // the temperature source the evaluations read (&pyms_tdev_)')
    if _shell_temp:
        h.append('  double pyms_itemp_, pyms_itemp_val_; bool pyms_itemp_given_;  // instance TEMP (K)')
    if _shell_dtemp:
        h.append('  double pyms_dtemp_;         // instance DTEMP (K)')
    if _temp_par_inst:
        h.append('  double pyms_tp_val_; bool pyms_tp_given_;  // the module\'s own TEMP (see processParams)')
    h.append('  void *vae_dl_;')
    h.append('  VaeEvalFn vae_eval_;')
    h.append('  VaeEvalFn vae_jac_;')
    h.append(f'  double prevV_[{n_total}];')
    h.append('  bool hasPrevV_;')
    # The label remaps between this shell and ITS math library (built on the
    # first evaluation). Per instance, not static: the library is keyed on
    # the instance's parameters, and two instances can load libraries whose
    # walks culled different contributions (different rows).
    h.append('  std::vector<int> pyms_branch_remap_, pyms_cur_row_, pyms_cur_socol_, pyms_so_col_map_;')
    h.append('  bool pyms_remap_built_;')
    # the JIT math library: loaded by this instance (not an override), and
    # the key of the parameter values it baked
    h.append('  bool pyms_jit_;')
    h.append('  std::size_t pyms_key_;')
    # held state: committed (read) and pending (written) values; the math
    # library's run-time messages; the cross events (slot, direction, the
    # last two accepted values and times)
    h.append('  std::vector<double> pyms_Hc_, pyms_Hp_;')
    # (the declaration values, vae_held_init: every analysis -- every .step
    # point's -- starts from them; the phase of the last evaluation)
    h.append('  std::vector<double> pyms_H0_;')
    h.append('  int pyms_nheld_, pyms_last_ph_;')
    h.append('  bool pyms_seen_init_, pyms_committed_, pyms_accept_eval_;')
    h.append('  const char* (*pyms_take_msg_)();')
    h.append('  const char* (*pyms_take_err_)();')
    # vae_needs_accept of the loaded library: read with the held state and
    # dropped with the library (a function-static cache keyed on the dlopen
    # handle went stale when a re-dlopen returned the same handle for a
    # library mapped elsewhere: SIGSEGV at a .step of a baked parameter).
    h.append('  int (*pyms_needs_accept_)();')
    # (1.1.8, B25) which held slots the last Newton evaluation wrote a new
    # value to (its branch of the held update), at which iteration, and how
    # often this solve was held back for a branch change (isConverged)
    h.append('  std::vector<char> pyms_cv_pat_; int pyms_cv_it_, pyms_cv_forced_; bool pyms_cv_ok_;')
    h.append('  std::vector<int> pyms_cx_slot_, pyms_cx_pend_, pyms_cx_dir_;')
    h.append('  std::vector<double> pyms_cx_c0_, pyms_cx_c1_, pyms_cx_t0_, pyms_cx_t1_;')
    h.append('  std::vector<int> pyms_cx_np_;  // predictions since the event last fired')
    h.append('  int pyms_cx_n_;')
    h.append('  void pymsPoll_();')
    h.append(f'  double F_[{n_unk}];')
    h.append(f'  double Q_[{n_unk}];')
    h.append(f'  double dFdx_[{n_unk}][{n_unk}];')
    h.append(f'  double dQdx_[{n_unk}][{n_unk}];')
    h.append('};')
    h.append('')

    # Model class
    h.append('class Model : public DeviceModel {')
    h.append('  friend class Instance;')
    h.append('  friend class Traits;')
    h.append('  friend class DeviceMaster<Traits>;')
    h.append('public:')
    h.append('  Model(const Configuration &config, const ModelBlock &mb,')
    h.append('        const FactoryBlock &fb);')
    h.append('  ~Model();')
    h.append('  bool processParams();')
    h.append('  bool processInstanceParams();')
    h.append('  void addInstance(Instance *inst) { instanceContainer.push_back(inst); }')
    h.append('  virtual void forEachInstance(DeviceInstanceOp &op) const;')
    h.append('  virtual std::ostream &printOutInstances(std::ostream &os) const;')
    h.append('')
    h.append('private:')
    h.append('  std::vector<Instance*> instanceContainer;')
    # Model parameters (and the model copies of the xyceAlsoModel ones)
    for pname, pdefault, is_inst, _cxx in params:
        if not is_inst or pname in _also_model:
            h.append(f'  {_ctype(pname)} {_cxx};')
    h.append('  std::string vaFile_;')
    if _temp_par_model:
        h.append('  double pyms_tp_val_; bool pyms_tp_given_;  // the module\'s own TEMP (see processParams)')
    h.append('};')
    h.append('')

    # Master
    h.append(f'void registerDevice(const DeviceCountMap &, const std::set<int> &);')
    h.append('')
    h.append(f'}} }} }}  // namespace PYMS_{NAME}')
    h.append(f'#endif')

    header = '\n'.join(h) + '\n'

    # ================================================================
    # Generate implementation (stub — details in next step)
    # ================================================================
    c = []
    c.append(f'// PyMS-generated Xyce device implementation: {name}')
    c.append(f'#include "N_DEV_PYMS_{NAME}.h"')
    c.append('#include <N_DEV_DeviceOptions.h>')
    c.append('#include <N_DEV_SolverState.h>')
    c.append('#include <N_DEV_ExternData.h>')
    c.append('#include <N_DEV_MatrixLoadData.h>')
    c.append('#include <N_DEV_Message.h>')
    c.append('#include <N_DEV_Const.h>')
    c.append('#include <N_LAS_Vector.h>')
    c.append('#include <N_LAS_Matrix.h>')
    c.append('#include <N_PDS_ParMap.h>')
    c.append('#include <cstring>')
    c.append('#include <cmath>')
    # See matching block in the .h: Xyce_config.h pulls in macro
    # definitions that collide with .va parameter names.
    for offender in ('VERSION', 'PACKAGE_VERSION', 'PACKAGE_NAME',
                     'PACKAGE_STRING', 'MAJOR', 'MINOR'):
        c.append(f'#ifdef {offender}\n#undef {offender}\n#endif')
    c.append('#include <cstdlib>')
    c.append('#include <cstdio>')
    c.append('#include <cctype>')
    c.append('#include <functional>')
    if _aliases:
        c.append('#include <algorithm>')
    c.append('#include <fstream>')
    c.append('#include <sstream>')
    c.append('#include <sys/stat.h>')
    c.append('#include <unistd.h>')
    c.append('')
    c.append(f'namespace Xyce {{ namespace Device {{ namespace PYMS_{NAME} {{')
    c.append('')
    c.append(IPOW_CXX)
    if _temp_par_inst or _temp_par_model or _shell_temp:
        c.append('// a parameter given by the device manager, not the netlist: not given (1.1.6, TEMP)')
        c.append('static void pyms_ungive_(DeviceEntity &e, const char *nm) {')
        c.append('  ParameterMap::const_iterator it = e.getParameterMap().find(nm);')
        c.append('  if (it == e.getParameterMap().end()) return;')
        c.append('  const Descriptor &d = *(*it).second;')
        c.append('  if (d.hasGivenMember()) d.setGiven(e, false);')
        c.append('  Xyce::Device::setValueGiven(e, d.getSerialNumber(), false);')
        c.append('}')

    if _aliases:
        # (1.1.8, B26) the netlist's parameters with every given alias renamed
        # to its target, and the target's default entry dropped when an alias
        # gives it: setParams applies the entries in order and the default's
        # entry (given = false) would undo the alias's value and given flag.
        # Both names given: an error.
        c.append('static const char *const pyms_alias_tab_[] = {'
                 + ', '.join('"%s", "%s"' % (a.upper(), t.upper()) for a, t in _aliases) + ', nullptr};')
        c.append('static std::vector<Param> pyms_unalias_(const std::vector<Param> &in) {')
        c.append('  std::vector<Param> out; std::vector<std::string> tg;')
        c.append('  for (const Param &q : in) for (int i = 0; pyms_alias_tab_[i]; i += 2)')
        c.append('    if (q.given() && q.tag() == pyms_alias_tab_[i]) {')
        c.append('      tg.push_back(pyms_alias_tab_[i + 1]);')
        c.append('      for (const Param &o : in) if (o.given() && o.tag() == pyms_alias_tab_[i + 1])')
        c.append('        Report::UserError0() << "PyMS: parameter " << pyms_alias_tab_[i + 1] << " given twice, as itself and as its aliasparam " << pyms_alias_tab_[i];')
        c.append('    }')
        c.append('  for (const Param &q : in) {')
        c.append('    const char *t = nullptr;')
        c.append('    for (int i = 0; pyms_alias_tab_[i]; i += 2) if (q.tag() == pyms_alias_tab_[i]) t = pyms_alias_tab_[i + 1];')
        c.append('    if (t) { if (q.given()) { Param r(q); r.setTag(t); out.push_back(r); } continue; }')
        c.append('    if (!q.given() && std::find(tg.begin(), tg.end(), q.tag()) != tg.end()) continue;')
        c.append('    out.push_back(q);')
        c.append('  }')
        c.append('  return out;')
        c.append('}')
        # a later setParam of an alias (.step Y1:ALIAS): the target counts as given
        c.append('static void pyms_alias_given_(DeviceEntity &e) {')
        c.append('  const ParameterMap &pm = e.getParameterMap();')
        c.append('  for (int i = 0; pyms_alias_tab_[i]; i += 2) {')
        c.append('    ParameterMap::const_iterator a = pm.find(pyms_alias_tab_[i]), t = pm.find(pyms_alias_tab_[i + 1]);')
        c.append('    if (a != pm.end() && t != pm.end() && Xyce::Device::wasValueGiven(e, (*a).second->getSerialNumber()))')
        c.append('      Xyce::Device::setValueGiven(e, (*t).second->getSerialNumber(), true);')
        c.append('  }')
        c.append('}')

    # Array-param tables — emitted once as namespace-scope static const double[].
    # These are compile-time constants baked in from the Verilog-A parameter
    # aggregate initialiser; cheaper and shared across all instances of the
    # device than per-Model heap members.
    for aname, asize, elements in array_params:
        if not elements or len(elements) != asize:
            # Fallback — zero-initialise when the aggregate is missing or wrong size.
            init = ', '.join(['0.0'] * asize)
        else:
            init = ', '.join(elements)
        c.append(f'static const double {aname}[{asize}] = {{ {init} }};')
    if array_params:
        c.append('')

    # Node names
    node_names_str = ', '.join(f'"{n}"' for n in [p.name for p in mod.ports])
    c.append(f'static const char *nodeNameArray[] = {{ {node_names_str} }};')
    c.append(f'const char **Traits::nodeNames() {{ return nodeNameArray; }}')
    c.append('')

    # The addPar default of a parameter: its Verilog-A default evaluated with
    # every parameter at its default (va_const, above). Paramset resolution
    # leaves bindings like ``.rsh = rsh_rsil`` where ``rsh_rsil`` is itself a
    # parameter -- the evaluation follows such chains, and a string parameter
    # registers as 0.0 (its member is a double).
    def _resolve_default(pname):
        v = _defaults.get(pname)
        if pname in _str_params:
            return 'std::string(' + _cxx_str(v if isinstance(v, str) else '') + ')'
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
        return 0.0

    # Parameter registration
    c.append('void Traits::loadInstanceParameters(ParametricData<Instance> &p) {')
    for pname, pdefault, is_inst, _cxx in params:
        if is_inst:
            dval = _resolve_default(pname)
            c.append(f'  p.addPar("{pname.upper()}", {dval}, &Instance::{_cxx})')
            if pname.upper() in ('TEMP', 'TNOM'):
                c.append('    .setAutoConvertTemperature(false)  // the module\'s own value, as the netlist gives it')
            c.append(f'    .setUnit(U_NONE)')
            c.append(f'    .setDescription("{pname}");')
    for a, t in _aliases:
        if _is_inst_of.get(t):
            c.append(f'  p.addPar("{a.upper()}", {_resolve_default(t)}, &Instance::{_cxx_of[t]})')
            if t.upper() in ('TEMP', 'TNOM'):
                c.append('    .setAutoConvertTemperature(false)')
            c.append(f'    .setUnit(U_NONE)')
            c.append(f'    .setDescription("aliasparam of {t}");')
    # the instance's Xyce temperature (kelvin), as Xyce's ADMS devices have it
    if _shell_temp:
        c.append('  p.addPar("TEMP", 0.0, &Instance::pyms_itemp_)')
        c.append('    .setUnit(U_DEGC)')
        c.append('    .setCategory(CAT_TEMP)')
        c.append('    .setDescription("PyMS: the device temperature of this instance (C; default the circuit\'s)");')
    if _shell_dtemp:
        c.append('  p.addPar("DTEMP", 0.0, &Instance::pyms_dtemp_)')
        c.append('    .setUnit(U_DEGC)')
        c.append('    .setCategory(CAT_TEMP)')
        c.append('    .setDescription("PyMS: the device temperature over the circuit\'s (C)");')
    c.append('  p.addPar("XYCEADMSINSTTEMP", 0.0, &Instance::pyms_temp_)')
    c.append('    .setExpressionAccess(NO_DOC)')
    c.append('    .setUnit(U_DEGK)')
    c.append('    .setCategory(CAT_TEMP)')
    c.append('    .setDescription("PyMS: the device temperature Xyce\'s device manager sets (internal)");')
    c.append('}')
    c.append('')

    c.append('void Traits::loadModelParameters(ParametricData<Model> &p) {')
    for pname, pdefault, is_inst, _cxx in params:
        if not is_inst or pname in _also_model:
            dval = _resolve_default(pname)
            c.append(f'  p.addPar("{pname.upper()}", {dval}, &Model::{_cxx})')
            if pname.upper() in ('TEMP', 'TNOM'):
                c.append('    .setAutoConvertTemperature(false)  // the module\'s own value, as the netlist gives it')
            c.append(f'    .setUnit(U_NONE)')
            c.append(f'    .setDescription("{pname}");')
    for a, t in _aliases:
        if not _is_inst_of.get(t) or t in _also_model:
            c.append(f'  p.addPar("{a.upper()}", {_resolve_default(t)}, &Model::{_cxx_of[t]})')
            if t.upper() in ('TEMP', 'TNOM'):
                c.append('    .setAutoConvertTemperature(false)')
            c.append(f'    .setUnit(U_NONE)')
            c.append(f'    .setDescription("aliasparam of {t}");')
    c.append('}')
    c.append('')

    # Instance implementation
    c.append('Instance::Instance(const Configuration &config,')
    c.append('                   const InstanceBlock &ib,')
    c.append('                   Model &model, const FactoryBlock &fb)')
    c.append('  : DeviceInstance(ib, config.getInstanceParameters(), fb),')
    c.append('    model_(model),')
    c.append('    vae_dl_(nullptr), vae_eval_(nullptr), vae_jac_(nullptr),')
    c.append('    hasPrevV_(true)  // start with zero as "previous" for limiting')
    c.append('{')
    c.append(f'  numExtVars = {n_ext};')
    c.append('  pyms_remap_built_ = false; pyms_jit_ = false; pyms_key_ = 0;')
    c.append('  pyms_nheld_ = 0; pyms_last_ph_ = -1; pyms_seen_init_ = false; pyms_committed_ = false; pyms_accept_eval_ = false;')
    c.append('  pyms_take_msg_ = nullptr; pyms_take_err_ = nullptr; pyms_needs_accept_ = nullptr; pyms_cx_n_ = 0;')
    c.append('  pyms_cv_it_ = -1; pyms_cv_forced_ = 0; pyms_cv_ok_ = true;')
    c.append(f'  numIntVars = {n_int + n_cur};  // internal nodes, then branch currents')
    c.append(f'  numStateVars = 0;')
    c.append(f'  numStoreVars = 0;')
    c.append('')
    c.append(f'  jacStamp_.resize(numUnknowns);')
    c.append(f'  for (int i = 0; i < numUnknowns; i++) {{')
    c.append(f'    jacStamp_[i].resize(numUnknowns);')
    c.append(f'    for (int j = 0; j < numUnknowns; j++)')
    c.append(f'      jacStamp_[i][j] = j;')
    c.append(f'  }}')
    c.append('')
    # Pull defaults from the addPar metadata, then apply per-instance
    # overrides from the netlist. Mirrors the standard Xyce device
    # pattern (cf. N_DEV_Resistor3.C). Without setParams(IB.params)
    # the member variables stay at their defaults regardless of what
    # the device card supplies — which is why ``resistance=100``
    # was being ignored.
    c.append('  setDefaultParams();')
    c.append('  setParams(' + ('pyms_unalias_(ib.params)' if _aliases else 'ib.params') + ');')
    # (1.1.8, A23/A24) A parameter given as an expression of a .global_param
    # (or of a .param a .DC/.STEP sweeps, which Xyce makes global) is a
    # dependent parameter: setParams leaves the member at its default and only
    # updateDependentParameters() writes the expression's value -- every
    # built-in device calls it here, before processParams.  Without it the
    # math library was built from the declaration default, and Xyce reprocesses
    # the instance only when the global's value CHANGES, so a .global_param
    # (never changed) and the first sweep point equal to the nominal ran with
    # the default.
    c.append('  updateDependentParameters();')
    c.append('  pyms_tsrc_ = &pyms_tdev_;')
    c.append('  if (!given("XYCEADMSINSTTEMP")) pyms_temp_ = getDeviceOptions().temp.getImmutableValue<double>();')
    if _shell_temp:
        c.append('  pyms_itemp_val_ = pyms_itemp_; pyms_itemp_given_ = given("TEMP");')
    if _temp_par_inst:
        c.append(f'  pyms_tp_val_ = {_cxx_of[_temp_par_inst]}; pyms_tp_given_ = given("TEMP");')
    c.append('')
    c.append('  processParams();')
    c.append('}')
    c.append('')

    c.append('Instance::~Instance() {')
    c.append('  if (vae_dl_) dlclose(vae_dl_);')
    c.append('}')
    c.append('')

    _va_esc = (va_path or '').replace('\\', '\\\\').replace('"', '\\"')
    # --- Runtime-callback param support ---------------------------------
    # A param kept SYMBOLIC in the vae .so (callback param, e.g. PSP103
    # DELVTO) is fetched per-eval via pyms_param_cb(name) instead of a baked
    # literal, so Xyce .SAMPLING/AGAUSS on that instance param reaches the
    # math live through ONE .so (no per-sample rebuild). g_pyms_cur names the
    # instance currently being evaluated; the shell sets it before each
    # vae_eval_/vae_jac_ call, so per-device local mismatch resolves to the
    # right instance's sampled value.
    c.append('double Instance::getCbParam(const char* nm) const {')
    for pname, pdefault, is_inst, cxx in params:
        if is_inst and pname not in _str_params:
            c.append(f'  if (!strcmp(nm, "{pname.upper()}")) return {cxx};')
    for pname, pdefault, is_inst, cxx in params:
        if not is_inst and pname not in _str_params:
            c.append(f'  if (!strcmp(nm, "{pname.upper()}")) return model_.{cxx};')
    c.append('  return 0.0;')
    c.append('}')
    c.append('static Instance* g_pyms_cur = nullptr;')
    c.append('static double pyms_param_cb(const char* nm) { return g_pyms_cur ? g_pyms_cur->getCbParam(nm) : 0.0; }')
    c.append('')
    c.append('bool Instance::processParams() {')
    if _aliases:
        c.append('  pyms_alias_given_(*this);')
    if _temp_par_inst:
        c.extend(_tp_restore(_cxx_of[_temp_par_inst]))
    if _shell_temp:
        c.extend(_tp_restore('pyms_itemp_', 'pyms_itemp_val_', 'pyms_itemp_given_', kelvin=True))
    # the device temperature the evaluations read (through pyms_tsrc_)
    c.append('  pyms_tdev_ = ' + ('given("TEMP") ? pyms_itemp_ : ' if _shell_temp else '')
             + 'pyms_temp_' + (' + pyms_dtemp_' if _shell_dtemp else '') + ';')
    # An xyceAlsoModel parameter the instance does not give takes the card's.
    for pname, pdefault, is_inst, cxx in params:
        if is_inst and pname in _also_model:
            c.append(f'  if (!given("{pname.upper()}") && model_.given("{pname.upper()}")) {cxx} = model_.{cxx};')
    # Re-evaluate, per instance, every instance parameter whose default reads
    # another parameter or $simparam and that the netlist did not give (the
    # addPar default is the value with every parameter at its default), in
    # dependency order; the model parameters were done in Model::processParams.
    for p, tree, deps in _dep_order:
        if not _is_inst_of.get(p.name, False) or p.name not in _cxx_of:
            continue
        if any(d not in _cxx_of for d in deps):
            continue
        if p.name in _str_params or any(d in _str_params for d in deps):
            continue  # a string default or one that reads a string: the math library evaluates it
        if p.name in _thermal:
            continue  # a default that reads $temperature / $vt: the math library evaluates it
        c.append(f'  if (!given("{p.name.upper()}")) {_cxx_of[p.name]} = {_cxx_default(p, tree, True)};')
    # (1.1.5) processParams runs again when a .step (or .sampling) changes a
    # parameter: the JIT library baked the old values, so the key of the
    # current values is computed again and a library of other values is
    # replaced by the one of these (built once per distinct value set and
    # cached).  Before, the first library stayed and a .step of a baked
    # parameter changed nothing, silently (all versions to 1.1.4).
    c.append('  if (!vae_eval_ || pyms_jit_) {')
    c.append('    // (1) explicit override: VAE_SO_PATH, or VAE_SO_DIR/<MODEL>.so')
    c.append('    std::string so_path;')
    c.append('    const char *so = getenv("VAE_SO_PATH");')
    c.append('    const char *dir = getenv("VAE_SO_DIR");')
    c.append('    if (so) so_path = so;')
    c.append('    else if (dir) so_path = std::string(dir) + "/" + model_.getName() + ".so";')
    c.append('    if (!vae_eval_ && !so_path.empty()) {')
    c.append('      vae_dl_ = dlopen(so_path.c_str(), RTLD_NOW);')
    c.append('      if (vae_dl_) { vae_eval_ = (VaeEvalFn)dlsym(vae_dl_, "vae_eval");')
    c.append('                     vae_jac_ = (VaeEvalFn)dlsym(vae_dl_, "vae_jacobian"); }')
    c.append('    }')
    c.append('    // (2) JIT per-instance build: bake THIS instance\'s params into a')
    c.append('    // GiNaC vae_eval .so via build_vae_so.py, cached by (va, params) hash.')
    c.append('    if (!vae_eval_ || pyms_jit_) {')
    c.append(f'      const char *_va = "{_va_esc}";')
    c.append('      std::string _p; char _pb[96];')
    # Callback (runtime) params are fetched live via the callback, not baked, so
    # their VALUE must not fork the .so cache — two instances differing only in a
    # callback param (e.g. per-device DELVTO in a Monte-Carlo run) must share ONE
    # .so. Serialize them with a fixed placeholder so the (va,params) hash is
    # independent of the sampled value.
    c.append('      std::string _cbwrap;')
    c.append('      { const char* _cbe = getenv("PYMS_CALLBACK_PARAMS");')
    c.append('        if (_cbe) { std::string _s=_cbe; for(char&ch:_s) ch=toupper((unsigned char)ch); _cbwrap = "," + _s + ","; } }')
    for pname, pdefault, is_inst, cxx in params:
        ref = f'{cxx}' if is_inst else f'model_.{cxx}'
        NM = pname.upper()
        if pname in _str_params:
            # NAME="text": build_vae_so.py reads it back as a string value
            c.append(f'      _p += "{NM}=\\"" + {ref} + "\\"\\n";')
            continue
        c.append(f'      snprintf(_pb,sizeof(_pb),"{NM}=%.12g\\n",(_cbwrap.find(",{NM},")!=std::string::npos)?0.0:(double)({ref})); _p += _pb;')
    # Serialize the set of params the user actually supplied (Xyce given()),
    # so build_vae_so.py can drive $param_given() correctly. Baking every
    # default as "given" mis-selects model branches (e.g. nVtm = NVTM = 0).
    c.append('      std::string _gv;')
    for pname, pdefault, is_inst, cxx in params:
        guard = (f'given("{pname.upper()}")' if is_inst
                 else f'model_.given("{pname.upper()}")')
        if is_inst and pname in _also_model:
            guard = f'given("{pname.upper()}") || model_.given("{pname.upper()}")'
        c.append(f'      if ({guard}) _gv += "{pname.upper()},";')
    c.append('      _p += "__GIVEN__=" + _gv + "\\n";')
    # Values the shell establishes itself, outside the netlist: the nmos/pmos
    # type variable Model::Model sets from the .model keyword. build_vae_so.py
    # bakes these like given values; every other default is the emitter's to
    # evaluate from the .va text (the shell's own reading of a default, 0 for
    # anything but a plain number, must not reach the math).
    if type_variable and any(pn == type_variable and not ii for pn, _pd, ii, _cx in params):
        c.append(f'      _p += "__SET__={type_variable.upper()},\\n";')
    # The simulator's $simparam values for the parameter defaults that read
    # them (tnom in degrees C, as the LRM has it; Xyce keeps it in K). They are
    # part of the cache key through _p, so a run with another gmin builds its
    # own math library.
    c.append('      { char _sb[160]; snprintf(_sb,sizeof(_sb),"__SIMPARAM__gmin=%.17g\\n__SIMPARAM__tnom=%.17g\\n",'
             'getDeviceOptions().gmin,getDeviceOptions().tnom-273.15); _p += _sb; }')
    # Callback (runtime) params for Monte-Carlo etc.: kept symbolic in the .so
    # and fetched via pyms_param_cb so .SAMPLING/AGAUSS reaches them. Appended
    # ONLY when set, so non-MC builds keep an identical hash (cache hit, no
    # rebuild) and behave byte-identically to before.
    c.append('      const char* _cbp = getenv("PYMS_CALLBACK_PARAMS");')
    c.append('      if (_cbp && _cbp[0]) _p += std::string("__CALLBACK__=") + _cbp + "\\n";')
    # Command-line macros (--define / PYMS_DEFINES) reach the math-library
    # build too: the shell passes them to build_vae_so.py as --define options
    # and, when there are any, keys the vae .so cache on them (the same .va
    # with other macros is another module).  Without macros the generated
    # shell is byte-identical to before.
    _defs_sh = ''.join(" --define '" + d.replace("'", "'\\''") + "'"
                       for d in (defines or []))
    if module_arg:
        # a module of a file with several, not the first: the math library
        # is built from this one (build_vae_so.py --module)
        _defs_sh += " --module '" + mod.name + "'"
    _defs_c = _defs_sh.replace('\\', '\\\\').replace('"', '\\"')
    if _defs_sh:
        c.append('      _p += "__DEFINES__=' + _defs_c + '\\n";')
    # The cache key must include the .va CONTENT, not just its path: keying on
    # (path, params) alone silently reuses a stale vae .so after the .va is
    # edited without changing its parameter list (the 2026-09-26 qal_gate
    # "toolchain regression" trap class — same-name/same-params, different
    # model equations). An unreadable .va degrades to the old (path, params)
    # key rather than failing the build.
    c.append('      std::string _vc; { std::ifstream _vf(_va, std::ios::binary); std::ostringstream _vs; _vs << _vf.rdbuf(); _vc = _vs.str(); }')
    # The PyMS version is part of the key: a math library built by another
    # PyMS (another node collapse, another Jacobian) is never reused.
    from vae import __version__ as _pyms_version
    c.append('      std::size_t _key = std::hash<std::string>{}(std::string(_va) + "|" + _p + "|" + _vc + "|' + VAE_STATE_ABI + '|PyMS-' + _pyms_version + '");')
    c.append('      if (!(vae_eval_ && _key == pyms_key_)) {')
    # another value set than the loaded library's: drop it and everything
    # read from it (the row remap, the held state, the cross events)
    c.append('      if (vae_dl_) {')
    c.append('        dlclose(vae_dl_); vae_dl_ = nullptr; vae_eval_ = nullptr; vae_jac_ = nullptr;')
    c.append('        pyms_remap_built_ = false; pyms_take_msg_ = nullptr; pyms_take_err_ = nullptr; pyms_needs_accept_ = nullptr;')
    c.append('        pyms_Hc_.clear(); pyms_Hp_.clear(); pyms_H0_.clear(); pyms_nheld_ = 0; pyms_last_ph_ = -1;')
    c.append('        pyms_cx_slot_.clear(); pyms_cx_pend_.clear(); pyms_cx_dir_.clear(); pyms_cx_n_ = 0;')
    c.append('      }')
    c.append('      const char *_cd = getenv("PYMS_VAE_CACHE");')
    c.append('      std::string _cache = _cd ? _cd : "/tmp/pyms_vae_cache";')
    c.append('      { std::string _mk = "mkdir -p \'" + _cache + "\'"; if(system(_mk.c_str())){} }')
    c.append(f'      char _sob[1024]; snprintf(_sob,sizeof(_sob),"%s/vae_%s_%zx.so",_cache.c_str(),"{mod.name}",_key);')
    c.append('      std::string _sopath = _sob; struct stat _st;')
    c.append('      if (stat(_sopath.c_str(), &_st) != 0) {')
    c.append('        std::string _bld; const char *_pd = getenv("PYMS_DIR");')
    c.append('        if (_pd) { std::string t = std::string(_pd)+"/vae/build_vae_so.py"; if(stat(t.c_str(),&_st)==0) _bld=t; }')
    c.append('        if (_bld.empty()) { const char* cds[]={"/usr/local/share/xyce/PyMS/vae/build_vae_so.py","/usr/local/src/xyce/utils/PyMS/vae/build_vae_so.py"}; for(auto cc:cds){ if(stat(cc,&_st)==0){_bld=cc;break;} } }')
    # build_vae_so.py reads the .params file; a concurrent run of the same
    # identity (a parallel sweep through one PYMS_VAE_CACHE) rewrites it with
    # the same bytes, so write it under a private name and rename() it into
    # place: the reader never sees a truncated file.
    c.append('        std::string _pf = _sopath + ".params"; { std::string _pt = _pf + "." + std::to_string((long)getpid()) + ".tmp"; FILE* _f=fopen(_pt.c_str(),"w"); if(_f){ fwrite(_p.data(),1,_p.size(),_f); fclose(_f); if(rename(_pt.c_str(),_pf.c_str())) unlink(_pt.c_str()); } }')
    c.append('        if (!_bld.empty()) { std::string _cmd = "python3 \'" + _bld + "\' \'" + std::string(_va) + "\' \'" + _sopath + "\' \'" + _pf + "\'' + _defs_c + ' 1>&2"; if(system(_cmd.c_str())){} }')
    c.append('      }')
    c.append('      vae_dl_ = dlopen(_sopath.c_str(), RTLD_NOW);')
    c.append('      if (vae_dl_) { vae_eval_ = (VaeEvalFn)dlsym(vae_dl_, "vae_eval");')
    c.append('                     vae_jac_ = (VaeEvalFn)dlsym(vae_dl_, "vae_jacobian"); }')
    c.append('      pyms_jit_ = true; pyms_key_ = _key;')
    # A math library that did not build or load is an error of this device,
    # not a device that silently conducts nothing (the stamp is skipped
    # without vae_eval_/vae_jac_). On every route: the simple R/L/C stamp
    # never calls the library, but a module PyMS cannot compile is not a
    # module whose stamp can be trusted (toys/capacitor.va, sg13g2 r3_cmc
    # ran rc 0 on that stamp while their library failed: BUGS.md CHANGES 1.1.1).
    c.append('      if (!vae_eval_ || !vae_jac_) {')
    c.append('        const char *_de = dlerror();')
    c.append(f'        UserError(*this) << "PyMS: the Verilog-A math library for module {mod.name} did not build or load ("')
    c.append('                         << (_de ? _de : "no vae_eval/vae_jacobian symbol") << "); see the build_vae_so messages above";')
    c.append('      }')
    c.append('      }  // a library of other values')
    c.append('    }')
    c.append('  }')
    # Install the runtime-param callback (idempotent; skipped for legacy .so')
    # files that predate vae_set_param_cb).
    c.append('  if (vae_dl_) {')
    c.append('    VaeSetCbFn _setcb = (VaeSetCbFn)dlsym(vae_dl_, "vae_set_param_cb");')
    c.append('    if (_setcb) _setcb(&pyms_param_cb);')
    # The library's held state (1.1.4; a library without vae_n_held has none),
    # its run-time messages and its cross events.
    c.append('    if (!pyms_take_msg_ && !pyms_take_err_ && pyms_Hc_.empty()) {')
    c.append('      typedef int (*PymsNFn)(); typedef void (*PymsHIFn)(double*); typedef int (*PymsIIFn)(int);')
    c.append('      PymsNFn _nh = (PymsNFn)dlsym(vae_dl_, "vae_n_held");')
    c.append('      pyms_nheld_ = _nh ? _nh() : 0;')
    c.append('      if (pyms_nheld_ > 0) {')
    c.append('        pyms_Hc_.assign(pyms_nheld_, 0.0);')
    c.append('        PymsHIFn _hi = (PymsHIFn)dlsym(vae_dl_, "vae_held_init");')
    c.append('        if (_hi) _hi(pyms_Hc_.data());')
    c.append('        pyms_Hp_ = pyms_Hc_; pyms_H0_ = pyms_Hc_;')
    c.append('        pyms_needs_accept_ = (PymsNFn)dlsym(vae_dl_, "vae_needs_accept");')
    c.append('      }')
    c.append('      pyms_take_msg_ = (const char* (*)())dlsym(vae_dl_, "vae_take_message");')
    c.append('      pyms_take_err_ = (const char* (*)())dlsym(vae_dl_, "vae_take_error");')
    c.append('      PymsNFn _nc = (PymsNFn)dlsym(vae_dl_, "vae_n_cross");')
    c.append('      PymsIIFn _cs = (PymsIIFn)dlsym(vae_dl_, "vae_cross_slot");')
    c.append('      PymsIIFn _cd = (PymsIIFn)dlsym(vae_dl_, "vae_cross_dir");')
    c.append('      PymsIIFn _cp = (PymsIIFn)dlsym(vae_dl_, "vae_cross_pending_slot");')
    c.append('      pyms_cx_n_ = (_nc && _cs && _cd && _cp) ? _nc() : 0;')
    c.append('      for (int k = 0; k < pyms_cx_n_; k++) { pyms_cx_slot_.push_back(_cs(k)); pyms_cx_pend_.push_back(_cp(k)); pyms_cx_dir_.push_back(_cd(k)); }')
    c.append('      pyms_cx_c0_.assign(pyms_cx_n_, 0.0); pyms_cx_c1_.assign(pyms_cx_n_, 0.0);')
    c.append('      pyms_cx_t0_.assign(pyms_cx_n_, -1.0); pyms_cx_t1_.assign(pyms_cx_n_, -1.0);')
    c.append('      pyms_cx_np_.assign(pyms_cx_n_, 0);')
    c.append('    }')
    c.append('  }')
    c.append('  return true;')
    c.append('}')
    c.append('')
    # Run-time messages of the math library (a division by zero, a loop
    # that does not end): a warning per message, an error stops the run.
    c.append('void Instance::pymsPoll_() {')
    c.append('  if (pyms_take_msg_) { while (const char* m = pyms_take_msg_()) UserWarning(*this) << "PyMS: " << m; }')
    c.append('  if (pyms_take_err_) { if (const char* m = pyms_take_err_()) UserFatal(*this) << "PyMS: " << m; }')
    c.append('}')
    c.append('')
    # acceptStep: Xyce accepted a point (a transient step, the transient\'s
    # operating point). A library with held variables or cross events is
    # evaluated once more at the accepted solution, and those values become
    # the held ones (committed); the cross values feed the breakpoint
    # prediction.
    c.append('void Instance::acceptStep() {')
    c.append('  if (pyms_nheld_ <= 0) return;')
    c.append('  typedef int (*PymsNFn)();')
    c.append('  PymsNFn _na = pyms_needs_accept_;')
    # (an event pending at this point fires in this evaluation: the circuit
    # changes here, and the crossing prediction starts afresh from it)
    c.append('  bool _fired = false;')
    c.append('  for (int k = 0; k < pyms_cx_n_; k++) { int ps = pyms_cx_pend_[k]; if (ps >= 0 && ps < pyms_nheld_ && pyms_Hc_[ps] != 0.0) _fired = true; }')
    c.append('  if (_na && _na()) { pyms_accept_eval_ = true; updateIntermediateVars(); pyms_accept_eval_ = false; }')
    c.append('  pyms_Hc_ = pyms_Hp_; pyms_committed_ = true;')
    c.append('  const double t = getSolverState().currTime_;')
    c.append('  for (int k = 0; k < pyms_cx_n_; k++) {')
    c.append('    int s = pyms_cx_slot_[k]; if (s < 0 || s >= pyms_nheld_) continue;')
    c.append('    if (_fired) { pyms_cx_t0_[k] = -1.0; pyms_cx_t1_[k] = t; pyms_cx_c1_[k] = pyms_Hc_[s]; pyms_cx_np_[k] = 0; continue; }')
    c.append('    if (pyms_cx_t1_[k] == t) { pyms_cx_c1_[k] = pyms_Hc_[s]; continue; }')
    c.append('    pyms_cx_c0_[k] = pyms_cx_c1_[k]; pyms_cx_t0_[k] = pyms_cx_t1_[k];')
    c.append('    pyms_cx_c1_[k] = pyms_Hc_[s]; pyms_cx_t1_[k] = t;')
    c.append('  }')
    c.append('}')
    c.append('')
    c.append('bool Instance::isConverged() { return pyms_cv_ok_; }')
    c.append('')
    # Timestep control of the cross events: from the last two accepted
    # values, a breakpoint at the predicted crossing when it falls within the
    # next two steps, so the transient lands on it instead of stepping over.
    c.append('bool Instance::getInstanceBreakPoints(std::vector<Util::BreakPoint> &breakPointTimes) {')
    c.append('  if (pyms_cx_n_ <= 0) return false;')
    c.append('  for (int k = 0; k < pyms_cx_n_; k++) {')
    c.append('    double t0 = pyms_cx_t0_[k], t1 = pyms_cx_t1_[k], c0 = pyms_cx_c0_[k], c1 = pyms_cx_c1_[k];')
    # a pending event fires at the accepted point of a breakpoint just after
    # the point that found the crossing (the change then comes where Xyce
    # restarts the integration)
    c.append('    int ps = pyms_cx_pend_[k];')
    c.append('    if (ps >= 0 && ps < pyms_nheld_ && pyms_Hc_[ps] != 0.0 && t1 >= 0.0) {')
    c.append('      double dt = (t0 >= 0.0 && t1 > t0) ? (t1 - t0) : 1e-9;')
    c.append('      breakPointTimes.push_back(Util::BreakPoint(t1 + std::max(1e-4 * dt, 1e-15)));')
    c.append('      continue;')
    c.append('    }')
    c.append('    if (t0 < 0.0 || t1 <= t0 || c1 == c0 || c1 == 0.0) continue;')
    c.append('    double dt = t1 - t0, tc = t1 - c1 * dt / (c1 - c0);')
    c.append('    int d = pyms_cx_dir_[k];')
    c.append('    bool rising = c1 < 0.0 && c1 > c0, falling = c1 > 0.0 && c1 < c0;')
    c.append('    if (!((d >= 0 && rising) || (d <= 0 && falling))) { pyms_cx_np_[k] = 0; continue; }')
    # one prediction per approach: after landing on it, the next step goes on
    # and finds the crossing (a prediction from the restart's tiny steps
    # would creep toward the crossing in ever smaller steps)
    c.append('    if (pyms_cx_np_[k] > 0) continue;')
    c.append('    if (tc > t1 + 1e-3 * dt && tc < t1 + 2.0 * dt) { pyms_cx_np_[k]++; breakPointTimes.push_back(Util::BreakPoint(tc)); }')
    c.append('  }')
    c.append('  return true;')
    c.append('}')
    c.append('')

    # Register LIDs. Optional external ports (i >= num_nodes_required)
    # may be absent if the netlist gave fewer nodes than the .va
    # declared — in that case extLIDVec.size() < n_ext. Use -1 as a
    # sentinel for missing ports; every place that indexes a solution
    # / load vector by li_<n> must guard against -1 (see stamp paths
    # below). Internal nodes are always present.
    c.append('void Instance::registerLIDs(const std::vector<int> &intLIDVec,')
    c.append('                            const std::vector<int> &extLIDVec) {')
    c.append('  DeviceInstance::registerLIDs(intLIDVec, extLIDVec);')
    c.append('  const int n_ext_supplied = (int)extLIDVec.size();')
    for i, n in enumerate(all_nodes):
        if i < n_ext:
            c.append(f'  li_{n} = ({i} < n_ext_supplied) ? extLIDVec[{i}] : -1;')
        else:
            c.append(f'  li_{n} = intLIDVec[{i - n_ext}];')
    for k in range(n_cur):
        c.append(f'  li_cur_{k} = intLIDVec[{n_int + k}];')
    c.append('}')
    c.append('')

    c.append('void Instance::registerStateLIDs(const std::vector<int> &v) {}')
    c.append('void Instance::registerStoreLIDs(const std::vector<int> &v) {}')
    c.append('std::map<int,std::string> &Instance::getIntNameMap() {')
    c.append('  static std::map<int,std::string> m;')
    for i, n in enumerate(internals):
        c.append(f'  m[{i}] = "{n}";')
    for k, (_a, _b, _label, nm) in enumerate(cur_info):
        c.append(f'  m[{n_int + k}] = "{nm}";')
    c.append('  return m;')
    c.append('}')
    c.append('std::map<int,std::string> &Instance::getStoreNameMap() {')
    c.append('  static std::map<int,std::string> m; return m;')
    c.append('}')
    c.append('const std::vector<std::string> &Instance::getDepSolnVars() {')
    c.append('  static std::vector<std::string> v; return v;')
    c.append('}')
    c.append('')

    # loadNodeSymbols: register internal nodes with the symbol table
    c.append('void Instance::loadNodeSymbols(Util::SymbolTable &symbol_table) const {')
    for i, n in enumerate(internals):
        c.append(f'  addInternalNode(symbol_table, li_{n}, getName(), "{n}");')
    for k, (_a, _b, _label, nm) in enumerate(cur_info):
        c.append(f'  addInternalNode(symbol_table, li_cur_{k}, getName(), "{nm}");')
    c.append('}')
    c.append('')

    c.append('const std::vector<std::vector<int>> &Instance::jacobianStamp() const {')
    c.append('  return jacStamp_;')
    c.append('}')
    c.append('')
    c.append('void Instance::registerJacLIDs(const std::vector<std::vector<int>> &j) {')
    c.append('  jacLIDs_ = j;')
    c.append('}')
    c.append('')

    # Build branch-to-node incidence from contribution list
    # When nodes are collapsed, map internal names to their external equivalents
    # (e.g. DI→D, SI→S, GP→G when series resistance is zero)
    branch_map = []
    seen_branches = {}
    node_idx = {n: i for i, n in enumerate(all_nodes)}

    # Build collapse map: for branch node names not in all_nodes,
    # find their target via V(a,b)<+0 shorting relationships
    from vae.ginac_emitter import GiNaCEmitter
    _unknown_nodes = set()
    for ck, cbranch, cexpr in contribs:
        for bn in cbranch:
            if bn and bn not in node_idx:
                _unknown_nodes.add(bn)
    _emitter = GiNaCEmitter(mod, param_values={})
    collapse_map = {}
    for n in _unknown_nodes:
        target = _emitter._resolve_shorted_node(n, all_nodes)
        if target and target in node_idx:
            collapse_map[n] = target

    def resolve_node(name):
        if name is None:
            return -1
        if name in node_idx:
            return node_idx[name]
        if name in collapse_map:
            return node_idx[collapse_map[name]]
        return -1

    # Track branch label strings alongside (a_idx, b_idx) so we can
    # emit a remap table the runtime uses to align the wrapper's
    # branch ordering with the GiNaC .so's. Format matches the GiNaC
    # emitter exactly: ``I(b_re1)`` / ``V(n1,gnd)`` etc., joined with
    # commas, no spaces.
    branch_labels = []
    # Resolve a contribution endpoint to a node-index. Verilog-A
    # accepts three shapes:
    #   I(node_a, node_b)         — explicit pair
    #   I(node_a)                 — pair (node_a, gnd)
    #   I(named_branch)           — branch_name resolves to (primary, secondary)
    # The parser stores single-element tuples for the named-branch
    # case; expand it here using ``mod.branch_map`` /
    # ``branch_neg_map`` so the wrapper stamps into both KCL rows
    # with the correct sign.
    mod_branch_map = getattr(mod, 'branch_map', {}) or {}
    mod_branch_neg_map = getattr(mod, 'branch_neg_map', {}) or {}
    for ck, cbranch, cexpr in contribs:
        key = (ck, cbranch)
        if key not in seen_branches:
            if len(cbranch) == 1 and cbranch[0] in mod_branch_map:
                a = mod_branch_map[cbranch[0]]
                b = mod_branch_neg_map.get(cbranch[0])
            else:
                a = cbranch[0] if len(cbranch) >= 1 else None
                b = cbranch[1] if len(cbranch) >= 2 else None
            a_idx = resolve_node(a)
            b_idx = resolve_node(b)
            seen_branches[key] = len(branch_map)
            branch_map.append((a_idx, b_idx))
            label = f'{ck.name}({",".join(cbranch)})'
            branch_labels.append(label)
    n_branches = len(branch_map)

    # updateIntermediateVars — call VAE .so, get branch-indexed F/Q
    c.append('bool Instance::updateIntermediateVars() {')
    c.append('  Linear::Vector *solVec = extData.nextSolVectorPtr;')
    c.append(f'  memset(F_, 0, sizeof(F_));')
    c.append(f'  memset(Q_, 0, sizeof(Q_));')
    c.append(f'  memset(dFdx_, 0, sizeof(dFdx_));')
    c.append(f'  memset(dQdx_, 0, sizeof(dQdx_));')
    c.append('')
    # When the device matched the simple R/L/C pattern, the inline
    # stamp below is the authoritative source of F/Q/Jacobian for
    # this device — skip the vae_eval dispatch entirely so we don't
    # double-stamp or pull in a possibly-singular compact-model
    # contribution.
    if simple_rlc:
        c.append('  if (false /* simple R/L/C path: skip vae_eval */) {')
    else:
        c.append('  if (vae_eval_ && vae_jac_) {')
    c.append('    VaeState state = {};')
    c.append('    g_pyms_cur = this;  // callback params resolve to THIS instance')
    for i, n in enumerate(all_nodes):
        if i < n_ext and i >= num_nodes_required:
            # Optional external port — netlist may not have supplied it.
            c.append(f'    state.V[{i}] = (li_{n} >= 0) ? (*solVec)[li_{n}] : 0.0;')
        else:
            c.append(f'    state.V[{i}] = (*solVec)[li_{n}];')
    # the device temperature through this instance's source (1.1.6); the
    # library computes $vt from it (the Vt field: the same, for the layout)
    c.append('    state.T = *pyms_tsrc_;')
    c.append('    state.Vt = CONSTKoverQ * state.T;')
    # The analysis context of this evaluation (the math library's shadow) and
    # its held state (1.1.4).  an: dc/op 0, tran 1, ac 2, noise 3.  phase: 0
    # the first point an analysis starts from (its operating point; a
    # transient with uic: its first evaluation), where @(initial_step) runs;
    # 1 the later points (dc sweep points, transient steps); 2 the
    # small-signal sweep of ac / noise (the frequency is set): the operating
    # point's evaluation again (the same held values it started from, Hr) with
    # the regime its last evaluation chose (Hl), writing nothing.  Going from
    # phase 0 to phase 1 commits the values of the last operating-point
    # evaluation; acceptStep commits a transient's accepted points.
    c.append('    {')
    c.append('      const SolverState &ss = getSolverState();')
    c.append('      int an = ss.noiseFlag ? 3 : (ss.acopFlag ? 2 : ((ss.tranopFlag || ss.transientFlag) ? 1 : 0));')
    c.append('      int ph;')
    c.append('      if (an >= 2 && ss.currFreq_ > 0.0) ph = 2;')
    c.append('      else if ((ss.dcopFlag && ss.timeStepNumber_ == 0) || !pyms_seen_init_) ph = 0;')
    c.append('      else ph = 1;')
    # (the evaluation acceptStep makes of a transient's accepted point: where
    # a cross / above event fires)
    c.append('      if (pyms_accept_eval_ && an == 1 && ph == 1) ph = 3;')
    # The first evaluation of an analysis (its operating point; with .step,
    # every step point's): the held state starts again from the declaration
    # values (1.1.5; 1.1.4 went on from the values the previous step point
    # ended with), and so does the crossing prediction of the cross events.
    c.append('      if (ph == 0 && pyms_last_ph_ != 0 && pyms_nheld_ > 0) {')
    c.append('        pyms_Hc_ = pyms_H0_; pyms_Hp_ = pyms_H0_;')
    c.append('        for (int k = 0; k < pyms_cx_n_; k++) { pyms_cx_c0_[k] = pyms_cx_c1_[k] = 0.0; pyms_cx_t0_[k] = pyms_cx_t1_[k] = -1.0; pyms_cx_np_[k] = 0; }')
    c.append('      }')
    c.append('      pyms_last_ph_ = ph;')
    c.append('      if (ph == 0) { pyms_seen_init_ = true; if (!pyms_accept_eval_) pyms_committed_ = false; }')
    c.append('      else if (ph == 1 && !pyms_committed_ && pyms_nheld_ > 0) { pyms_Hc_ = pyms_Hp_; pyms_committed_ = true; }')
    c.append('      state.t = ss.currTime_; state.an = an; state.phase = ph;')
    c.append('      state.Hr = pyms_nheld_ > 0 ? pyms_Hc_.data() : nullptr;')
    c.append('      state.Hw = (pyms_nheld_ > 0 && ph != 2) ? pyms_Hp_.data() : nullptr;')
    c.append('      state.Hl = (pyms_nheld_ > 0 && ph == 2) ? pyms_Hp_.data() : nullptr;')
    c.append('    }')
    c.append('')
    # The library's shape: rows (branches), node columns, and its
    # branch-current unknowns (columns after the nodes, one equation row
    # each). The shell's arrays are sized for the shell's own counts, which
    # bound the library's (its walk culls dead code; the unknowns are the
    # same list): MAXB rows, MAXC columns (the VaeState capacity).
    MAXB = n_branches + n_cur + 1
    MAXC = n_unk
    c.append(f'    // The math library: row count, node count, branch-current unknowns')
    c.append(f'    int so_n_branches = {n_branches};')
    c.append(f'    int so_n_nodes = {n_total};')
    c.append(f'    int so_n_cur = 0;')
    c.append(f'    typedef int (*IntFn)();')
    c.append(f'    typedef int (*IntIntFn)(int);')
    c.append(f'    typedef const char* (*LabelFn)(int);')
    c.append(f'    IntFn nb_fn = vae_dl_ ? (IntFn)dlsym(vae_dl_, "vae_n_branches") : 0;')
    c.append(f'    IntFn nn_fn = vae_dl_ ? (IntFn)dlsym(vae_dl_, "vae_n_nodes") : 0;')
    c.append(f'    IntFn nc_fn = vae_dl_ ? (IntFn)dlsym(vae_dl_, "vae_n_currents") : 0;')
    c.append(f'    LabelFn lbl_fn = vae_dl_ ? (LabelFn)dlsym(vae_dl_, "vae_branch_label") : 0;')
    c.append(f'    if (nb_fn) so_n_branches = nb_fn();')
    c.append(f'    if (nn_fn) so_n_nodes = nn_fn();')
    c.append(f'    if (nc_fn) so_n_cur = nc_fn();')
    c.append(f'    // A library that exports its branch labels is branch-indexed by')
    c.append(f'    // construction (the GiNaC emitter always does); the count test is')
    c.append(f'    // only for legacy libraries without labels.')
    c.append(f'    bool node_indexed = !lbl_fn && (so_n_branches == so_n_nodes);')
    c.append(f'    const int so_stride = so_n_nodes + so_n_cur;  // Jacobian columns of the library')
    c.append(f'    if (so_n_branches > {MAXB} || so_stride > {MAXC}) {{')
    c.append(f'      Report::UserError0() << "PyMS: the math library of {mod.name} has " << so_n_branches')
    c.append(f'        << " rows and " << so_stride << " columns; this device shell holds at most {MAXB} rows and {MAXC} columns (a stale shell cache? empty PYMS_CACHE)";')
    c.append(f'      return false;')
    c.append(f'    }}')
    c.append('')
    # Label-keyed remaps, built once: shell branch wi -> library row gi (or
    # -1: dead code in this elaboration); shell unknown k -> the library's
    # row of its equation (cur_row) and its column (cur_socol); library
    # column -> shell column (so_col_map). A library unknown the shell does
    # not know cannot be fed: a stale shell, reported.
    c.append(f'    static const char* wrapper_branch_labels[] = {{')
    for label in branch_labels:
        esc = label.replace('\\', '\\\\').replace('"', '\\"')
        c.append(f'      "{esc}",')
    c.append(f'      ""')
    c.append(f'    }};')
    c.append(f'    static const char* wrapper_cur_labels[] = {{')
    for (_a, _b, label, _nm) in cur_info:
        esc = label.replace('\\', '\\\\').replace('"', '\\"')
        c.append(f'      "{esc}",')
    c.append(f'      ""')
    c.append(f'    }};')
    c.append(f'    static const int wrapper_branch_nodes[][2] = {{')
    for (a_idx, b_idx) in branch_map:
        c.append(f'      {{{a_idx}, {b_idx}}},')
    c.append(f'      {{-1, -1}}')
    c.append(f'    }};')
    c.append(f'    static const int wrapper_cur_nodes[][2] = {{')
    for (a_idx, b_idx, _label, _nm) in cur_info:
        c.append(f'      {{{a_idx}, {b_idx}}},')
    c.append(f'      {{-1, -1}}')
    c.append(f'    }};')
    c.append(f'    const int li_cur[] = {{ ' + ''.join(f'li_cur_{k}, ' for k in range(n_cur)) + '-1 };')
    c.append(f'    const int li_node[] = {{ ' + ''.join(f'li_{n}, ' for n in all_nodes) + '-1 };')
    # Ground's local index (its global index is -1; a port wired to ground
    # carries this LID, an absent optional port -1).
    c.append('    const int li_gnd = solVec->omap() ? solVec->omap()->globalToLocalIndex(-1) : -1;')
    c.append(f'    bool &remap_built = pyms_remap_built_;')
    c.append(f'    if (!remap_built) {{')
    c.append(f'      pyms_branch_remap_.assign({max(n_branches, 1)}, -1);')
    c.append(f'      pyms_cur_row_.assign({max(n_cur, 1)}, -1);')
    c.append(f'      pyms_cur_socol_.assign({max(n_cur, 1)}, -1);')
    c.append(f'      pyms_so_col_map_.assign({MAXC}, -1);')
    c.append(f'    }}')
    c.append(f'    int *branch_remap = pyms_branch_remap_.data(), *cur_row = pyms_cur_row_.data();')
    c.append(f'    int *cur_socol = pyms_cur_socol_.data(), *so_col_map = pyms_so_col_map_.data();')
    c.append(f'    if (!remap_built) {{')
    c.append(f'      for (int wi = 0; wi < {n_branches}; wi++) branch_remap[wi] = -1;')
    c.append(f'      for (int k = 0; k < {n_cur}; k++) {{ cur_row[k] = -1; cur_socol[k] = -1; }}')
    c.append(f'      for (int ci = 0; ci < {MAXC}; ci++) so_col_map[ci] = (ci < so_n_nodes && ci < {n_total}) ? ci : -1;')
    c.append(f'      if (lbl_fn) {{')
    c.append(f'        for (int gi = 0; gi < so_n_branches; gi++) {{')
    c.append(f'          const char *lbl = lbl_fn(gi);')
    c.append(f'          for (int wi = 0; wi < {n_branches}; wi++) {{')
    c.append(f'            if (lbl && std::string(lbl) == wrapper_branch_labels[wi]) {{ branch_remap[wi] = gi; break; }}')
    c.append(f'          }}')
    c.append(f'        }}')
    c.append(f'      }} else {{')
    c.append(f'        // No label export — assume index-aligned (legacy .so).')
    c.append(f'        for (int wi = 0; wi < {n_branches}; wi++) branch_remap[wi] = wi;')
    c.append(f'      }}')
    c.append(f'      LabelFn clbl_fn = vae_dl_ ? (LabelFn)dlsym(vae_dl_, "vae_current_label") : 0;')
    c.append(f'      IntIntFn crow_fn = vae_dl_ ? (IntIntFn)dlsym(vae_dl_, "vae_current_row") : 0;')
    c.append(f'      for (int sk = 0; sk < so_n_cur; sk++) {{')
    c.append(f'        const char *lbl = clbl_fn ? clbl_fn(sk) : 0;')
    c.append(f'        int wk = -1;')
    c.append(f'        for (int k = 0; k < {n_cur}; k++) if (lbl && std::string(lbl) == wrapper_cur_labels[k]) {{ wk = k; break; }}')
    c.append(f'        if (wk < 0) {{')
    c.append(f'          Report::UserError0() << "PyMS: the math library of {mod.name} has a branch-current unknown "')
    c.append(f'            << (lbl ? lbl : "?") << " this device shell does not know (a stale shell cache? empty PYMS_CACHE)";')
    c.append(f'          continue;')
    c.append(f'        }}')
    c.append(f'        cur_row[wk] = crow_fn ? crow_fn(sk) : -1;')
    c.append(f'        cur_socol[wk] = so_n_nodes + sk;')
    c.append(f'        so_col_map[so_n_nodes + sk] = {n_total} + wk;')
    c.append(f'      }}')
    c.append(f'      remap_built = true;')
    c.append(f'    }}')
    c.append(f'    for (int k = 0; k < {n_cur}; k++)')
    c.append(f'      if (cur_socol[k] >= 0) state.I[cur_socol[k] - so_n_nodes] = (*solVec)[li_cur[k]];')
    c.append('')
    c.append(f'    double Fb[{MAXB}]={{}}, Qb[{MAXB}]={{}};')
    c.append(f'    vae_eval_(&state, Fb, Qb);')
    # (1.1.8, B25) A Newton evaluation whose held update took another branch
    # than the previous iteration's (`if (V > pk) pk = V;` written at one
    # iterate, not at the next) leaves the iterate a solution of neither
    # branch; the solver's update test can still pass (the step is tiny), and
    # Xyce's default trapezoid rule carries the residual error of an
    # algebraic row on, alternating in sign, without damping it: a held peak
    # that grazes its own value (a periodic source) rang +-8e-8 V and, with a
    # tight reltol, crushed the time step to 1e-17 s (FIX8 h1.cir).  Such a
    # solve is not converged (isConverged; Xyce's device convergence test) --
    # one more iteration solves the branch the iterate is in; at most three
    # per solve, so a solution on the boundary still converges.
    c.append('    if (pyms_nheld_ > 0 && state.Hw && !pyms_accept_eval_) {')
    c.append('      const int _it = getSolverState().newtonIter;')
    c.append('      std::vector<char> _pat(pyms_nheld_);')
    c.append('      for (int k = 0; k < pyms_nheld_; k++) _pat[k] = (state.Hw[k] != state.Hr[k]);')
    c.append('      if (_it <= 0 || _it != pyms_cv_it_ + 1) { pyms_cv_forced_ = 0; pyms_cv_ok_ = true; }')
    c.append('      else if (_pat != pyms_cv_pat_ && pyms_cv_forced_ < 3) { pyms_cv_ok_ = false; pyms_cv_forced_++; }')
    c.append('      else pyms_cv_ok_ = true;')
    c.append('      pyms_cv_pat_.swap(_pat); pyms_cv_it_ = _it;')
    c.append('    }')
    c.append('')
    c.append('    // Sanitize NaN/Inf')
    c.append(f'    for (int i = 0; i < {MAXB}; i++) {{')
    c.append(f'      if (std::isnan(Fb[i]) || std::isinf(Fb[i])) Fb[i] = 0.0;')
    c.append(f'      if (std::isnan(Qb[i]) || std::isinf(Qb[i])) Qb[i] = 0.0;')
    c.append(f'    }}')
    c.append('')
    c.append(f'    if (node_indexed) {{')
    c.append(f'      // Node-indexed: F[i] = KCL contribution for node i')
    c.append(f'      for (int i = 0; i < {n_total}; i++) {{')
    c.append(f'        F_[i] += Fb[i];')
    c.append(f'        Q_[i] += Qb[i];')
    c.append(f'      }}')
    c.append(f'      // Node-indexed Jacobian: dF[i]/dV[j]')
    c.append(f'      double dFn[{n_total*n_total}]={{}}, dQn[{n_total*n_total}]={{}};')
    c.append(f'      vae_jac_(&state, dFn, dQn);')
    c.append(f'      for (int i = 0; i < {n_total}; i++)')
    c.append(f'        for (int j = 0; j < {n_total}; j++) {{')
    c.append(f'          dFdx_[i][j] += dFn[i*{n_total}+j];')
    c.append(f'          dQdx_[i][j] += dQn[i*{n_total}+j];')
    c.append(f'        }}')
    c.append(f'    }} else {{')
    c.append(f'      // Branch-indexed. The library walks the same AST with parameter-')
    c.append(f'      // resolved condition culling, so its rows are a subset of the')
    c.append(f'      // shell\'s contribution list; the label remaps above align them.')
    c.append(f'      // Its Jacobian has so_stride columns: the nodes, then its')
    c.append(f'      // branch-current unknowns; so_col_map carries each to a shell column.')
    c.append(f'      double dFb[{MAXB * MAXC}]={{}}, dQb[{MAXB * MAXC}]={{}};')
    c.append(f'      vae_jac_(&state, dFb, dQb);')
    c.append(f'      // Current contributions: KCL rows of the branch\'s two nodes.')
    c.append(f'      for (int wi = 0; wi < {n_branches}; wi++) {{')
    c.append(f'        int gi = branch_remap[wi]; if (gi < 0) continue;')
    c.append(f'        int a = wrapper_branch_nodes[wi][0], b = wrapper_branch_nodes[wi][1];')
    c.append(f'        if (a >= 0) {{ F_[a] += Fb[gi]; Q_[a] += Qb[gi]; }}')
    c.append(f'        if (b >= 0) {{ F_[b] -= Fb[gi]; Q_[b] -= Qb[gi]; }}')
    c.append(f'        for (int ci = 0; ci < so_stride; ci++) {{')
    c.append(f'          int sc = so_col_map[ci]; if (sc < 0) continue;')
    c.append(f'          double df = dFb[gi*so_stride + ci], dq = dQb[gi*so_stride + ci];')
    c.append(f'          if (a >= 0) {{ dFdx_[a][sc] += df; dQdx_[a][sc] += dq; }}')
    c.append(f'          if (b >= 0) {{ dFdx_[b][sc] -= df; dQdx_[b][sc] -= dq; }}')
    c.append(f'        }}')
    c.append(f'      }}')
    c.append(f'      // Branch-current unknowns: the branch equation (the library\'s')
    c.append(f'      // residual) in the unknown\'s own row, the current in the two KCL')
    c.append(f'      // rows. An unknown the library has no equation for (its branch is')
    c.append(f'      // dead in this elaboration) is pinned to 0, and so is one whose two')
    c.append(f'      // ends are one circuit node (both on ground, or a port left open:')
    c.append(f'      // BSIM-CMG\'s ``Temp(t) <+ 0.0`` with t grounded): the branch')
    c.append(f'      // voltage is 0 by the netlist, the current would be undetermined,')
    c.append(f'      // and the matrix singular.')
    c.append(f'      for (int k = 0; k < {n_cur}; k++) {{')
    c.append(f'        int R = {n_total} + k;')
    c.append(f'        double ik = (*solVec)[li_cur[k]];')
    c.append(f'        int a = wrapper_cur_nodes[k][0], b = wrapper_cur_nodes[k][1];')
    c.append(f'        int la = a >= 0 ? li_node[a] : -1, lb = b >= 0 ? li_node[b] : -1;')
    c.append(f'        if (la == li_gnd) la = -1;')
    c.append(f'        if (lb == li_gnd) lb = -1;')
    c.append(f'        if (la == lb) {{')
    c.append(f'          F_[R] += ik; dFdx_[R][R] += 1.0; continue;')
    c.append(f'        }}')
    c.append(f'        if (a >= 0) {{ F_[a] += ik; dFdx_[a][R] += 1.0; }}')
    c.append(f'        if (b >= 0) {{ F_[b] -= ik; dFdx_[b][R] -= 1.0; }}')
    c.append(f'        int gi = cur_row[k];')
    c.append(f'        if (gi >= 0) {{')
    c.append(f'          F_[R] += Fb[gi]; Q_[R] += Qb[gi];')
    c.append(f'          for (int ci = 0; ci < so_stride; ci++) {{')
    c.append(f'            int sc = so_col_map[ci]; if (sc < 0) continue;')
    c.append(f'            dFdx_[R][sc] += dFb[gi*so_stride + ci]; dQdx_[R][sc] += dQb[gi*so_stride + ci];')
    c.append(f'          }}')
    c.append(f'        }} else {{')
    c.append(f'          F_[R] += ik; dFdx_[R][R] += 1.0;')
    c.append(f'        }}')
    c.append(f'      }}')
    c.append(f'    }}')
    c.append('    pymsPoll_();')
    c.append('  }')  # close if (vae_eval_ && vae_jac_)
    c.append('')
    # ----------------------------------------------------------------
    # Built-in linear-element stamp for the SIMPLE R / L / C pattern.
    # Equivalent to what Xyce's primitive R / L / C devices do.
    # Skips the vae_eval pipeline entirely (no .so dlopen, no
    # characterisation needed) for the cases where the device IS just
    # a linear element with a single value parameter. xyceModelGroup
    # + a conventional instance-param name is the only signal we look
    # for.
    # ----------------------------------------------------------------
    if simple_rlc:
        kind = simple_rlc[0]
        # Pick the resistor's two terminals. For most R/L/C primitives
        # the first two declared ports are the +/- terminals, but the
        # IHP r3_cmc-family declares (n1, nc, n2, ...) where nc is the
        # body-bias control node and the actual resistor body sits
        # between n1 and n2. Use n1/n2 by name when present.
        port_names = [p[0] for p in ports]
        if 'n1' in port_names and 'n2' in port_names:
            p0, p1 = 'n1', 'n2'
        elif kind == 'R_SHEET' and len(port_names) >= 3:
            # Conventional compact-resistor port order n1, nc, n2, ...
            p0, p1 = port_names[0], port_names[2]
        else:
            p0, p1 = port_names[0], port_names[1]
        # Map back to the position-indexed F_/dFdx_ slots so the stamp
        # writes to the right rows even when p0/p1 aren't the literal
        # first two indices.
        a_pos = port_names.index(p0)
        b_pos = port_names.index(p1)
        c.append(f'  // Simple {kind} element: stamp linear contribution')
        # For Resistor/Capacitor/Inductor model groups, always take
        # this path — don't gate on ``!vae_eval_``. The Verilog-A
        # compact model's eval may compile and dlopen successfully
        # but still produce a singular Jacobian (e.g. r3_cmc's
        # internal nodes with sparse columns). Behavioral linear-
        # element stamping converges, which is what "if it's a
        # resistor, it should behave like one" actually means.
        c.append('  {')
        c.append(f'    double V_drop = (*solVec)[li_{p0}] - (*solVec)[li_{p1}];')
        a, b = a_pos, b_pos
        if kind == 'R':
            vparam = simple_rlc[1]
            c.append(f'    double R_val = {vparam};')
            c.append('    if (R_val < 1e-12) R_val = 1e-12;  // div-by-zero guard')
            c.append('    double G = 1.0 / R_val;')
            c.append(f'    F_[{a}] += G * V_drop;')
            c.append(f'    F_[{b}] -= G * V_drop;')
            c.append(f'    dFdx_[{a}][{a}] += G;')
            c.append(f'    dFdx_[{a}][{b}] -= G;')
            c.append(f'    dFdx_[{b}][{a}] -= G;')
            c.append(f'    dFdx_[{b}][{b}] += G;')
        elif kind == 'R_SHEET':
            # Compact-resistor pattern: R = rsh * l / w. The rsh
            # parameter may be Model:: or Instance::; l, w are
            # always Instance::. See pattern-detection above.
            _, rsh, lpar, wpar = simple_rlc
            c.append(f'    double rsh_val = {rsh};')
            c.append(f'    double l_val = {lpar};')
            c.append(f'    double w_val = {wpar};')
            c.append('    if (w_val < 1e-12) w_val = 1e-12;')
            c.append('    double R_val = rsh_val * l_val / w_val;')
            c.append('    if (R_val < 1e-12) R_val = 1e-12;')
            c.append('    double G = 1.0 / R_val;')
            c.append(f'    F_[{a}] += G * V_drop;')
            c.append(f'    F_[{b}] -= G * V_drop;')
            c.append(f'    dFdx_[{a}][{a}] += G;')
            c.append(f'    dFdx_[{a}][{b}] -= G;')
            c.append(f'    dFdx_[{b}][{a}] -= G;')
            c.append(f'    dFdx_[{b}][{b}] += G;')
        elif kind == 'C':
            vparam = simple_rlc[1]
            c.append(f'    double C_val = {vparam};')
            if rlc_loss and rlc_loss['rpar']:
                _emit_rpar(c, rlc_loss['rpar'], a, b)
            if rlc_loss and rlc_loss['mid_name'] is not None:
                # Series ESR: p --RSER-- mid, capacitor between mid and n.
                _emit_rser(c, rlc_loss['rser'], p0, rlc_loss['mid_name'], a, rlc_loss['mid_idx'])
                m, mn = rlc_loss['mid_idx'], rlc_loss['mid_name']
                c.append(f'    double Vmn = (*solVec)[li_{mn}] - (*solVec)[li_{p1}];')
                c.append('    double Q = C_val * Vmn;')
                c.append(f'    Q_[{m}] += Q;')
                c.append(f'    Q_[{b}] -= Q;')
                c.append(f'    dQdx_[{m}][{m}] += C_val;')
                c.append(f'    dQdx_[{m}][{b}] -= C_val;')
                c.append(f'    dQdx_[{b}][{m}] -= C_val;')
                c.append(f'    dQdx_[{b}][{b}] += C_val;')
            else:
                # Ideal C between p and n (no series ESR).
                c.append('    double Q = C_val * V_drop;')
                c.append(f'    Q_[{a}] += Q;')
                c.append(f'    Q_[{b}] -= Q;')
                c.append(f'    dQdx_[{a}][{a}] += C_val;')
                c.append(f'    dQdx_[{a}][{b}] -= C_val;')
                c.append(f'    dQdx_[{b}][{a}] -= C_val;')
                c.append(f'    dQdx_[{b}][{b}] += C_val;')
        elif kind == 'L':
            # NB: the inductive reactance is still an OP-only near-short
            # placeholder (transient needs the full VAE current-state model);
            # but RSER/RPAR loss are honored so they aren't silently dropped.
            vparam = simple_rlc[1]
            c.append(f'    double L_val = {vparam};')
            c.append('    (void) L_val;  // OP-only fallback; reactance needs full VAE')
            if rlc_loss and rlc_loss['rpar']:
                _emit_rpar(c, rlc_loss['rpar'], a, b)
            if rlc_loss and rlc_loss['mid_name'] is not None:
                _emit_rser(c, rlc_loss['rser'], p0, rlc_loss['mid_name'], a, rlc_loss['mid_idx'])
                m, mn = rlc_loss['mid_idx'], rlc_loss['mid_name']
                c.append(f'    double Vmn = (*solVec)[li_{mn}] - (*solVec)[li_{p1}];')
                c.append(f'    F_[{m}] += 1e3 * Vmn;')
                c.append(f'    F_[{b}] -= 1e3 * Vmn;')
                c.append(f'    dFdx_[{m}][{m}] += 1e3;')
                c.append(f'    dFdx_[{m}][{b}] -= 1e3;')
                c.append(f'    dFdx_[{b}][{m}] -= 1e3;')
                c.append(f'    dFdx_[{b}][{b}] += 1e3;')
            else:
                c.append(f'    F_[{a}] += 1e3 * V_drop;')
                c.append(f'    F_[{b}] -= 1e3 * V_drop;')
                c.append(f'    dFdx_[{a}][{a}] += 1e3;')
                c.append(f'    dFdx_[{a}][{b}] -= 1e3;')
                c.append(f'    dFdx_[{b}][{a}] -= 1e3;')
                c.append(f'    dFdx_[{b}][{b}] += 1e3;')
        c.append('  }')
        c.append('')
    # Diagonal regularization: a conductance gmin from every node of the
    # device to ground keeps the matrix non-singular at the initial guess
    # (internal and optional nodes often have sparse or conditional Jacobian
    # columns: r3_cmc's thermal dt with only Pwr() contributions, body-pickup
    # pins with a dead diode branch). It is the device gmin of the options
    # (.options device gmin=..., default 1e-12), as Xyce's own devices use
    # it, on every node: it was 1e-12 on the ports and a fixed 1e-9 on the
    # internal nodes, which put a 1e-9 S leak on an internal node and a
    # relative error of 6.7e-7 in a series V/I divider (PYMS round 1, E-4).
    c.append('  // Diagonal regularization (the device gmin) to keep the matrix non-singular')
    c.append('  const double gmin_ext = getDeviceOptions().gmin;')
    c.append('  const double gmin_int = getDeviceOptions().gmin;')
    c.append('  Linear::Vector *solVec2 = extData.nextSolVectorPtr;')
    for i, n in enumerate(all_nodes):
        if i < num_nodes_required:
            # Required external port — full external observable.
            c.append(f'  F_[{i}] += gmin_ext * (*solVec2)[li_{n}];')
            c.append(f'  dFdx_[{i}][{i}] += gmin_ext;')
        elif i < n_ext:
            # Optional external port — may be absent. When present,
            # treat as internal-style (the netlist usually wires it to
            # a per-instance floating node, see gnucap2xyce padding).
            c.append(f'  if (li_{n} >= 0) {{ F_[{i}] += gmin_int * (*solVec2)[li_{n}]; dFdx_[{i}][{i}] += gmin_int; }}')
        else:
            # Internal node — always present.
            c.append(f'  F_[{i}] += gmin_int * (*solVec2)[li_{n}];')
            c.append(f'  dFdx_[{i}][{i}] += gmin_int;')
    c.append('')
    c.append('  return true;')
    c.append('}')
    c.append('')

    c.append('bool Instance::updatePrimaryState() {')
    c.append('  return updateIntermediateVars();')
    c.append('}')
    c.append('')

    # DAE loading — stamp node-indexed contributions into Xyce matrix
    # Helper to skip optional-port stamps when the netlist didn't
    # supply the port. Required-port stamps stay unconditional (those
    # LIDs are guaranteed valid).
    def _maybe_optional(i, n, line):
        if i < n_ext and i >= num_nodes_required:
            return f'  if (li_{n} >= 0) {{ {line.strip()} }}'
        return line

    # Rows: the nodes (KCL), then the branch-current unknowns (their
    # equations); columns: every unknown.
    c.append('bool Instance::loadDAEFVector() {')
    c.append('  Linear::Vector &fVec = *(extData.daeFVectorPtr);')
    for i, n in enumerate(all_nodes):
        c.append(_maybe_optional(i, n, f'  fVec[li_{n}] += F_[{i}];'))
    for k in range(n_cur):
        c.append(f'  fVec[li_cur_{k}] += F_[{n_total + k}];')
    c.append('  return true;')
    c.append('}')
    c.append('')

    c.append('bool Instance::loadDAEQVector() {')
    c.append('  Linear::Vector &qVec = *(extData.daeQVectorPtr);')
    for i, n in enumerate(all_nodes):
        c.append(_maybe_optional(i, n, f'  qVec[li_{n}] += Q_[{i}];'))
    for k in range(n_cur):
        c.append(f'  qVec[li_cur_{k}] += Q_[{n_total + k}];')
    c.append('  return true;')
    c.append('}')
    c.append('')

    c.append('bool Instance::loadDAEdFdx() {')
    c.append('  Linear::Matrix &dFdx = *(extData.dFdxMatrixPtr);')
    for i, n in enumerate(all_nodes):
        for j in range(n_unk):
            line = f'  dFdx[li_{n}][jacLIDs_[{i}][{j}]] += dFdx_[{i}][{j}];'
            c.append(_maybe_optional(i, n, line))
    for k in range(n_cur):
        for j in range(n_unk):
            c.append(f'  dFdx[li_cur_{k}][jacLIDs_[{n_total + k}][{j}]] += dFdx_[{n_total + k}][{j}];')
    c.append('  return true;')
    c.append('}')
    c.append('')

    c.append('bool Instance::loadDAEdQdx() {')
    c.append('  Linear::Matrix &dQdx = *(extData.dQdxMatrixPtr);')
    for i, n in enumerate(all_nodes):
        for j in range(n_unk):
            line = f'  dQdx[li_{n}][jacLIDs_[{i}][{j}]] += dQdx_[{i}][{j}];'
            c.append(_maybe_optional(i, n, line))
    for k in range(n_cur):
        for j in range(n_unk):
            c.append(f'  dQdx[li_cur_{k}][jacLIDs_[{n_total + k}][{j}]] += dQdx_[{n_total + k}][{j}];')
    c.append('  return true;')
    c.append('}')
    c.append('')

    # Model implementation
    c.append('Model::Model(const Configuration &config, const ModelBlock &mb,')
    c.append('             const FactoryBlock &fb)')
    c.append('  : DeviceModel(mb, config.getModelParameters(), fb)')
    c.append('{')
    # Initialise every model member to its declared default, then let
    # the standard setModParams flow apply any overrides from the
    # .MODEL line. (Without setModParams the .MODEL line's param=
    # entries would be silently ignored — exactly the bug the
    # Instance constructor was hitting for instance params.)
    for pname, pdefault, is_inst, _cxx in params:
        if not is_inst or pname in _also_model:
            dval = _resolve_default(pname)
            c.append(f'  {_cxx} = {dval};')
    c.append('  setModParams(' + ('pyms_unalias_(mb.params)' if _aliases else 'mb.params') + ');')
    c.append('  updateDependentParameters();  // card values given as expressions of globals (A23/A24)')
    if _temp_par_model:
        c.append(f'  pyms_tp_val_ = {_cxx_of[_temp_par_model]}; pyms_tp_given_ = given("TEMP");')
    # Wire the MOSFET channel-type variable (xyceTypeVariable, e.g. TYPE) from
    # the .model nmos/pmos keyword. Xyce registers both model types but leaves
    # the TYPE var at its +1 default; without this a `pmos` model bakes TYPE=+1
    # into the JIT math and simulates as an nmos.
    _tv = _cxx_safe(type_variable) if type_variable else ''
    if _tv and any(_cxx == _tv for (_pn, _pd, _ii, _cxx) in params if not _ii):
        c.append('  { std::string _mt = getType(); for(auto &ch:_mt) ch = tolower(ch);')
        c.append(f'    if (_mt.find("pmos") != std::string::npos || _mt == "p") {_tv} = -1.0;')
        c.append(f'    else if (_mt.find("nmos") != std::string::npos || _mt == "n") {_tv} = 1.0; }}')
    c.append('  processParams();')
    c.append('}')
    c.append('')
    c.append('Model::~Model() {}')
    # Re-evaluate every model parameter whose default reads another parameter
    # or $simparam and that the card did not give, in dependency order, from
    # the card's values (the addPar default is the value with every parameter
    # at its default). One whose default reads an instance parameter cannot be
    # (the Model is shared); the math library evaluates it per instance.
    c.append('bool Model::processParams() {')
    if _aliases:
        c.append('  pyms_alias_given_(*this);')
    if _temp_par_model:
        c.extend(_tp_restore(_cxx_of[_temp_par_model]))
    for p, tree, deps in _dep_order:
        if _is_inst_of.get(p.name, True) or p.name not in _cxx_of:
            continue
        if any(d not in _cxx_of or _is_inst_of.get(d, False) for d in deps):
            c.append(f'  // {p.name}: its default reads an instance parameter; the math library evaluates it per instance')
            continue
        if p.name in _str_params or any(d in _str_params for d in deps):
            c.append(f'  // {p.name}: a string default or one that reads a string; the math library evaluates it')
            continue
        if p.name in _thermal:
            c.append(f'  // {p.name}: its default reads $temperature / $vt; the math library evaluates it at run time')
            continue
        c.append(f'  if (!given("{p.name.upper()}")) {_cxx_of[p.name]} = {_cxx_default(p, tree, False)};')
    c.append('  return true;')
    c.append('}')
    # A model parameter changed after the instances were built (.step of a
    # model parameter, a .global_param the card reads): every instance
    # processes its parameters again, and its math library follows the new
    # values (Instance::processParams).  Before 1.1.5 this did nothing.
    c.append('bool Model::processInstanceParams() {')
    c.append('  for (auto *inst : instanceContainer) inst->processParams();')
    c.append('  return true;')
    c.append('}')
    c.append('')
    c.append('void Model::forEachInstance(DeviceInstanceOp &op) const {')
    c.append('  for (auto *inst : instanceContainer) op(inst);')
    c.append('}')
    c.append('std::ostream &Model::printOutInstances(std::ostream &os) const {')
    c.append('  return os;')
    c.append('}')
    c.append('')

    # Factory
    c.append('Device *Traits::factory(const Configuration &configuration,')
    c.append('                        const FactoryBlock &factory_block) {')
    c.append('  return new DeviceMaster<Traits>(configuration, factory_block,')
    c.append('    factory_block.solverState_, factory_block.deviceOptions_);')
    c.append('}')
    c.append('')

    # Registration — use ADMS attributes if available so .model cards resolve
    # e.g. xyceModelGroup="MOSFET" → register as "m" with "nmos"/"pmos" model types
    c.append('void registerDevice(const DeviceCountMap &deviceMap,')
    c.append('                    const std::set<int> &levelSet) {')
    if model_group.upper() == 'MOSFET' and level_number:
        # Register as a MOSFET level, matching ADMS convention
        c.append(f'  Config<Traits>::addConfiguration()')
        c.append(f'    .registerDevice("m", {level_number})')
        c.append(f'    .registerModelType("nmos", {level_number})')
        c.append(f'    .registerModelType("pmos", {level_number});')
    elif model_group.upper() == 'DIODE' and level_number:
        c.append(f'  Config<Traits>::addConfiguration()')
        c.append(f'    .registerDevice("d", {level_number})')
        c.append(f'    .registerModelType("d", {level_number});')
    elif model_group.upper() == 'BJT' and level_number:
        c.append(f'  Config<Traits>::addConfiguration()')
        c.append(f'    .registerDevice("q", {level_number})')
        c.append(f'    .registerModelType("npn", {level_number})')
        c.append(f'    .registerModelType("pnp", {level_number});')
    elif model_group.upper() == 'RESISTOR' and level_number:
        c.append(f'  Config<Traits>::addConfiguration()')
        c.append(f'    .registerDevice("r", {level_number})')
        c.append(f'    .registerModelType("r", {level_number})')
        c.append(f'    .registerModelType("res", {level_number});')
    elif model_group.upper() == 'CAPACITOR' and level_number:
        c.append(f'  Config<Traits>::addConfiguration()')
        c.append(f'    .registerDevice("c", {level_number})')
        c.append(f'    .registerModelType("c", {level_number})')
        c.append(f'    .registerModelType("cap", {level_number});')
    elif model_group.upper() == 'INDUCTOR' and level_number:
        c.append(f'  Config<Traits>::addConfiguration()')
        c.append(f'    .registerDevice("l", {level_number})')
        c.append(f'    .registerModelType("l", {level_number})')
        c.append(f'    .registerModelType("ind", {level_number});')
    else:
        # Generic Y device — register with module name
        c.append(f'  Config<Traits>::addConfiguration()')
        c.append(f'    .registerDevice("{name.lower()}", 1)')
        c.append(f'    .registerModelType("{name.lower()}", 1);')
    c.append('}')
    c.append('')
    c.append(f'}} }} }}  // namespace PYMS_{NAME}')
    c.append('')

    # Auto-register on dlopen via constructor attribute
    c.append('__attribute__((constructor)) static void pyms_register() {')
    c.append(f'  Xyce::Device::PYMS_{NAME}::registerDevice(')
    c.append('    Xyce::Device::DeviceCountMap(), std::set<int>());')
    c.append('}')

    impl = '\n'.join(c) + '\n'
    return header, impl


def _decl_line(va_path, pname):
    """``file:line`` of ``parameter ... pname`` in the entry file by a search of
    its text, for an error message when the module carries no Param.loc
    (_param_loc); otherwise the entry file alone."""
    try:
        with open(va_path or '', errors='replace') as f:
            for i, ln in enumerate(f, 1):
                if re.search(r'\bparameter\b[^;]*\b' + re.escape(pname) + r'\b', ln):
                    return f'{va_path}:{i}'
    except OSError:
        pass
    return va_path or '<module>'


def resolve_entry_point(va_path: str) -> str:
    """Prefer ``<name>.va`` over ``<name>_main.va`` when both live in
    the same directory.

    ADMS-style compact models split the entry point in two: a thin
    wrapper ``<name>.va`` that defines convention macros (``\\`attr``,
    ``\\`__XYCE__``, etc.) and then ``\\`include``s ``<name>_main.va``,
    which carries the actual ``module`` declaration. Xyce's auto-loader
    grep over .va files lands on the file that contains ``module`` — the
    body — and hands that to PyMS, which sees the body's ``\\`attr(...)``
    as an undefined macro and loses the xyceModelGroup attrs.

    When invoked on a ``*_main.va`` (or, for some models like
    ``bsimcmg_nqsmod3.va``, any file in a code/ dir alongside a sibling
    wrapper that includes us), reroute to the wrapper so the macros and
    preprocessor predefines that come with it are picked up. Caller can
    still bypass via the env var ``PYMS_NO_WRAPPER_REROUTE``."""
    if os.environ.get('PYMS_NO_WRAPPER_REROUTE'):
        return va_path
    if not va_path.endswith('_main.va'):
        return va_path
    candidate = va_path[:-len('_main.va')] + '.va'
    if not os.path.exists(candidate) or os.path.samefile(candidate, va_path):
        return va_path
    # Confirm the candidate actually includes us — guards against
    # picking an unrelated sibling that happens to share a prefix.
    try:
        with open(candidate, errors='replace') as f:
            head = f.read(8192)
    except OSError:
        return va_path
    main_basename = os.path.basename(va_path)
    if f'"{main_basename}"' in head or f"'{main_basename}'" in head:
        return candidate
    return va_path


_USAGE = ("Usage: xyce_device_gen.py <input.va> [--output <dir>] "
          "[--define NAME[=VALUE]]... | --version")


def _parse_cli(argv):
    """``<input.va> [--output DIR] [--define NAME[=VALUE]]...`` or
    ``--version`` -> (va_path, output_dir, defines).

    Macros also come from the PYMS_DEFINES environment variable (shell
    words, ``NAME`` or ``NAME=VALUE`` each): the route through Xyce's
    ``.hdl`` line, whose PyMS subprocess inherits the environment
    (N_DEV_PyMS.C passes only ``<va> --output <cache>``).  --define
    options come after them, so they win.  ``-D NAME[=VALUE]`` is
    accepted as openvaf-r spells it."""
    import shlex
    va_path = None
    output_dir = '.'
    try:
        defines = shlex.split(os.environ.get('PYMS_DEFINES', ''))
    except ValueError as e:
        sys.exit(f"xyce_device_gen: error: PYMS_DEFINES={os.environ['PYMS_DEFINES']!r}: {e}")
    args = list(argv)
    while args:
        a = args.pop(0)
        if a == '--version':
            from vae import __version__
            print(f'PyMS {__version__}')
            sys.exit(0)
        elif a == '--output' and args:
            output_dir = args.pop(0)
        elif a.startswith('--output='):
            output_dir = a[len('--output='):]
        elif a in ('--define', '-D') and args:
            defines.append(args.pop(0))
        elif a.startswith('--define='):
            defines.append(a[len('--define='):])
        elif a.startswith('-D') and len(a) > 2:
            defines.append(a[2:])
        elif a.startswith('-') or va_path is not None:
            sys.stderr.write(f'xyce_device_gen.py: unknown argument {a!r}\n{_USAGE}\n')
            sys.exit(1)
        else:
            va_path = a
    if va_path is None:
        print(_USAGE)
        sys.exit(1)
    return va_path, output_dir, defines


def _record_defines(output_dir: str, module: str, defines: list) -> None:
    """The device-shell cache of N_DEV_PyMS.C reuses <output>/pyms_<module>.so
    on an exact .va identity, path + content hash, kept in the sidecar
    <so>.src; it does not see command-line macros, and the same .va built
    with other macros is another device.  Record the macros beside the
    shell and, when they differ from the record, remove the identity
    record so that N_DEV_PyMS.C rebuilds the shell.  With no macros and
    no record nothing is written.  (A shell cache whose slot name carries
    the identity hash must put the macros into that hash: BUGS.md.)"""
    rec = os.path.join(output_dir, f'pyms_{module}.so.defines')
    now = ''.join(d + '\n' for d in defines)
    old = ''
    if os.path.exists(rec):
        with open(rec) as f:
            old = f.read()
    if old == now:
        return
    src = os.path.join(output_dir, f'pyms_{module}.so.src')
    if os.path.exists(src):
        os.unlink(src)
    tmp = f'{rec}.{os.getpid()}.tmp'
    with open(tmp, 'w') as f:
        f.write(now)
    os.replace(tmp, rec)


if __name__ == '__main__':
    va_path, output_dir, defines = _parse_cli(sys.argv[1:])
    va_path = resolve_entry_point(va_path)

    from vae.parser import parse_file_all
    try:
        mod, others = parse_file_all(va_path, defines=defines)
    except (PreprocessError, ParseError) as e:
        # A source PyMS cannot read (an undefined macro, an attribute with
        # no statement after it, ...): one line, no traceback.
        sys.exit(f"xyce_device_gen: error: {e}")

    from vae.ginac_emitter import EmitError as _EmitError

    def _generate(m, secondary):
        header, impl = generate_device_cpp(m, '', va_path=va_path, defines=defines,
                                           module_arg=secondary,
                                           primary=mod.name if secondary else None)
        NAME = m.name.upper()
        h_path = os.path.join(output_dir, f'N_DEV_PYMS_{NAME}.h')
        c_path = os.path.join(output_dir, f'N_DEV_PYMS_{NAME}.C')
        with open(h_path, 'w') as f:
            f.write(header)
        with open(c_path, 'w') as f:
            f.write(impl)
        _record_defines(output_dir, m.name, defines)
        print(f"Generated {h_path} and {c_path}")
        print(f"Module: {m.name}, Ports: {[p.name for p in m.ports]}, "
              f"Internal: {m.internal_nodes}, Params: {len(m.params)}")

    try:
        _generate(mod, False)
    except (ValueError, ConstExprError, ParseError, _EmitError) as e:
        # (ParseError covers CurrentProbeError: a port current probe I(<p>)
        # ended in a Python traceback before 1.1.1.)
        # What PyMS cannot build into an Xyce device (DeviceGenError is a
        # ValueError; a parameter default PyMS cannot evaluate): one line naming the
        # construct (Xyce shows this output under ".HDL: PyMS compilation failed").
        sys.exit(f"xyce_device_gen: error: {e}")
    # Every other module of the file is a device of its own (1.1.4): a
    # netlist may name any of them (Xyce registers each "Module:" line; before,
    # only the file's first module was a device and a netlist naming another
    # stopped with an unknown device).  One PyMS cannot build is reported and
    # left out; the first module decides the exit status, as before.
    for m in others:
        try:
            _generate(m, True)
        except (ValueError, ConstExprError, ParseError, _EmitError) as e:
            print(f"xyce_device_gen: note: module {m.name} of {va_path} is not built "
                  f"as a device: {e}")
