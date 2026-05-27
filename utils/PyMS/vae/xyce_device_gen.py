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
    python3 xyce_device_gen.py input.va [--params W=1u,L=1u] [--output dir/]
"""

import os
import sys
import subprocess
from pathlib import Path

# Add parent for vae imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from vae.parser import parse_verilog_a, parse_file, NodeKind, ContribKind


def scan_va_attributes(va_path: str) -> dict:
    """Extract ``(* key="value" ... *)`` attributes from a Verilog-A
    module declaration.

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
        text = preprocess_file(va_path)
    except Exception:
        # Fall back to raw source — better to find *some* attrs than
        # to fail outright if preprocess hits something unexpected.
        with open(va_path, 'r', errors='replace') as f:
            text = f.read()

    # Candidate windows: each is a ``(* ... *)`` block adjacent to a
    # ``module`` keyword (either side). The block can span multiple
    # lines and may contain comma- or space-separated key=value pairs.
    candidates = []
    # Post-decl: ``module name(ports) (* ... *)``
    for m in re.finditer(
            r'module\s+\w+\s*\([^)]*\)\s*\(\*(.*?)\*\)', text, re.DOTALL):
        candidates.append(m.group(1))
    # Pre-decl: ``(* ... *) module name(ports)``
    for m in re.finditer(
            r'\(\*(.*?)\*\)\s*module\s+\w+\s*\(', text, re.DOTALL):
        candidates.append(m.group(1))

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

    # No xyce* attrs found — return whatever the first generic
    # candidate yielded (still better than {}).
    if candidates:
        for am in re.finditer(r'(\w+)\s*=\s*"([^"]*)"', candidates[0]):
            attrs[am.group(1)] = am.group(2)
    return attrs


def generate_device_cpp(mod, xyce_src_dir: str, va_path: str = '') -> tuple[str, str]:
    """Generate Xyce device C++ source (.h and .C) from parsed VA module.

    Returns (header_source, impl_source) strings.
    """
    name = mod.name  # e.g. 'rlc', 'PSP103_VA'
    NAME = name.upper()

    # Extract Xyce ADMS-style attributes from the VA source
    va_attrs = scan_va_attributes(va_path)
    model_group = va_attrs.get('xyceModelGroup', '')  # e.g. "MOSFET", "Diode"
    level_number = va_attrs.get('xyceLevelNumber', '')  # e.g. "77"
    type_variable = va_attrs.get('xyceTypeVariable', '')  # e.g. "TYPE"

    ports = [(p.name, p.direction.name) for p in mod.ports]
    internals = list(mod.internal_nodes)
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
        num_nodes_required = min(_GROUP_REQUIRED_TERMINALS[model_group], n_ext)
        num_nodes_optional = max(0, n_ext - num_nodes_required)
    else:
        # Y-devices and unknown groups: every declared port is required,
        # zero optional — matches what Xyce's generic Y handler expects.
        num_nodes_required = n_ext
        num_nodes_optional = 0

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
    h.append('#include <dlfcn.h>')
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
    h.append('')

    # VAE function pointer types
    h.append('struct VaeState { double V[16]; double Vt; };')
    h.append('typedef void (*VaeEvalFn)(VaeState*, double*, double*);')
    h.append('')

    # Traits
    h.append('class Model;')
    h.append('class Instance;')
    h.append('')
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
    h.append('  bool updateIntermediateVars();')
    h.append('  bool updatePrimaryState();')
    h.append('  bool loadDAEFVector();')
    h.append('  bool loadDAEQVector();')
    h.append('  bool loadDAEdFdx();')
    h.append('  bool loadDAEdQdx();')
    h.append('  void loadNodeSymbols(Util::SymbolTable &symbol_table) const;')
    h.append('')
    h.append('  Model &getModel() { return model_; }')
    h.append('')
    h.append('private:')
    h.append('  Model &model_;')
    # Instance parameters
    for pname, pdefault, is_inst, _cxx in params:
        if is_inst:
            h.append(f'  double {_cxx};')
    h.append('')
    # LID storage
    for n in all_nodes:
        h.append(f'  int li_{n};')
    h.append('')
    # Jacobian offsets
    h.append(f'  std::vector<std::vector<int>> jacStamp_;')
    h.append(f'  std::vector<std::vector<int>> jacLIDs_;')
    h.append('')
    # VAE state
    h.append('  // VAE compiled model')
    h.append('  void *vae_dl_;')
    h.append('  VaeEvalFn vae_eval_;')
    h.append('  VaeEvalFn vae_jac_;')
    h.append(f'  double prevV_[{n_total}];')
    h.append('  bool hasPrevV_;')
    h.append(f'  double F_[{n_total}];')
    h.append(f'  double Q_[{n_total}];')
    h.append(f'  double dFdx_[{n_total}][{n_total}];')
    h.append(f'  double dQdx_[{n_total}][{n_total}];')
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
    # Model parameters
    for pname, pdefault, is_inst, _cxx in params:
        if not is_inst:
            h.append(f'  double {_cxx};')
    h.append('  std::string vaFile_;')
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
    c.append('#include <N_LAS_Vector.h>')
    c.append('#include <N_LAS_Matrix.h>')
    c.append('#include <cstring>')
    c.append('#include <cmath>')
    # See matching block in the .h: Xyce_config.h pulls in macro
    # definitions that collide with .va parameter names.
    for offender in ('VERSION', 'PACKAGE_VERSION', 'PACKAGE_NAME',
                     'PACKAGE_STRING', 'MAJOR', 'MINOR'):
        c.append(f'#ifdef {offender}\n#undef {offender}\n#endif')
    c.append('#include <cstdlib>')
    c.append('')
    c.append(f'namespace Xyce {{ namespace Device {{ namespace PYMS_{NAME} {{')
    c.append('')

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

    # Parameter registration
    c.append('void Traits::loadInstanceParameters(ParametricData<Instance> &p) {')
    for pname, pdefault, is_inst, _cxx in params:
        if is_inst:
            try:
                dval = float(pdefault)
            except:
                dval = 0.0
            c.append(f'  p.addPar("{pname.upper()}", {dval}, &Instance::{_cxx})')
            c.append(f'    .setUnit(U_NONE)')
            c.append(f'    .setDescription("{pname}");')
    c.append('}')
    c.append('')

    c.append('void Traits::loadModelParameters(ParametricData<Model> &p) {')
    for pname, pdefault, is_inst, _cxx in params:
        if not is_inst:
            try:
                dval = float(pdefault)
            except:
                dval = 0.0
            c.append(f'  p.addPar("{pname.upper()}", {dval}, &Model::{_cxx})')
            c.append(f'    .setUnit(U_NONE)')
            c.append(f'    .setDescription("{pname}");')
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
    c.append(f'  numIntVars = {n_int};')
    c.append(f'  numStateVars = 0;')
    c.append(f'  numStoreVars = 0;')
    c.append('')
    c.append(f'  jacStamp_.resize(numNodes);')
    c.append(f'  for (int i = 0; i < numNodes; i++) {{')
    c.append(f'    jacStamp_[i].resize(numNodes);')
    c.append(f'    for (int j = 0; j < numNodes; j++)')
    c.append(f'      jacStamp_[i][j] = j;')
    c.append(f'  }}')
    c.append('')
    c.append('  // Set default instance param values')
    for pname, pdefault, is_inst, _cxx in params:
        if is_inst:
            try:
                dval = float(pdefault)
            except:
                dval = 0.0
            c.append(f'  {_cxx} = {dval};')
    c.append('')
    c.append('  processParams();')
    c.append('}')
    c.append('')

    c.append('Instance::~Instance() {')
    c.append('  if (vae_dl_) dlclose(vae_dl_);')
    c.append('}')
    c.append('')

    c.append('bool Instance::processParams() {')
    c.append('  if (!vae_eval_) {')
    c.append('    // Try VAE_SO_PATH (single .so) or VAE_SO_DIR/<modelname>.so')
    c.append('    std::string so_path;')
    c.append('    const char *so = getenv("VAE_SO_PATH");')
    c.append('    const char *dir = getenv("VAE_SO_DIR");')
    c.append('    if (so) {')
    c.append('      so_path = so;')
    c.append('    } else if (dir) {')
    c.append('      so_path = std::string(dir) + "/" + model_.getName() + ".so";')
    c.append('    }')
    c.append('    if (!so_path.empty()) {')
    c.append('      vae_dl_ = dlopen(so_path.c_str(), RTLD_NOW);')
    c.append('      if (vae_dl_) {')
    c.append('        vae_eval_ = (VaeEvalFn)dlsym(vae_dl_, "vae_eval");')
    c.append('        vae_jac_ = (VaeEvalFn)dlsym(vae_dl_, "vae_jacobian");')
    c.append('      }')
    c.append('    }')
    c.append('  }')
    c.append('  return true;')
    c.append('}')
    c.append('')

    # Register LIDs
    c.append('void Instance::registerLIDs(const std::vector<int> &intLIDVec,')
    c.append('                            const std::vector<int> &extLIDVec) {')
    c.append('  DeviceInstance::registerLIDs(intLIDVec, extLIDVec);')
    for i, n in enumerate(all_nodes):
        if i < n_ext:
            c.append(f'  li_{n} = extLIDVec[{i}];')
        else:
            c.append(f'  li_{n} = intLIDVec[{i - n_ext}];')
    c.append('}')
    c.append('')

    c.append('void Instance::registerStateLIDs(const std::vector<int> &v) {}')
    c.append('void Instance::registerStoreLIDs(const std::vector<int> &v) {}')
    c.append('std::map<int,std::string> &Instance::getIntNameMap() {')
    c.append('  static std::map<int,std::string> m;')
    for i, n in enumerate(internals):
        c.append(f'  m[{i}] = "{n}";')
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

    for ck, cbranch, cexpr in contribs:
        key = (ck, cbranch)
        if key not in seen_branches:
            a = cbranch[0] if len(cbranch) >= 1 else None
            b = cbranch[1] if len(cbranch) >= 2 else None
            a_idx = resolve_node(a)
            b_idx = resolve_node(b)
            seen_branches[key] = len(branch_map)
            branch_map.append((a_idx, b_idx))
    n_branches = len(branch_map)

    # updateIntermediateVars — call VAE .so, get branch-indexed F/Q
    c.append('bool Instance::updateIntermediateVars() {')
    c.append('  Linear::Vector *solVec = extData.nextSolVectorPtr;')
    c.append(f'  memset(F_, 0, sizeof(F_));')
    c.append(f'  memset(Q_, 0, sizeof(Q_));')
    c.append(f'  memset(dFdx_, 0, sizeof(dFdx_));')
    c.append(f'  memset(dQdx_, 0, sizeof(dQdx_));')
    c.append('')
    c.append('  if (vae_eval_ && vae_jac_) {')
    c.append('    VaeState state = {};')
    for i, n in enumerate(all_nodes):
        c.append(f'    state.V[{i}] = (*solVec)[li_{n}];')
    c.append('    state.Vt = 8.617087e-5 * 300.15;')
    c.append('')
    # Check if VAE .so uses node-indexed or branch-indexed output.
    # We generate support for both: try node-indexed first (n_nodes output),
    # fall back to branch-indexed (n_branches output) for ADMS-style .so files.
    c.append(f'    // Call VAE .so — supports both node-indexed and branch-indexed output')
    c.append(f'    int so_n_branches = {n_branches};')
    c.append(f'    int so_n_nodes = {n_total};')
    c.append(f'    // Try to query .so for its output size')
    c.append(f'    typedef int (*IntFn)();')
    c.append(f'    IntFn nb_fn = vae_dl_ ? (IntFn)dlsym(vae_dl_, "vae_n_branches") : 0;')
    c.append(f'    IntFn nn_fn = vae_dl_ ? (IntFn)dlsym(vae_dl_, "vae_n_nodes") : 0;')
    c.append(f'    if (nb_fn) so_n_branches = nb_fn();')
    c.append(f'    if (nn_fn) so_n_nodes = nn_fn();')
    c.append(f'    bool node_indexed = (so_n_branches == so_n_nodes);')
    c.append('')
    c.append(f'    double Fb[{max(n_branches, n_total)}]={{}}, Qb[{max(n_branches, n_total)}]={{}};')
    c.append(f'    vae_eval_(&state, Fb, Qb);')
    c.append('')
    c.append('    // Sanitize NaN/Inf')
    c.append(f'    for (int i = 0; i < {max(n_branches, n_total)}; i++) {{')
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
    c.append(f'      // Branch-indexed: map branches to KCL node contributions')
    for bi, (a_idx, b_idx) in enumerate(branch_map):
        if a_idx >= 0:
            c.append(f'      F_[{a_idx}] += Fb[{bi}];')
        if b_idx >= 0:
            c.append(f'      F_[{b_idx}] -= Fb[{bi}];')
        if a_idx >= 0:
            c.append(f'      Q_[{a_idx}] += Qb[{bi}];')
        if b_idx >= 0:
            c.append(f'      Q_[{b_idx}] -= Qb[{bi}];')
    c.append(f'      double dFb[{max(n_branches*n_total,1)}]={{}}, dQb[{max(n_branches*n_total,1)}]={{}};')
    c.append(f'      vae_jac_(&state, dFb, dQb);')
    for bi, (a_idx, b_idx) in enumerate(branch_map):
        for j in range(n_total):
            if a_idx >= 0:
                c.append(f'      dFdx_[{a_idx}][{j}] += dFb[{bi}*{n_total}+{j}];')
                c.append(f'      dQdx_[{a_idx}][{j}] += dQb[{bi}*{n_total}+{j}];')
            if b_idx >= 0:
                c.append(f'      dFdx_[{b_idx}][{j}] -= dFb[{bi}*{n_total}+{j}];')
                c.append(f'      dQdx_[{b_idx}][{j}] -= dQb[{bi}*{n_total}+{j}];')
    c.append(f'    }}')
    c.append('  }')  # close if (vae_eval_ && vae_jac_)
    c.append('')
    c.append('  // Add gmin conductance to prevent singular matrix')
    c.append('  const double gmin = 1e-12;  // large gmin for initial convergence')
    c.append('  Linear::Vector *solVec2 = extData.nextSolVectorPtr;')
    for i, n in enumerate(all_nodes):
        c.append(f'  F_[{i}] += gmin * (*solVec2)[li_{n}];')
        c.append(f'  dFdx_[{i}][{i}] += gmin;')
    c.append('')
    c.append('  return true;')
    c.append('}')
    c.append('')

    c.append('bool Instance::updatePrimaryState() {')
    c.append('  return updateIntermediateVars();')
    c.append('}')
    c.append('')

    # DAE loading — stamp node-indexed contributions into Xyce matrix
    c.append('bool Instance::loadDAEFVector() {')
    c.append('  Linear::Vector &fVec = *(extData.daeFVectorPtr);')
    for i, n in enumerate(all_nodes):
        c.append(f'  fVec[li_{n}] += F_[{i}];')
    c.append('  return true;')
    c.append('}')
    c.append('')

    c.append('bool Instance::loadDAEQVector() {')
    c.append('  Linear::Vector &qVec = *(extData.daeQVectorPtr);')
    for i, n in enumerate(all_nodes):
        c.append(f'  qVec[li_{n}] += Q_[{i}];')
    c.append('  return true;')
    c.append('}')
    c.append('')

    c.append('bool Instance::loadDAEdFdx() {')
    c.append('  Linear::Matrix &dFdx = *(extData.dFdxMatrixPtr);')
    for i, n in enumerate(all_nodes):
        for j in range(n_total):
            c.append(f'  dFdx[li_{n}][jacLIDs_[{i}][{j}]] += dFdx_[{i}][{j}];')
    c.append('  return true;')
    c.append('}')
    c.append('')

    c.append('bool Instance::loadDAEdQdx() {')
    c.append('  Linear::Matrix &dQdx = *(extData.dQdxMatrixPtr);')
    for i, n in enumerate(all_nodes):
        for j in range(n_total):
            c.append(f'  dQdx[li_{n}][jacLIDs_[{i}][{j}]] += dQdx_[{i}][{j}];')
    c.append('  return true;')
    c.append('}')
    c.append('')

    # Model implementation
    c.append('Model::Model(const Configuration &config, const ModelBlock &mb,')
    c.append('             const FactoryBlock &fb)')
    c.append('  : DeviceModel(mb, config.getModelParameters(), fb)')
    c.append('{')
    for pname, pdefault, is_inst, _cxx in params:
        if not is_inst:
            try:
                dval = float(pdefault)
            except:
                dval = 0.0
            c.append(f'  {_cxx} = {dval};')
    c.append('  processParams();')
    c.append('}')
    c.append('')
    c.append('Model::~Model() {}')
    c.append('bool Model::processParams() { return true; }')
    c.append('bool Model::processInstanceParams() { return true; }')
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


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: xyce_device_gen.py <input.va> [--output <dir>]")
        sys.exit(1)

    va_path = resolve_entry_point(sys.argv[1])
    output_dir = '.'
    if '--output' in sys.argv:
        output_dir = sys.argv[sys.argv.index('--output') + 1]

    mod = parse_file(va_path)
    NAME = mod.name.upper()

    header, impl = generate_device_cpp(mod, '', va_path=va_path)

    h_path = os.path.join(output_dir, f'N_DEV_PYMS_{NAME}.h')
    c_path = os.path.join(output_dir, f'N_DEV_PYMS_{NAME}.C')

    with open(h_path, 'w') as f:
        f.write(header)
    with open(c_path, 'w') as f:
        f.write(impl)

    print(f"Generated {h_path} and {c_path}")
    print(f"Module: {mod.name}, Ports: {[p.name for p in mod.ports]}, "
          f"Internal: {mod.internal_nodes}, Params: {len(mod.params)}")
