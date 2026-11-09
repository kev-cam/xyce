# VAE — Verilog-A Extension for Xyce
"""PyMS: the Verilog-A compiler behind Xyce's ``.hdl`` line
(``vae/xyce_device_gen.py``, run by N_DEV_PyMS.C).

``__version__`` is PyMS's version marker.  The same string sits in the
``VERSION`` file beside ``vae/`` and ``xyce_device_gen.py --version``
prints it, so a layer that depends on PyMS's behaviour (vamos's spectre
personality) can require a minimum version instead of guarding every
Verilog-A form itself.  Bump it with every change in behaviour.
"""

__version__ = '1.1.8'
