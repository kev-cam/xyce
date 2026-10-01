library ieee; use ieee.std_logic_1164.all;
library sv2vhdl;
use sv2vhdl.logic3d_types_pkg.all;
use sv2vhdl.logic3da_pkg.all;

entity cosim_glitch is end entity;

architecture tb of cosim_glitch is
    signal q : resolved_logic3da;          -- D2A: drives analog node nin
begin
    -- a change that arrives while the previous 1ns D2A edge is still ramping:
    -- the driven voltage must reverse from where the ramp is (~0.5V), not jump
    process begin
        q <= L3DA_0; wait for 50 ns;
        q <= L3DA_1; wait for 500 ps;
        q <= L3DA_0; wait for 49.5 ns;
        q <= L3DA_1; wait;
    end process;
end architecture;
