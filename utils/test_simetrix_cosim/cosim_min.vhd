library ieee; use ieee.std_logic_1164.all;
library sv2vhdl;
use sv2vhdl.logic3d_types_pkg.all;
use sv2vhdl.logic3da_pkg.all;

entity cosim_min is end entity;

architecture tb of cosim_min is
    signal q : resolved_logic3da;          -- D2A: drives analog node nin
begin
    -- 100ns-period square wave
    process begin
        q <= L3DA_0; wait for 50 ns;
        q <= L3DA_1; wait for 50 ns;
    end process;
end architecture;
