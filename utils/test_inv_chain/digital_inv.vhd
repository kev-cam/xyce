-- Digital inverter using logic3da voltage threshold
library ieee;
use ieee.std_logic_1164.all;

library sv2vhdl;
use sv2vhdl.logic3d_types_pkg.all;
use sv2vhdl.logic3da_pkg.all;

entity digital_inv is
    port (
        inp  : in  resolved_logic3da;
        outp : out resolved_logic3da
    );
end entity;

architecture rtl of digital_inv is
begin
    process(inp)
    begin
        if inp.voltage > 0.5 then
            outp <= L3DA_0;
        else
            outp <= L3DA_1;
        end if;
    end process;
end architecture;
