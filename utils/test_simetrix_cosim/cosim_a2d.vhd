library ieee; use ieee.std_logic_1164.all;
library sv2vhdl;
use sv2vhdl.logic3d_types_pkg.all;
use sv2vhdl.logic3da_pkg.all;

entity cosim_a2d is end entity;

architecture tb of cosim_a2d is
    signal ain : resolved_logic3da;    -- A2D: sampled from Xyce node nsense
    signal dq  : resolved_logic3da;    -- D2A: drives Xyce node nout
begin
    -- threshold the analog sample at 0.5V -> digital decision -> back to analog
    process(ain) begin
        if ain.voltage > 0.5 then dq <= L3DA_1;
        else                      dq <= L3DA_0; end if;
    end process;
end architecture;
