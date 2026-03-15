-- 5-stage alternating analog/digital inverter chain testbench
-- Stages 1,3,5: analog (Xyce cmos_inv)
-- Stages 2,4: digital (NVC digital_inv)
library ieee;
use ieee.std_logic_1164.all;

library sv2vhdl;
use sv2vhdl.logic3d_types_pkg.all;
use sv2vhdl.logic3da_pkg.all;

entity cosim_tb is
end entity;

architecture test of cosim_tb is
    signal stim    : resolved_logic3da := L3DA_0;
    signal s1      : resolved_logic3da;  -- after analog stage 1
    signal s2      : resolved_logic3da;  -- after digital stage 2
    signal s3      : resolved_logic3da;  -- after analog stage 3
    signal s4      : resolved_logic3da;  -- after digital stage 4
    signal chain_out : resolved_logic3da;  -- after analog stage 5
begin

    -- Stage 1: analog inverter (Xyce)
    u1: entity work.cmos_inv
        port map (in_p => stim, out_p => s1);

    -- Stage 2: digital inverter (NVC)
    u2: entity work.digital_inv
        port map (inp => s1, outp => s2);

    -- Stage 3: analog inverter (Xyce)
    u3: entity work.cmos_inv
        port map (in_p => s2, out_p => s3);

    -- Stage 4: digital inverter (NVC)
    u4: entity work.digital_inv
        port map (inp => s3, outp => s4);

    -- Stage 5: analog inverter (Xyce)
    u5: entity work.cmos_inv
        port map (in_p => s4, out_p => chain_out);

    stimulus: process
    begin
        stim <= L3DA_0;
        wait for 20 ns;

        stim <= L3DA_1;
        wait for 20 ns;

        stim <= L3DA_0;
        wait for 20 ns;

        stim <= L3DA_1;
        wait for 20 ns;

        stim <= L3DA_0;
        wait for 20 ns;

        report "DONE";
        wait;
    end process;

    monitor: process(chain_out)
    begin
        report "chain_out: V=" & real'image(chain_out.voltage)
             & " R=" & real'image(chain_out.resistance);
    end process;

end architecture test;
