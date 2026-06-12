-- xspice_digital.vhd — VHDL equivalents of the SIMetrix/XSPICE digital code
-- models, for analog-on-top co-simulation with Xyce (Xyce drives the analog,
-- these run under nvc, the boundary is Xyce PWL "code:" hooks). Each entity
-- uses logic3da (Thevenin voltage/resistance) so the analog/digital boundary
-- is native -- inputs are thresholded on .voltage, outputs drive L3DA levels.
--
-- Covers the code models simetrix2xyce.pl catalogued from real SIMetrix decks:
--   logic:    d_inverter d_buffer d_nand d_or d_tff d_pullup d_pulldown
--   boundary: adc_bridge dac_bridge adc_schmitt
-- Model parameters (rise/fall delay, in_low/in_high, out_low/out_high) map to
-- generics, defaulted to the XSPICE/SIMetrix defaults.

library ieee;
use ieee.std_logic_1164.all;

library sv2vhdl;
use sv2vhdl.logic3d_types_pkg.all;
use sv2vhdl.logic3da_pkg.all;

-- ---------------------------------------------------------------- d_inverter
entity xsp_d_inverter is
    generic ( VTH : real := VDD/2.0;
              RISE_DELAY : time := 1 ns; FALL_DELAY : time := 1 ns );
    port ( i : in resolved_logic3da; o : out resolved_logic3da );
end entity;
architecture rtl of xsp_d_inverter is
begin
    process(i) begin
        if i.voltage > VTH then o <= L3DA_0 after FALL_DELAY;
        else                    o <= L3DA_1 after RISE_DELAY; end if;
    end process;
end architecture;

-- ------------------------------------------------------------------ d_buffer

library ieee;
use ieee.std_logic_1164.all;
library sv2vhdl;
use sv2vhdl.logic3d_types_pkg.all;
use sv2vhdl.logic3da_pkg.all;

entity xsp_d_buffer is
    generic ( VTH : real := VDD/2.0;
              RISE_DELAY : time := 1 ns; FALL_DELAY : time := 1 ns );
    port ( i : in resolved_logic3da; o : out resolved_logic3da );
end entity;
architecture rtl of xsp_d_buffer is
begin
    process(i) begin
        if i.voltage > VTH then o <= L3DA_1 after RISE_DELAY;
        else                    o <= L3DA_0 after FALL_DELAY; end if;
    end process;
end architecture;

-- -------------------------------------------------------------------- d_nand

library ieee;
use ieee.std_logic_1164.all;
library sv2vhdl;
use sv2vhdl.logic3d_types_pkg.all;
use sv2vhdl.logic3da_pkg.all;

entity xsp_d_nand is
    generic ( N : positive := 2; VTH : real := VDD/2.0; DELAY : time := 1 ns );
    port ( i : in logic3da_vector(0 to 7); n_used : integer := 2;
           o : out resolved_logic3da );
end entity;
architecture rtl of xsp_d_nand is
begin
    process(i) variable allhi : boolean; begin
        allhi := true;
        for k in 0 to n_used-1 loop
            if i(k).voltage <= VTH then allhi := false; end if;
        end loop;
        if allhi then o <= L3DA_0 after DELAY; else o <= L3DA_1 after DELAY; end if;
    end process;
end architecture;

-- ---------------------------------------------------------------------- d_or

library ieee;
use ieee.std_logic_1164.all;
library sv2vhdl;
use sv2vhdl.logic3d_types_pkg.all;
use sv2vhdl.logic3da_pkg.all;

entity xsp_d_or is
    generic ( VTH : real := VDD/2.0; DELAY : time := 1 ns );
    port ( i : in logic3da_vector(0 to 7); n_used : integer := 2;
           o : out resolved_logic3da );
end entity;
architecture rtl of xsp_d_or is
begin
    process(i) variable anyhi : boolean; begin
        anyhi := false;
        for k in 0 to n_used-1 loop
            if i(k).voltage > VTH then anyhi := true; end if;
        end loop;
        if anyhi then o <= L3DA_1 after DELAY; else o <= L3DA_0 after DELAY; end if;
    end process;
end architecture;

-- -------------------------------------------------------------------- d_tff
-- Toggle flip-flop: output toggles on the rising edge of the clock input.

library ieee;
use ieee.std_logic_1164.all;
library sv2vhdl;
use sv2vhdl.logic3d_types_pkg.all;
use sv2vhdl.logic3da_pkg.all;

entity xsp_d_tff is
    generic ( VTH : real := VDD/2.0; DELAY : time := 1 ns );
    port ( clk : in resolved_logic3da; q : out resolved_logic3da );
end entity;
architecture rtl of xsp_d_tff is
    signal state : std_ulogic := '0';
    signal prev  : boolean := false;
begin
    process(clk) variable hi : boolean; begin
        hi := clk.voltage > VTH;
        if hi and not prev then state <= not state; end if;
        prev <= hi;
    end process;
    q <= L3DA_1 after DELAY when state = '1' else L3DA_0 after DELAY;
end architecture;

-- ------------------------------------------------------ d_pullup / d_pulldown

library ieee;
use ieee.std_logic_1164.all;
library sv2vhdl;
use sv2vhdl.logic3d_types_pkg.all;
use sv2vhdl.logic3da_pkg.all;

entity xsp_d_pullup is
    port ( o : out resolved_logic3da );
end entity;
architecture rtl of xsp_d_pullup is begin o <= L3DA_PULL1; end architecture;


library ieee;
use ieee.std_logic_1164.all;
library sv2vhdl;
use sv2vhdl.logic3d_types_pkg.all;
use sv2vhdl.logic3da_pkg.all;

entity xsp_d_pulldown is
    port ( o : out resolved_logic3da );
end entity;
architecture rtl of xsp_d_pulldown is begin o <= L3DA_PULL0; end architecture;

-- ----------------------------------------------------------------- adc_bridge
-- Analog -> logic: threshold the analog input. Output is digital-strong.

library ieee;
use ieee.std_logic_1164.all;
library sv2vhdl;
use sv2vhdl.logic3d_types_pkg.all;
use sv2vhdl.logic3da_pkg.all;

entity xsp_adc_bridge is
    generic ( IN_LOW : real := 1.0; IN_HIGH : real := 2.0;
              RISE_DELAY : time := 0 ns; FALL_DELAY : time := 0 ns );
    port ( a : in resolved_logic3da; d : out resolved_logic3da );
end entity;
architecture rtl of xsp_adc_bridge is
    signal st : std_ulogic := '0';
begin
    process(a) begin
        if    a.voltage > IN_HIGH then st <= '1';
        elsif a.voltage < IN_LOW  then st <= '0'; end if;  -- hysteresis band
    end process;
    d <= L3DA_1 after RISE_DELAY when st = '1' else L3DA_0 after FALL_DELAY;
end architecture;

-- ----------------------------------------------------------------- dac_bridge
-- Logic -> analog: drive OUT_LOW/OUT_HIGH onto the analog node.

library ieee;
use ieee.std_logic_1164.all;
library sv2vhdl;
use sv2vhdl.logic3d_types_pkg.all;
use sv2vhdl.logic3da_pkg.all;

entity xsp_dac_bridge is
    generic ( OUT_LOW : real := 0.0; OUT_HIGH : real := 5.0; VTH : real := VDD/2.0;
              T_RISE : time := 2 ns; T_FALL : time := 2 ns );
    port ( d : in resolved_logic3da; a : out resolved_logic3da );
end entity;
architecture rtl of xsp_dac_bridge is begin
    process(d) begin
        if d.voltage > VTH then
            a <= (voltage => OUT_HIGH, resistance => R_STRONG, flags => AFL_KNOWN) after T_RISE;
        else
            a <= (voltage => OUT_LOW,  resistance => R_STRONG, flags => AFL_KNOWN) after T_FALL;
        end if;
    end process;
end architecture;

-- ---------------------------------------------------------------- adc_schmitt
-- Schmitt-trigger analog->logic: hysteretic threshold (in_low/in_high).

library ieee;
use ieee.std_logic_1164.all;
library sv2vhdl;
use sv2vhdl.logic3d_types_pkg.all;
use sv2vhdl.logic3da_pkg.all;

entity xsp_adc_schmitt is
    generic ( IN_LOW : real := 1.0; IN_HIGH : real := 3.0 );
    port ( a : in resolved_logic3da; d : out resolved_logic3da );
end entity;
architecture rtl of xsp_adc_schmitt is
    signal st : std_ulogic := '0';
begin
    process(a) begin
        if    a.voltage > IN_HIGH then st <= '1';
        elsif a.voltage < IN_LOW  then st <= '0'; end if;
    end process;
    d <= L3DA_1 when st = '1' else L3DA_0;
end architecture;
