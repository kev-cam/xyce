-- Auto-generated VHDL wrapper for Xyce subcircuit: cmos_inv
-- Source ports: IN OUT VDD VSS
-- Analog ports use logic3da (Thevenin V,R) for PWL bridge interface

library ieee;
use ieee.std_logic_1164.all;

library sv2vhdl;
use sv2vhdl.spice_subckt_pkg.all;
use sv2vhdl.logic3da_pkg.all;

entity cmos_inv is
    port (
        in_p : in resolved_logic3da;
        out_p : out resolved_logic3da
    );
end entity cmos_inv;

architecture spice of cmos_inv is

    -- Inherited power ports
    attribute spice_supply : real;
    signal vdd : resolved_logic3da;
    attribute spice_supply of vdd : signal is 1.0;
    signal vss : resolved_logic3da;
    attribute spice_supply of vss : signal is 0.0;

    -- Bridge signals for SPICE PWL interface
    attribute spice_bridge : string;
    signal in_p_rcv : logic3da;
    attribute spice_bridge of in_p_rcv : signal is "receiver:IN";
    signal out_p_drv : logic3da;
    attribute spice_bridge of out_p_drv : signal is "driver:OUT";

begin

    -- Receiver: SPICE reads input value from digital side
    in_p_rcv <= in_p'receiver;
    -- Driver: SPICE drives output value to digital side
    out_p'driver <= out_p_drv;

    spice_subckt(
        "DIALECT:xyce|SUBCKT:cmos_inv|PORT:IN:in_p:in|PORT:OUT:out_p:out|PORT:VDD:vdd:power|PORT:VSS:vss:power||.SUBCKT cmos_inv IN OUT VDD VSS~NL~" &
        "M1 OUT IN VDD VDD PMOS W=2u L=0.18u~NL~" &
        "M2 OUT IN VSS VSS NMOS W=1u L=0.18u~NL~" &
        ".ENDS cmos_inv~NL~" &
        "* CMOS inverter - just the transistors~NL~" &
        "* Ports: IN OUT VDD VSS~NL~" &
        ".MODEL NMOS NMOS (LEVEL=1 VTO=0.4 KP=120u GAMMA=0.4 PHI=0.65)~NL~" &
        ".MODEL PMOS PMOS (LEVEL=1 VTO=-0.4 KP=60u GAMMA=0.4 PHI=0.65)"
    );

end architecture spice;
