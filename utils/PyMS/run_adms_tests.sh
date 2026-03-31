#!/bin/bash
#
# Run ADMS example benchmarks through Xyce, comparing ADMS vs PyMS paths.
#
# ADMS path: -adms flag (ignore .hdl, use compiled-in models)
# PyMS path: .hdl auto-compile via cadence2xyce preprocessor
#
# Usage:
#   ./run_adms_tests.sh            # run both
#   ./run_adms_tests.sh --adms     # ADMS only
#   ./run_adms_tests.sh --pyms     # PyMS only
#

set -uo pipefail

XYCE="${XYCE:-/usr/local/src/Xyce-8/xyce-build/src/Xyce}"
ADMS_DIR="/usr/local/src/Xyce-8/xyce/utils/ADMS/examples"
CADENCE2XYCE="/usr/local/src/Xyce-8/xyce/utils/cadence2xyce.pl"
WORK="/tmp/adms_vs_pyms"
TIMEOUT=120

export LD_LIBRARY_PATH="/usr/local/src/Xyce-8/xyce-build/src:${LD_LIBRARY_PATH:-}"

run_adms=1
run_pyms=1
case "${1:-both}" in
    --adms) run_pyms=0 ;;
    --pyms) run_adms=0 ;;
esac

rm -rf "$WORK"
mkdir -p "$WORK"

# ── Prepare a test netlist for Xyce ──────────────────────────────────

prepare_test() {
    # Copy all files to testdir, run cadence2xyce preprocessor.
    # The preprocessor handles: X→M/Y, dc=→dc, .probe→comment,
    # .model moduletype → Xyce type+level, and modifies included modelcards.
    local sp="$1" testdir="$2" dir="$3"

    # Copy the .sp and all dependencies to testdir
    cp "$sp" "$testdir/"
    for dep in "$dir"/modelcard* "$dir"/*.lib "$dir"/*.inc; do
        [[ -f "$dep" ]] && cp "$dep" "$testdir/" 2>/dev/null
    done
    for vadir in "$dir/../code" "$dir/../../code" "$dir/.."; do
        [[ -d "$vadir" ]] && find "$vadir" -maxdepth 1 \( -name "*.va" -o -name "*.include" \) -exec cp {} "$testdir/" \; 2>/dev/null
    done

    # Run preprocessor on the copy (modifies modelcards in testdir)
    local name
    name=$(basename "$sp" .sp)
    local sp_copy="$testdir/${name}.sp"
    local cir="$testdir/${name}.cir"
    perl "$CADENCE2XYCE" "$sp_copy" -o "$cir" 2>/dev/null

    # Fix .print — remove Cadence expressions
    if grep -qi '^\s*\.print.*{' "$cir" 2>/dev/null; then
        sed -i '/^\s*\.print/s/{[^}]*}//g' "$cir"
    fi
    if grep -qi '^\s*\.print\s*\(dc\|tran\|ac\)\s*$' "$cir" 2>/dev/null; then
        sed -i 's/^\(\.print\s*\S\+\)\s*$/\1 V(*)/I' "$cir"
    fi

    echo "$cir"
}

# ── Run one test ─────────────────────────────────────────────────────

run_test() {
    local cir="$1" mode="$2" logfile="$3"
    local flags=""
    [[ "$mode" == "adms" ]] && flags="-adms"

    local dir
    dir=$(dirname "$cir")
    cd "$dir"
    rm -rf /tmp/pyms_hdl_cache
    timeout "$TIMEOUT" "$XYCE" $flags "$(basename "$cir")" > "$logfile" 2>&1
    local rc=$?
    cd /usr/local/src

    if [[ $rc -eq 0 ]] && grep -q "End of Xyce" "$logfile"; then
        return 0
    fi
    return 1
}

# ── Main ─────────────────────────────────────────────────────────────

echo "═══════════════════════════════════════════════════════════"
echo "  ADMS vs PyMS benchmark — $(date)"
echo "═══════════════════════════════════════════════════════════"
echo ""

mapfile -t sp_files < <(find "$ADMS_DIR" -name "*.sp" -not -path "*/non-functional/*" | sort)
total=${#sp_files[@]}

adms_ok=0 adms_fail=0
pyms_ok=0 pyms_fail=0

REPORT="$WORK/report.csv"
echo "test,adms_status,adms_steps,adms_time,pyms_status,pyms_steps,pyms_time" > "$REPORT"

printf "%-55s  %-12s  %-12s\n" "Test" "ADMS" "PyMS"
printf "%-55s  %-12s  %-12s\n" "----" "----" "----"

for sp in "${sp_files[@]}"; do
    rel="${sp#$ADMS_DIR/}"
    name=$(basename "$sp" .sp)
    dir=$(dirname "$sp")

    adms_result="-" adms_steps=0 adms_time="-"
    pyms_result="-" pyms_steps=0 pyms_time="-"

    # ── ADMS ──
    if (( run_adms )); then
        testdir="$WORK/adms/$(echo "$rel" | tr '/' '_' | sed 's/\.sp$//')"
        mkdir -p "$testdir"
        cir=$(prepare_test "$sp" "$testdir" "$dir")
        if [[ -f "$cir" ]]; then
            t_start=$(date +%s%N)
            if run_test "$cir" "adms" "$testdir/run.log"; then
                t_end=$(date +%s%N)
                adms_time=$(echo "scale=3; ($t_end - $t_start) / 1000000000" | bc)
                adms_steps=$(grep "Number Successful Steps" "$testdir/run.log" 2>/dev/null | awk '{print $NF}')
                adms_result="OK"
                (( adms_ok++ ))
            else
                t_end=$(date +%s%N)
                adms_time=$(echo "scale=3; ($t_end - $t_start) / 1000000000" | bc)
                adms_result="FAIL"
                (( adms_fail++ ))
            fi
        else
            adms_result="SKIP"
        fi
    fi

    # ── PyMS ──
    if (( run_pyms )); then
        testdir="$WORK/pyms/$(echo "$rel" | tr '/' '_' | sed 's/\.sp$//')"
        mkdir -p "$testdir"
        cir=$(prepare_test "$sp" "$testdir" "$dir")
        if [[ -f "$cir" ]]; then
            t_start=$(date +%s%N)
            if run_test "$cir" "pyms" "$testdir/run.log"; then
                t_end=$(date +%s%N)
                pyms_time=$(echo "scale=3; ($t_end - $t_start) / 1000000000" | bc)
                pyms_steps=$(grep "Number Successful Steps" "$testdir/run.log" 2>/dev/null | awk '{print $NF}')
                pyms_result="OK"
                (( pyms_ok++ ))
            else
                t_end=$(date +%s%N)
                pyms_time=$(echo "scale=3; ($t_end - $t_start) / 1000000000" | bc)
                pyms_result="FAIL"
                (( pyms_fail++ ))
            fi
        else
            pyms_result="SKIP"
        fi
    fi

    printf "%-55s  %-12s  %-12s\n" "$rel" \
        "$([ "$adms_result" = OK ] && echo "OK ${adms_steps}st ${adms_time}s" || echo "$adms_result")" \
        "$([ "$pyms_result" = OK ] && echo "OK ${pyms_steps}st ${pyms_time}s" || echo "$pyms_result")"

    echo "$rel,$adms_result,$adms_steps,$adms_time,$pyms_result,$pyms_steps,$pyms_time" >> "$REPORT"
done

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Total: $total tests"
(( run_adms )) && echo "  ADMS:  $adms_ok OK / $adms_fail FAIL"
(( run_pyms )) && echo "  PyMS:  $pyms_ok OK / $pyms_fail FAIL"
echo ""
echo "  Report: $REPORT"
echo "  Logs:   $WORK/{adms,pyms}/<test>/run.log"
echo "═══════════════════════════════════════════════════════════"
