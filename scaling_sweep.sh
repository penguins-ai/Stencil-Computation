#!/usr/bin/env bash
# Runs the sweeps needed for the M2 report, now capturing BOTH the naive
# (static) and fixed (dynamic,1) scheduling policies so the "before vs
# after" comparison is automatic, plus optional taskset-pinned P-core
# reference points.
#
#   1. Strong scaling: fixed N, varying thread count, for EACH schedule
#      policy in SCHEDULES below -> strong_scaling.csv (now has a
#      `schedule` column).
#   2. Throughput vs N: fixed at the single best (schedule, threads)
#      combo found in step 1 -> throughput_vs_n.csv.
#   3. (Optional) taskset-pinned P-core-only reference points, if you set
#      PCORE_CPULIST below to your machine's P-core logical CPU ids
#      (comma-separated, as accepted by `taskset -c`). Leave empty to
#      skip -- this is machine-specific and won't be portable to a
#      grading machine, so treat it as a diagnostic, not part of the
#      "official" result.
#
# Each (schedule, N, threads) triple is run REPEATS times; the minimum
# elapsed time is kept, matching the M1 "best trial per N" convention.
#
# Usage: ./scaling_sweep.sh

set -e

REPEATS=3
STEPS=1000
MAXT=$(nproc)
SCHEDULES=("static" "dynamic,1")

# Set this to your P-core logical CPU ids (from `lscpu -e`, the ones with
# the highest MAXMHZ) to also capture taskset-pinned reference points,
# e.g. PCORE_CPULIST="0,1,2,3". Leave empty to skip.
PCORE_CPULIST=""

gcc -O2 -o heat2d_serial heat2d_serial.c -lm
gcc -O2 -fopenmp -o heat2d_omp heat2d_omp.c -lm

# Build the thread list: 1, 2, 4, 8, ... up to MAXT, plus MAXT itself if not
# already a power of two in the list.
THREADS=()
t=1
while [ "$t" -le "$MAXT" ]; do
    THREADS+=("$t")
    t=$((t * 2))
done
if [ "${THREADS[-1]}" -ne "$MAXT" ]; then
    THREADS+=("$MAXT")
fi

run_min_time () {
    # $1 = binary, $2 = N, $3 = steps, $4 = threads (or "" for serial),
    # $5 = OMP_SCHEDULE value (or "" to leave unset / use program default),
    # $6 = taskset cpulist (or "" for no pinning)
    local bin=$1 N=$2 steps=$3 threads=$4 sched=$5 cpulist=$6
    local best=""
    for ((r = 0; r < REPEATS; r++)); do
        local cmd=()
        if [ -n "$cpulist" ]; then
            cmd=(taskset -c "$cpulist")
        fi
        cmd+=("./$bin" "$N" "$steps")
        if [ -n "$threads" ]; then
            cmd+=("$threads")
        fi

        if [ -n "$sched" ]; then
            out=$(OMP_SCHEDULE="$sched" "${cmd[@]}")
        else
            out=$("${cmd[@]}")
        fi

        t=$(echo "$out" | grep "Elapsed time" | awk '{print $3}')
        if [ -z "$best" ] || awk -v a="$t" -v b="$best" 'BEGIN{exit !(a<b)}'; then
            best=$t
        fi
    done
    echo "$best"
}

# ---------- Sweep 1: Strong scaling at fixed N, across schedules ----------
N_FIXED=1024
echo "schedule,N,threads,elapsed_s,speedup,efficiency" > strong_scaling.csv

echo "Serial baseline at N=$N_FIXED (shared reference for all schedules)..."
T_SERIAL=$(run_min_time heat2d_serial "$N_FIXED" "$STEPS" "" "" "")
echo "serial,$N_FIXED,1(serial),$T_SERIAL,1.0000,1.0000" >> strong_scaling.csv

BEST_T=""
BEST_SCHED=""
BEST_ELAPSED=""

for sched in "${SCHEDULES[@]}"; do
    sched_label=$(echo "$sched" | tr ',' '-')
    for t in "${THREADS[@]}"; do
        echo "OpenMP N=$N_FIXED threads=$t schedule=$sched..."
        T=$(run_min_time heat2d_omp "$N_FIXED" "$STEPS" "$t" "$sched" "")
        speedup=$(awk -v s="$T_SERIAL" -v p="$T" 'BEGIN{printf "%.4f", s/p}')
        eff=$(awk -v sp="$speedup" -v t="$t" 'BEGIN{printf "%.4f", sp/t}')
        echo "$sched_label,$N_FIXED,$t,$T,$speedup,$eff" >> strong_scaling.csv

        if [ -z "$BEST_ELAPSED" ] || awk -v a="$T" -v b="$BEST_ELAPSED" 'BEGIN{exit !(a<b)}'; then
            BEST_ELAPSED=$T
            BEST_T=$t
            BEST_SCHED="$sched"
        fi
    done
done

# ---------- Optional: taskset-pinned P-core reference points ----------
if [ -n "$PCORE_CPULIST" ]; then
    echo "Taskset-pinned P-core reference (cpulist=$PCORE_CPULIST)..."
    NCPUS=$(echo "$PCORE_CPULIST" | awk -F',' '{print NF}')
    for sched in "${SCHEDULES[@]}"; do
        sched_label=$(echo "$sched" | tr ',' '-')
        T=$(run_min_time heat2d_omp "$N_FIXED" "$STEPS" "$NCPUS" "$sched" "$PCORE_CPULIST")
        speedup=$(awk -v s="$T_SERIAL" -v p="$T" 'BEGIN{printf "%.4f", s/p}')
        eff=$(awk -v sp="$speedup" -v t="$NCPUS" 'BEGIN{printf "%.4f", sp/t}')
        echo "${sched_label}_pinned_pcores,$N_FIXED,$NCPUS,$T,$speedup,$eff" >> strong_scaling.csv

        if awk -v a="$T" -v b="$BEST_ELAPSED" 'BEGIN{exit !(a<b)}'; then
            BEST_ELAPSED=$T
            BEST_T=$NCPUS
            BEST_SCHED="${sched}_pinned_pcores"
        fi
    done
fi

echo ""
echo "Best config found: schedule=$BEST_SCHED threads=$BEST_T (elapsed=${BEST_ELAPSED}s)"

# ---------- Sweep 2: Throughput vs N at the single best config ----------
echo "N,elapsed_s,threads,schedule,throughput_gridpts_s,gflops" > throughput_vs_n.csv
for N in 128 256 384 512 768 1024 2048 4096; do
    echo "OpenMP N=$N (best config: threads=$BEST_T schedule=$BEST_SCHED)..."
    sched_arg="${BEST_SCHED/_pinned_pcores/}"
    cpulist_arg=""
    if [[ "$BEST_SCHED" == *_pinned_pcores ]]; then
        cpulist_arg="$PCORE_CPULIST"
    fi
    if [ -n "$cpulist_arg" ]; then
        out=$(OMP_SCHEDULE="$sched_arg" taskset -c "$cpulist_arg" ./heat2d_omp "$N" "$STEPS" "$BEST_T")
    else
        out=$(OMP_SCHEDULE="$sched_arg" ./heat2d_omp "$N" "$STEPS" "$BEST_T")
    fi
    t=$(echo "$out" | grep "Elapsed time" | awk '{print $3}')
    tp=$(echo "$out" | grep "Throughput" | awk '{print $2}')
    gf=$(echo "$out" | grep "Compute rate" | awk '{print $3}')
    best_sched_label=$(echo "$BEST_SCHED" | tr ',' '-')
    echo "$N,$t,$BEST_T,$best_sched_label,$tp,$gf" >> throughput_vs_n.csv
done

echo ""
echo "Done. Results in strong_scaling.csv and throughput_vs_n.csv"