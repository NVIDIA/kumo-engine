#!/usr/bin/env bash
# Live, human-friendly disk I/O statistics powered by sysstat/iostat.

set -euo pipefail

interval=1
device_regex='.*'
once=0
use_color=auto
clear_screen=auto

usage() {
    cat <<'EOF'
Usage: scripts/diskstats.sh [OPTIONS]

Continuously display per-device disk activity. Press q, Esc, or Ctrl-C to stop.

Options:
  -i, --interval SECONDS   Refresh interval (default: 1)
  -d, --device REGEX       Show device names matching REGEX (default: all)
      --once               Print one sample and exit
      --no-color           Disable utilization colors
      --no-clear           Scroll instead of redrawing the terminal
  -h, --help               Show this help

Examples:
  ./scripts/diskstats.sh
  ./scripts/diskstats.sh -d '^nvme'
  ./scripts/diskstats.sh -i 2 -d '^(nvme[0-3]n1|md127)$'
  ./scripts/diskstats.sh --once --no-color

Requires the iostat command from the sysstat package.
EOF
}

die() {
    printf 'diskstats: %s\n' "$*" >&2
    exit 1
}

while (($#)); do
    case $1 in
        -i|--interval)
            (($# >= 2)) || die "$1 requires a value"
            interval=$2
            shift 2
            ;;
        -d|--device)
            (($# >= 2)) || die "$1 requires a value"
            device_regex=$2
            shift 2
            ;;
        --once)
            once=1
            shift
            ;;
        --no-color)
            use_color=0
            shift
            ;;
        --no-clear)
            clear_screen=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            (($# == 0)) || die "unexpected argument: $1"
            ;;
        -*|*)
            die "unknown option: $1 (try --help)"
            ;;
    esac
done

command -v iostat >/dev/null 2>&1 ||
    die "iostat was not found; install the sysstat package"

awk -v value="$interval" 'BEGIN { exit !(value ~ /^[0-9]+([.][0-9]+)?$/ && value + 0 > 0) }' ||
    die "interval must be a positive number of seconds"

if [[ $use_color == auto ]]; then
    if [[ -t 1 && ${TERM:-dumb} != dumb ]]; then
        use_color=1
    else
        use_color=0
    fi
fi

if [[ $clear_screen == auto ]]; then
    if [[ -t 1 && ${TERM:-dumb} != dumb ]]; then
        clear_screen=1
    else
        clear_screen=0
    fi
fi

# Verify the regular expression before starting an endless iostat process.
awk -v pattern="$device_regex" 'BEGIN { if ("" ~ pattern) exit 0; exit 0 }' 2>/dev/null ||
    die "invalid device regular expression: $device_regex"

count_args=()
if ((once)); then
    count_args=(1)
fi

cursor_hidden=0
key_input=0
source_pid=''
renderer_pid=''

cleanup() {
    [[ -z $source_pid ]] || kill "$source_pid" 2>/dev/null || true
    [[ -z $renderer_pid ]] || kill "$renderer_pid" 2>/dev/null || true
    [[ -z $source_pid ]] || wait "$source_pid" 2>/dev/null || true
    [[ -z $renderer_pid ]] || wait "$renderer_pid" 2>/dev/null || true
    if ((cursor_hidden)); then
        printf '\033[?25h'
    fi
}

trap cleanup EXIT
trap 'exit 0' INT TERM

if ((clear_screen)); then
    printf '\033[?25l'
    cursor_hidden=1
fi

if [[ -t 0 && -t 1 ]]; then
    key_input=1
fi

# -y omits the misleading first report, which is averaged since boot.
# LC_ALL=C makes field names and decimal formatting stable for the parser.
LC_ALL=C S_TIME_FORMAT=ISO \
    iostat -d -x -m -y "$interval" "${count_args[@]}" |
    awk \
        -v filter="$device_regex" \
        -v color="$use_color" \
        -v redraw="$clear_screen" \
        -v exit_keys="$key_input" \
        -v sample_interval="$interval" '
function metric(name, fallback,    column) {
    column = index_of[name]
    if (!column && fallback != "")
        column = index_of[fallback]
    return column ? $(column) + 0 : 0
}

function utilization_color(value) {
    if (!color)
        return ""
    if (value >= 95)
        return "\033[1;31m"
    if (value >= 80)
        return "\033[1;33m"
    return "\033[1;32m"
}

function print_legend() {
    if (!printed_rows)
        printf "\n  No devices matched /%s/.\n", filter

    print ""
    printf "  %-14s  %s\n", "R/s, W/s", "read and write operations per second (IOPS)"
    printf "  %-14s  %s\n", "READ, WRITE", "data throughput in MiB per second"
    printf "  %-14s  %s\n", "R_LAT, W_LAT", "average read/write completion latency in milliseconds"
    printf "  %-14s  %s\n", "R_SIZE, W_SIZE", "average read/write request size in KiB"
    printf "  %-14s  %s\n", "QUEUE", "average requests queued or in flight"
    if (color)
        printf "  %-14s  time device was busy: \033[1;32m<80%%\033[0m  \033[1;33m80-94.9%%\033[0m  \033[1;31m>=95%%\033[0m\n", "UTIL"
    else
        printf "  %-14s  %s\n", "UTIL", "percentage of time the device was busy"
    fflush()
}

/^Device[[:space:]]/ {
    if (in_table)
        print_legend()

    delete index_of
    for (column = 1; column <= NF; column++)
        index_of[$column] = column

    if (redraw)
        printf "\033[H\033[2J"
    else if (sample_number)
        print ""

    sample_number++
    in_table = 1
    printed_rows = 0

    quit_hint = exit_keys ? "  (q/Esc/Ctrl-C to quit)" : ""
    if (color)
        printf "\033[1;36mDisk I/O — %s — %ss samples\033[0m%s\n\n", strftime("%Y-%m-%d %H:%M:%S"), sample_interval, quit_hint
    else
        printf "Disk I/O — %s — %ss samples%s\n\n", strftime("%Y-%m-%d %H:%M:%S"), sample_interval, quit_hint

    if (color)
        printf "\033[1;36m%-14s %9s %9s %10s %10s %10s %10s %10s %10s %9s %8s\033[0m\n", \
            "DEVICE", "R/s", "W/s", "READ", "WRITE", "R_LAT", "W_LAT", "R_SIZE", "W_SIZE", "QUEUE", "UTIL"
    else
        printf "%-14s %9s %9s %10s %10s %10s %10s %10s %10s %9s %8s\n", \
            "DEVICE", "R/s", "W/s", "READ", "WRITE", "R_LAT", "W_LAT", "R_SIZE", "W_SIZE", "QUEUE", "UTIL"

    printf "%-14s %9s %9s %10s %10s %10s %10s %10s %10s %9s %8s\n", \
        "", "IOPS", "IOPS", "MiB/s", "MiB/s", "ms", "ms", "KiB", "KiB", "avg", "%"
    next
}

in_table && NF == 0 {
    print_legend()
    in_table = 0
    next
}

in_table && NF > 1 {
    device = $1
    if (device !~ filter)
        next

    reads = metric("r/s", "")
    writes = metric("w/s", "")
    read_mib = metric("rMB/s", "")
    write_mib = metric("wMB/s", "")
    read_latency = metric("r_await", "await")
    write_latency = metric("w_await", "await")
    queue = metric("aqu-sz", "avgqu-sz")
    util = metric("%util", "")

    read_size = metric("rareq-sz", "")
    write_size = metric("wareq-sz", "")

    highlight = utilization_color(util)
    reset = color ? "\033[0m" : ""
    printf "%-14s %9.1f %9.1f %10.1f %10.1f %10.2f %10.2f %10.1f %10.1f %9.2f %s%7.1f%%%s\n", \
        device, reads, writes, read_mib, write_mib, read_latency, write_latency, \
        read_size, write_size, queue, highlight, util, reset
    printed_rows++
    fflush()
    next
}

END {
    if (in_table)
        print_legend()
}
' &

renderer_pid=$!
# In a background pipeline, $! is the final renderer and jobs -p reports the
# first process (iostat). Track both so an interactive quit stops immediately.
source_pid=$(jobs -p %% 2>/dev/null || true)

if ((key_input)); then
    while kill -0 "$renderer_pid" 2>/dev/null; do
        if IFS= read -rsn1 -t 0.1 key </dev/tty; then
            case $key in
                q|Q|$'\e')
                    exit 0
                    ;;
            esac
        fi
    done
fi

renderer_status=0
wait "$renderer_pid" || renderer_status=$?
source_pid=''
renderer_pid=''
exit "$renderer_status"
