#!/usr/bin/env python
"""
isolate_pytest.py - turn one pytest invocation into N isolated pytest invocations.

Why this exists
---------------
Long BabelBrain pytest runs (`pytest Tests -s -m "basic_babelbrain_params"` collects
~1300 cases) sometimes die half way through with nothing useful left on screen:
the interpreter is killed by a signal, the terminal scrollback is gone, and the
single html report is never written.  Memory also looks like it creeps up across
cases even though every BabelBrain step is supposed to release its GPU buffers.

This tool takes *the same arguments you would give pytest*, asks pytest which
cases that selects, and then runs every case in its own pytest process:

  * a process per case is the only bullet proof deallocation - when the process
    exits, every host buffer, Metal/CUDA/OpenCL context and Qt object is gone,
    so any leak that survives is either on disk or in the OS/GPU driver;
  * each case gets its own log file, so a segfault/abort/OOM kill leaves a file
    you can read afterwards instead of lost scrollback;
  * exit codes and signals are recorded per case, so "which case broke it" is a
    lookup, not a bisect;
  * peak RSS and system free memory are sampled per case, so a leak shows up as
    a trend in summary.csv instead of a hunch.

Usage
-----
  # 1) generate the per-case runner script (does not run anything)
  python Tests/Tools/isolate_pytest.py -- Tests -s -m "basic_babelbrain_params"

  # 2) or collect and run right away, one case per process
  python Tests/Tools/isolate_pytest.py --run -- Tests -s -m "basic_babelbrain_params"

  # rerun only what broke last time
  python Tests/Tools/isolate_pytest.py --run --from-file PyTest_Reports/isolated/<run>/failed_nodeids.txt

  # leak hunting: run each case 5x inside ONE process and watch peak RSS
  python Tests/Tools/isolate_pytest.py --run --repeat-each 5 --group 5 -- Tests -k "H317 and Metal and NONE" -m basic_babelbrain_params

  # bisect "does it only die when cases share a process?": 1, then 5, then 20
  python Tests/Tools/isolate_pytest.py --run --group 20 -- Tests -m "basic_babelbrain_params"

Everything after `--` is passed to pytest verbatim for the collection step.
For the per-case runs, path/`-k`/`-m` selection arguments are dropped (the node
id already selects exactly one case) and every other flag is kept. With no path
argument the whole Tests tree is selected, so this is enough:

  python Tests/Tools/isolate_pytest.py --run -- -s -m "basic_babelbrain_params"

Where it runs
-------------
This file lives in the BabelBrainPyTest repo, which is cloned as <BabelBrain>/Tests.
pytest itself must run from the BabelBrain checkout above that clone (conftest.py
reads 'Tests/config.ini' and adds './BabelBrain/' to sys.path, both relative to the
working directory), so the tool derives that root from its own location and chdirs
there for you - invoke it from anywhere, including from inside Tests/. Point it at
a different checkout with --root.

Artifacts land in <BabelBrain>/PyTest_Reports/isolated/<timestamp>/, next to the
html reports pytest.ini already writes there (gitignored by BabelBrain, and outside
the Tests clone so nothing lands in the test repo).
"""

import argparse
import csv
import datetime
import os
import platform
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

try:
    import psutil
except Exception:  # psutil is optional, we just lose the memory columns
    psutil = None

IS_WINDOWS = platform.system() == "Windows"
IS_DARWIN = platform.system() == "Darwin"

# This tool lives in the BabelBrainPyTest repo (cloned as <BabelBrain>/Tests), but
# pytest has to be invoked from the BabelBrain checkout that contains that clone:
# Tests/conftest.py reads 'Tests/config.ini' and appends './BabelBrain/' to sys.path,
# both relative to the working directory. Anchor everything on __file__ so the tool
# works from any directory, and so several BabelBrain checkouts can each hold their
# own Tests clone without the runs getting mixed up.
TOOL_PATH = Path(__file__).resolve()
TESTS_DIR = TOOL_PATH.parent.parent               # <BabelBrain>/Tests
DEFAULT_ROOT = TESTS_DIR.parent                   # <BabelBrain>

# pytest options that consume the following token as their value
VALUE_OPTS = {
    "-m", "-k", "-p", "-o", "-c", "-n", "-r", "-W", "--tb", "--color", "--capture",
    "--html", "--junitxml", "--junit-xml", "--deselect", "--ignore", "--ignore-glob",
    "--rootdir", "--confcutdir", "--import-mode", "--durations", "--maxfail",
    "--log-file", "--log-level", "--log-cli-level", "--basetemp", "--override-ini",
}
# selection options: meaningless (and risky) once we run by exact node id
SELECTION_OPTS = {"-m", "-k", "--deselect", "--ignore", "--ignore-glob", "--last-failed", "--lf",
                  "--failed-first", "--ff", "--collect-only", "--co"}

STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"
STATUS_CRASH = "CRASH"
STATUS_TIMEOUT = "TIMEOUT"
STATUS_MEMLIMIT = "MEMLIMIT"
STATUS_NOTFOUND = "NO_TESTS_RUN"
STATUS_ERROR = "ERROR"
BAD_STATUSES = {STATUS_FAIL, STATUS_CRASH, STATUS_TIMEOUT, STATUS_MEMLIMIT, STATUS_NOTFOUND, STATUS_ERROR}


# ----------------------------------------------------------------------------- args
def split_argv(argv):
    """Split our own options from the pytest arguments (separated by `--`)."""
    if "--" in argv:
        i = argv.index("--")
        return argv[:i], argv[i + 1:]
    return argv, []


def parse_args(argv):
    p = argparse.ArgumentParser(
        prog="isolate_pytest.py",
        description="Run a pytest selection one case per process, with per-case logs and memory stats.",
        epilog="Put the pytest arguments after `--`, e.g. ... --run -- Tests -s -m \"basic_babelbrain_params\"",
    )
    p.add_argument("--run", action="store_true",
                   help="run the cases now (default: only collect and write the runner scripts)")
    p.add_argument("--from-file", metavar="FILE",
                   help="read node ids from FILE (one per line) instead of running a pytest collection")
    p.add_argument("--root", metavar="DIR", default=None,
                   help=f"BabelBrain checkout to run pytest from (default: the one containing this "
                        f"Tests clone, {DEFAULT_ROOT})")
    p.add_argument("--out", metavar="DIR", default=None,
                   help="output directory (default PyTest_Reports/isolated/<timestamp>)")
    p.add_argument("--emit", choices=["sh", "bat", "both", "none"], default=None,
                   help="which runner script to write (default: sh on posix, bat on Windows)")
    p.add_argument("--group", type=int, default=1, metavar="N",
                   help="cases per pytest process (default 1 = full isolation). Raise it to test whether "
                        "failures only appear when cases share a process")
    p.add_argument("--repeat-each", type=int, default=1, metavar="N",
                   help="repeat every case N times back to back (pair with --group N to look for growth "
                        "inside a single process)")
    p.add_argument("--max-cases", type=int, default=None, metavar="N", help="only take the first N cases")
    p.add_argument("--start-at", type=int, default=1, metavar="N", help="start at the Nth case (1-based)")
    p.add_argument("--timeout", type=float, default=None, metavar="SEC",
                   help="kill a case after SEC seconds and mark it TIMEOUT")
    p.add_argument("--max-rss-mb", type=float, default=None, metavar="MB",
                   help="kill a case if its process tree exceeds MB of RSS (needs psutil)")
    p.add_argument("--stop-on-crash", action="store_true",
                   help="stop the whole run at the first crash/timeout (failures do not stop it)")
    p.add_argument("--stop-on-fail", action="store_true", help="stop at the first non-passing case")
    p.add_argument("--no-html", dest="keep_html", action="store_false", default=True,
                   help="disable the per-case pytest-html report. Off by default because conftest.py "
                        "attaches the GUI screenshots to it through pytest_html.extras, which only exists "
                        "while the html plugin is loaded")
    p.add_argument("--quiet-child", action="store_true",
                   help="do not echo the pytest output to the console (it always goes to the log file)")
    p.add_argument("--python", default=sys.executable, metavar="EXE",
                   help="python interpreter used to run pytest (default: the one running this script)")
    p.add_argument("--heartbeat", type=float, default=60.0, metavar="SEC",
                   help="while a case produces no output, print elapsed time and RSS every SEC seconds "
                        "(0 disables). Long GPU steps are silent for minutes")
    p.add_argument("--cooldown", type=float, default=0.0, metavar="SEC",
                   help="sleep between cases, e.g. to let the GPU driver settle")
    p.add_argument("--dry-run", action="store_true", help="print the per-case commands instead of running them")
    return p.parse_args(argv)


# ------------------------------------------------------------------- pytest args
def positional_args(pytest_args):
    """The path / node id arguments of a pytest command line."""
    pos, i = [], 0
    while i < len(pytest_args):
        a = pytest_args[i]
        if a.startswith("-"):
            name = a.split("=", 1)[0]
            if name in VALUE_OPTS and "=" not in a and not (len(a) > 2 and not a.startswith("--")):
                i += 1
        else:
            pos.append(a)
        i += 1
    return pos


def rebase_positional_args(pytest_args, root):
    """Rewrite path arguments so they still work after we chdir to the BabelBrain root.

    `Tests/Tools/isolate_pytest.py -- Unit/BabelBrain/test_FileManager.py` typed from
    inside Tests/ has to become `Tests/Unit/BabelBrain/test_FileManager.py`."""
    here = Path.cwd()
    if here == root:
        return list(pytest_args)
    positionals = set(positional_args(pytest_args))
    out = []
    for a in pytest_args:
        if a in positionals:
            path_part, sep, rest = a.partition("::")
            if not (root / path_part).exists() and (here / path_part).exists():
                rebased = os.path.relpath((here / path_part).resolve(), root)
                a = rebased + sep + rest
        out.append(a)
    return out


def resolve_nodeid_prefix(nodeids, cwd, pytest_args):
    """Node ids are printed relative to pytest's rootdir (Tests/ here, because
    Tests/pytest.ini is the configfile) but they are interpreted relative to the
    directory pytest is invoked from - and the conftest needs to be invoked from
    the repo root, since it reads 'Tests/config.ini'. Work out the prefix that
    makes the collected ids resolvable from cwd."""
    first = nodeids[0].split("::")[0]
    if (cwd / first).exists():
        return ""
    candidates = []
    for a in positional_args(pytest_args):
        base = Path(a.split("::")[0])
        candidates.append(base if base.is_dir() else base.parent)
        for parent in base.parents:
            candidates.append(parent)
    for c in candidates:
        if c and (cwd / c / first).exists():
            prefix = c.as_posix().rstrip("/")
            return "" if prefix in ("", ".") else prefix + "/"
    print(f"WARNING: could not locate '{first}' from {cwd}; node ids may not resolve.", file=sys.stderr)
    return ""


def filter_pytest_flags(pytest_args):
    """Drop positional (path/node id) args and selection options, keep everything else."""
    kept, i = [], 0
    while i < len(pytest_args):
        a = pytest_args[i]
        if a.startswith("-"):
            name = a.split("=", 1)[0]
            takes_value = name in VALUE_OPTS and "=" not in a
            is_selection = name in SELECTION_OPTS or (
                len(a) > 2 and a[0] == "-" and a[1] in "mk" and not a.startswith("--")
            )
            if is_selection:
                i += 2 if takes_value else 1
                continue
            kept.append(a)
            if takes_value and i + 1 < len(pytest_args):
                kept.append(pytest_args[i + 1])
                i += 1
        # else: positional (a path or node id) -> dropped, we supply our own node id
        i += 1
    return kept


def html_args(run_dir, tag_expr, slug_expr, keep_html, sep="/"):
    """`--html` for one case. tag/slug are literal for the driver and shell variable
    references for the generated scripts, so both produce the same file names."""
    if not keep_html:
        return []
    return ["--html", f"{run_dir}{sep}html{sep}{tag_expr}_{slug_expr}.html", "--self-contained-html"]


def build_per_case_flags(pytest_args, keep_html):
    flags = ["-o", "addopts="]           # ini addopts drives the shared html report, drop it
    user = filter_pytest_flags(pytest_args)
    if not keep_html:
        # the html plugin is disabled below, so its options would be unrecognised
        user = [f for f in user if not f.startswith("--html") and f != "--self-contained-html"]
    flags += user
    if not any(f == "-s" or f.startswith("--capture") for f in flags):
        flags.append("-s")               # unbuffered: native/C crash output reaches the log
    if not any(f.startswith("--tb") for f in flags):
        flags += ["--tb=long"]
    if not any(f.startswith("--color") for f in flags):
        flags += ["--color=no"]          # log files without ANSI escapes
    flags += ["-p", "no:cacheprovider", "-rA"]
    if not keep_html:
        flags += ["-p", "no:html"]
    return flags


# ------------------------------------------------------------------- collection
NODEID_RE = re.compile(r"^[^\s<>][^\s]*::[^\s].*$")


VERBOSITY_FLAGS = re.compile(r"^(-v+|-q+|--verbose(=.*)?|--quiet|--no-header|--collect-only|--co)$")


def collect_nodeids(python_exe, pytest_args, cwd):
    # `--collect-only -q` prints one node id per line, but only at verbosity -1
    # exactly: -vv (from the ini addopts) prints a tree and -qq prints per-file
    # counts. So clear addopts, strip any verbosity flag the caller passed, and
    # set our own single -q.
    clean = [a for a in pytest_args if not VERBOSITY_FLAGS.match(a)]
    # Clearing addopts already drops the ini's --html, so pytest-html writes nothing
    # for a collection; no need to unload the plugin here (and unloading it is what
    # breaks conftest's screenshot hook, so keep the two passes consistent).
    cmd = [python_exe, "-m", "pytest", *clean,
           "--collect-only", "-q", "--color=no", "--no-header",
           "-o", "addopts=", "-p", "no:cacheprovider", "-p", "no:warnings"]
    print("Collecting cases (collection only, no tests are run):\n  "
          + " ".join(shlex.quote(c) for c in cmd) + "\n", flush=True)
    proc = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, errors="replace")
    out = proc.stdout
    ids, seen = [], set()
    for line in out.splitlines():
        line = line.rstrip()
        if not line or "::" not in line or line[0].isspace():
            continue
        if line.startswith(("<", "E ", "ERROR", "FAILED", "warning", "WARNING")):
            continue
        if not NODEID_RE.match(line):
            continue
        if line not in seen:
            seen.add(line)
            ids.append(line)
    if not ids:
        print(out[-4000:], file=sys.stderr)
        raise SystemExit(f"No node ids collected (pytest exit code {proc.returncode}). See output above.")
    prefix = resolve_nodeid_prefix(ids, cwd, pytest_args)
    if prefix:
        ids = [prefix + i for i in ids]
        print(f"Node ids are relative to pytest's rootdir; prefixed with '{prefix}' so they resolve from {cwd}.")
    print(f"Collected {len(ids)} cases.", flush=True)
    return ids


# ------------------------------------------------------------------- helpers
def tool_hint(root):
    """This script's path, written relative to the BabelBrain root when possible."""
    try:
        return TOOL_PATH.relative_to(root).as_posix()
    except ValueError:
        return str(TOOL_PATH)


def slugify(nodeid, maxlen=110):
    s = re.sub(r"[^0-9A-Za-z._-]+", "_", nodeid.split("::")[-1]).strip("_")
    return s[:maxlen] if s else "case"


def describe_returncode(rc):
    if rc == 0:
        return STATUS_PASS, ""
    if rc < 0:
        signum = -rc
        try:
            name = signal.Signals(signum).name
        except ValueError:
            name = f"SIG{signum}"
        note = {"SIGSEGV": "segmentation fault", "SIGBUS": "bus error",
                "SIGABRT": "abort (assert / C++ throw / malloc error)",
                "SIGKILL": "killed by the OS (out of memory / jetsam?)",
                "SIGILL": "illegal instruction"}.get(name, "")
        return STATUS_CRASH, f"{name} {note}".strip()
    if rc == 1:
        return STATUS_FAIL, "tests failed"
    if rc == 2:
        return STATUS_ERROR, "interrupted"
    if rc == 3:
        return STATUS_ERROR, "internal pytest error"
    if rc == 4:
        return STATUS_ERROR, "pytest usage error"
    if rc == 5:
        return STATUS_NOTFOUND, "node id matched nothing"
    if rc > 128:  # shell-style signal reporting
        return STATUS_CRASH, f"signal {rc - 128}"
    return STATUS_ERROR, f"exit code {rc}"


def sys_mem():
    if psutil is None:
        return None, None
    vm = psutil.virtual_memory()
    try:
        swap = psutil.swap_memory().used / 1e6
    except Exception:
        swap = None
    return vm.available / 1e6, swap


class MemWatcher(threading.Thread):
    """Sample RSS of the pytest process tree; optionally kill it above a limit."""

    def __init__(self, pid, limit_mb=None, interval=0.25):
        super().__init__(daemon=True)
        self.pid, self.limit_mb, self.interval = pid, limit_mb, interval
        self.peak_mb = 0.0
        self.min_avail_mb = None
        self.killed_for_memory = False
        self._stop = threading.Event()

    def run(self):
        if psutil is None:
            return
        try:
            proc = psutil.Process(self.pid)
        except Exception:
            return
        while not self._stop.is_set():
            try:
                children = proc.children(recursive=True)
                rss = proc.memory_info().rss
                for c in children:
                    try:
                        rss += c.memory_info().rss
                    except Exception:
                        pass
                self.peak_mb = max(self.peak_mb, rss / 1e6)
                avail, _ = sys_mem()
                if avail is not None:
                    self.min_avail_mb = avail if self.min_avail_mb is None else min(self.min_avail_mb, avail)
                if self.limit_mb and rss / 1e6 > self.limit_mb:
                    self.killed_for_memory = True
                    for c in children:
                        try:
                            c.kill()
                        except Exception:
                            pass
                    proc.kill()
                    return
            except psutil.NoSuchProcess:
                return
            except Exception:
                pass
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()


# ------------------------------------------------------------------- run one chunk
def run_chunk(idx, total_chunks, nodeids, flags, python_exe, cwd, run_dir, args):
    logs_dir = run_dir / "logs"
    tag = f"{idx:04d}"
    slug = chunk_slug(nodeids)
    log_path = logs_dir / f"{tag}_{slug}.log"

    cmd = [python_exe, "-m", "pytest", *nodeids, *flags,
           "-o", f"log_file={logs_dir / (tag + '_' + slug + '.pytest.log')}"]
    cmd += html_args(run_dir, tag, slug, args.keep_html, sep=os.sep)

    header = f"[{idx}/{total_chunks}] " + (nodeids[0] if len(nodeids) == 1 else f"{len(nodeids)} cases starting with {nodeids[0]}")
    print("\n" + "=" * 100 + f"\n{header}\n" + "=" * 100, flush=True)

    if args.dry_run:
        print("  " + " ".join(shlex.quote(c) for c in cmd), flush=True)
        return dict(index=idx, nodeid=nodeids[0], n_cases=len(nodeids), status="DRY_RUN", returncode="",
                    detail="", duration_s=0.0, peak_rss_mb="", sys_avail_before_mb="", sys_avail_after_mb="",
                    sys_avail_min_mb="", swap_used_after_mb="", log=str(log_path.relative_to(run_dir)))

    env = dict(os.environ)
    env["PYTHONFAULTHANDLER"] = "1"   # a segfault then prints the python stack that caused it
    env["PYTHONUNBUFFERED"] = "1"

    avail_before, swap_before = sys_mem()
    t0 = time.monotonic()

    logs_dir.mkdir(parents=True, exist_ok=True)
    if args.keep_html:
        (run_dir / "html").mkdir(parents=True, exist_ok=True)

    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        log.write("# " + " ".join(shlex.quote(c) for c in cmd) + "\n")
        log.write(f"# cwd: {cwd}\n# started: {datetime.datetime.now().isoformat(timespec='seconds')}\n")
        for nid in nodeids:
            log.write(f"# case: {nid}\n")
        log.write("#" + "-" * 90 + "\n")
        log.flush()

        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1)

        last_output = [time.monotonic()]

        def pump():
            for line in proc.stdout:
                last_output[0] = time.monotonic()
                log.write(line)
                log.flush()          # flush per line: a hard crash keeps everything written so far
                if not args.quiet_child:
                    sys.stdout.write(line)
            try:
                sys.stdout.flush()
            except Exception:
                pass

        pump_thread = threading.Thread(target=pump, daemon=True)
        pump_thread.start()

        watcher = MemWatcher(proc.pid, args.max_rss_mb)
        watcher.start()

        done = threading.Event()

        def heartbeat():
            # a silent case is normal here (an FDTD step can run for minutes without
            # printing), so show that it is alive and how its memory is trending
            last_beat = time.monotonic()
            while not done.wait(2.0):
                now = time.monotonic()
                quiet = now - last_output[0]
                if quiet >= args.heartbeat and now - last_beat >= args.heartbeat:
                    last_beat = now
                    rss = f", RSS {watcher.peak_mb:.0f} MB peak" if watcher.peak_mb else ""
                    print(f"    ... running {(now - t0) / 60:.1f} min, no output for {quiet:.0f}s{rss}",
                          flush=True)

        hb_thread = None
        if args.heartbeat and args.heartbeat > 0:
            hb_thread = threading.Thread(target=heartbeat, daemon=True)
            hb_thread.start()

        timed_out = False
        try:
            proc.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            log.write(f"\n### isolate_pytest: timeout after {args.timeout}s, terminating\n")
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                log.write("### isolate_pytest: still alive, SIGKILL\n")
                proc.kill()
                proc.wait()

        watcher.stop()
        done.set()
        if hb_thread is not None:
            hb_thread.join(timeout=10)
        pump_thread.join(timeout=30)
        rc = proc.returncode

    duration = time.monotonic() - t0
    avail_after, swap_after = sys_mem()

    pytest_log = logs_dir / f"{tag}_{slug}.pytest.log"
    if pytest_log.exists() and pytest_log.stat().st_size == 0:
        pytest_log.unlink()          # these tests print rather than log; do not leave empty files around

    if timed_out:
        status, detail = STATUS_TIMEOUT, f"exceeded {args.timeout}s"
    elif watcher.killed_for_memory:
        status, detail = STATUS_MEMLIMIT, f"exceeded --max-rss-mb {args.max_rss_mb}"
    else:
        status, detail = describe_returncode(rc)

    row = dict(
        index=idx, nodeid=nodeids[0] if len(nodeids) == 1 else ";".join(nodeids),
        n_cases=len(nodeids), status=status, returncode=rc, detail=detail,
        duration_s=round(duration, 1),
        peak_rss_mb=round(watcher.peak_mb, 1) if watcher.peak_mb else "",
        sys_avail_before_mb=round(avail_before) if avail_before is not None else "",
        sys_avail_after_mb=round(avail_after) if avail_after is not None else "",
        sys_avail_min_mb=round(watcher.min_avail_mb) if watcher.min_avail_mb is not None else "",
        swap_used_after_mb=round(swap_after) if swap_after is not None else "",
        log=str(log_path.relative_to(run_dir)),
    )
    mem = f", peak RSS {row['peak_rss_mb']} MB" if row["peak_rss_mb"] else ""
    freed = ""
    if avail_before is not None and avail_after is not None:
        freed = f", system free {avail_before:.0f} -> {avail_after:.0f} MB"
    print(f"--> {status} (rc={rc}{', ' + detail if detail else ''}) in {duration:.1f}s{mem}{freed}\n"
          f"    log: {log_path}", flush=True)
    return row


# ------------------------------------------------------------------- script emitters
SH_TEMPLATE = r"""@@SHEBANG@@
# Generated by Tests/Tools/isolate_pytest.py on @@WHEN@@
# Source command: @@SRCCMD@@
#
# Runs every collected case in its own pytest process, so all memory (host, GPU,
# Qt) is reclaimed by the OS between cases and a crash is attributable to exactly
# one case. Per case output goes to $LOGDIR, results to $SUMMARY.
# Comment out any run_case line below to skip that case.

cd "@@CWD@@" || exit 1

PYTHON="@@PYTHON@@"
RUNDIR="@@RUNDIR@@"
LOGDIR="$RUNDIR/logs"
HTMLDIR="$RUNDIR/html"
SUMMARY="$RUNDIR/summary_shell.csv"
TOTAL=@@TOTAL@@
STOP_ON_CRASH=0            # set to 1 to abort at the first crash
PYTEST_FLAGS=(@@FLAGS@@)

export PYTHONFAULTHANDLER=1   # turn a segfault into a printable python traceback
export PYTHONUNBUFFERED=1

mkdir -p "$LOGDIR" "$HTMLDIR"
echo "index,status,returncode,duration_s,peak_rss_mb,nodeid" > "$SUMMARY"

TIMECMD=()
if [ "$(uname)" = "Darwin" ] && [ -x /usr/bin/time ]; then
  TIMECMD=(/usr/bin/time -l)
elif [ -x /usr/bin/time ]; then
  TIMECMD=(/usr/bin/time -v)
fi

FAILED=0
run_case() {
  IDX="$1"; SLUG="$2"; shift 2      # everything left is one or more pytest node ids
  LOG="$LOGDIR/${IDX}_${SLUG}.log"
  echo ""
  echo "===================================================================================="
  echo "[$((10#$IDX))/$TOTAL] $*"
  echo "===================================================================================="
  START=$(date +%s)
  {
    echo "# $*"
    "${TIMECMD[@]}" "$PYTHON" -m pytest "$@" "${PYTEST_FLAGS[@]}" @@HTMLARGS_SH@@ \
        -o "log_file=$LOGDIR/${IDX}_${SLUG}.pytest.log"
  } > "$LOG" 2>&1
  RC=$?
  [ -s "$LOGDIR/${IDX}_${SLUG}.pytest.log" ] || rm -f "$LOGDIR/${IDX}_${SLUG}.pytest.log"
  END=$(date +%s)
  DUR=$((END-START))
  PEAK=$(grep -i -m1 "maximum resident set size" "$LOG" | tr -dc '0-9')
  if [ -n "$PEAK" ]; then PEAK=$((PEAK/1048576)); fi   # bytes (macOS) -> MB
  case $RC in
    0) STATUS="PASS" ;;
    1) STATUS="FAIL" ;;
    5) STATUS="NO_TESTS_RUN" ;;
    *) if [ $RC -gt 128 ]; then STATUS="CRASH(sig $((RC-128)))"; else STATUS="ERROR"; fi ;;
  esac
  echo "$IDX,$STATUS,$RC,$DUR,${PEAK:-},\"$*\"" >> "$SUMMARY"
  echo "--> $STATUS rc=$RC in ${DUR}s peakRSS=${PEAK:-?}MB  log: $LOG"
  if [ "$STATUS" != "PASS" ]; then FAILED=$((FAILED+1)); fi
  if [ "$STOP_ON_CRASH" = "1" ] && [ $RC -ne 0 ] && [ $RC -ne 1 ]; then
    echo "Stopping: case $IDX crashed (rc=$RC). Inspect $LOG"
    exit $RC
  fi
}

@@CASES@@

echo ""
echo "Done. $FAILED of $TOTAL cases did not pass. Summary: $SUMMARY"
grep -v ",PASS," "$SUMMARY" | tail -n +1
"""

BAT_TEMPLATE = r"""@echo off
REM Generated by Tests\Tools\isolate_pytest.py on @@WHEN@@
REM Source command: @@SRCCMD@@
REM Runs every collected case in its own pytest process. Logs in %LOGDIR%.
setlocal enabledelayedexpansion

cd /d "@@CWD@@" || exit /b 1

set "PYTHON=@@PYTHON@@"
set "RUNDIR=@@RUNDIR@@"
set "LOGDIR=%RUNDIR%\logs"
set "HTMLDIR=%RUNDIR%\html"
set "SUMMARY=%RUNDIR%\summary_shell.csv"
set TOTAL=@@TOTAL@@
set STOP_ON_CRASH=0
set "PYTEST_FLAGS=@@FLAGS_FLAT@@"
set PYTHONFAULTHANDLER=1
set PYTHONUNBUFFERED=1
set FAILED=0

if not exist "%LOGDIR%" mkdir "%LOGDIR%"
if not exist "%HTMLDIR%" mkdir "%HTMLDIR%"
echo index,status,returncode,nodeid> "%SUMMARY%"

@@CASES@@

echo.
echo Done. %FAILED% of %TOTAL% cases did not pass. Summary: %SUMMARY%
goto :eof

:run_case
set "IDX=%~1"
set "SLUG=%~2"
shift
shift
set "NODEID="
:collect_ids
if "%~1"=="" goto :run_now
set NODEID=!NODEID! "%~1"
shift
goto :collect_ids
:run_now
set "LOG=%LOGDIR%\%IDX%_%SLUG%.log"
echo.
echo ====================================================================================
echo [%IDX%/%TOTAL%] !NODEID!
echo ====================================================================================
"%PYTHON%" -m pytest !NODEID! %PYTEST_FLAGS% @@HTMLARGS_BAT@@ -o "log_file=%LOGDIR%\%IDX%_%SLUG%.pytest.log" > "%LOG%" 2>&1
set RC=!errorlevel!
if !RC! equ 0 (set "STATUS=PASS") else if !RC! equ 1 (set "STATUS=FAIL") else if !RC! equ 5 (set "STATUS=NO_TESTS_RUN") else (set "STATUS=CRASH")
echo %IDX%,!STATUS!,!RC!,"!NODEID!">> "%SUMMARY%"
echo --^> !STATUS! rc=!RC!  log: %LOG%
if not "!STATUS!"=="PASS" set /a FAILED+=1
if "%STOP_ON_CRASH%"=="1" if !RC! gtr 1 (
  echo Stopping: case %IDX% crashed rc=!RC!. Inspect %LOG%
  exit /b !RC!
)
goto :eof
"""


def chunk_slug(chunk):
    if len(chunk) == 1:
        return slugify(chunk[0])
    return f"{slugify(chunk[0], 60)}__plus{len(chunk) - 1}"


def emit_sh(path, chunks, flags, python_exe, cwd, run_dir, src_cmd, keep_html=True):
    cases = []
    for i, chunk in enumerate(chunks, 1):
        ids = " ".join(shlex.quote(n) for n in chunk)
        cases.append(f'run_case {i:04d} {shlex.quote(chunk_slug(chunk))} {ids}')
    body = (SH_TEMPLATE
            .replace("@@SHEBANG@@", "#!/bin/zsh" if IS_DARWIN else "#!/usr/bin/env bash")
            .replace("@@WHEN@@", datetime.datetime.now().isoformat(timespec="seconds"))
            .replace("@@SRCCMD@@", src_cmd)
            .replace("@@CWD@@", str(cwd))
            .replace("@@PYTHON@@", python_exe)
            .replace("@@RUNDIR@@", str(run_dir))
            .replace("@@TOTAL@@", str(len(chunks)))
            .replace("@@FLAGS@@", " ".join(shlex.quote(f) for f in flags))
            .replace("@@HTMLARGS_SH@@",
                     '--html "$HTMLDIR/${IDX}_${SLUG}.html" --self-contained-html' if keep_html else "")
            .replace("@@CASES@@", "\n".join(cases)))
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def emit_bat(path, chunks, flags, python_exe, cwd, run_dir, src_cmd, keep_html=True):
    cases = []
    for i, chunk in enumerate(chunks, 1):
        ids = " ".join(f'"{n}"' for n in chunk)
        cases.append(f'call :run_case {i:04d} "{chunk_slug(chunk)}" {ids}')
    body = (BAT_TEMPLATE
            .replace("@@WHEN@@", datetime.datetime.now().isoformat(timespec="seconds"))
            .replace("@@SRCCMD@@", src_cmd)
            .replace("@@CWD@@", str(cwd))
            .replace("@@PYTHON@@", python_exe)
            .replace("@@RUNDIR@@", str(run_dir))
            .replace("@@TOTAL@@", str(len(chunks)))
            .replace("@@FLAGS_FLAT@@", " ".join(flags))
            .replace("@@HTMLARGS_BAT@@",
                     '--html "%HTMLDIR%\\%IDX%_%SLUG%.html" --self-contained-html' if keep_html else "")
            .replace("@@CASES@@", "\n".join(cases)))
    path.write_text(body, encoding="utf-8")


# ------------------------------------------------------------------- main
def main():
    own_argv, pytest_args = split_argv(sys.argv[1:])
    args = parse_args(own_argv)
    if not pytest_args and not args.from_file:
        raise SystemExit("Nothing to do: pass the pytest arguments after `--`, or use --from-file.\n"
                         'Example: python Tests/Tools/isolate_pytest.py --run -- Tests -s -m "basic_babelbrain_params"')

    root = Path(args.root).resolve() if args.root else DEFAULT_ROOT
    if not (root / TESTS_DIR.name / "conftest.py").exists():
        print(f"WARNING: {root} does not look like a BabelBrain checkout holding this Tests clone "
              f"(no {TESTS_DIR.name}/conftest.py). Pass --root if the run fails.", file=sys.stderr)
    cwd = root                       # every pytest process runs from the BabelBrain root
    pytest_args = rebase_positional_args(pytest_args, root)
    if pytest_args and not positional_args(pytest_args):
        pytest_args = [TESTS_DIR.name] + pytest_args   # no path given: select the whole Tests tree

    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(args.out).resolve() if args.out else root / "PyTest_Reports" / "isolated" / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)

    # ---- what to run
    if args.from_file:
        nodeids = [l.strip() for l in Path(args.from_file).read_text().splitlines()
                   if l.strip() and not l.startswith("#")]
        print(f"Loaded {len(nodeids)} cases from {args.from_file}")
    else:
        nodeids = collect_nodeids(args.python, pytest_args, cwd)

    nodeids = nodeids[args.start_at - 1:]
    if args.max_cases:
        nodeids = nodeids[:args.max_cases]
    if args.repeat_each > 1:
        nodeids = [n for n in nodeids for _ in range(args.repeat_each)]
    if not nodeids:
        raise SystemExit("No cases left to run after --start-at/--max-cases.")

    group = max(1, args.group)
    chunks = [nodeids[i:i + group] for i in range(0, len(nodeids), group)]

    src_cmd = "pytest " + " ".join(shlex.quote(a) for a in pytest_args) if pytest_args else f"--from-file {args.from_file}"
    flags = build_per_case_flags(pytest_args, args.keep_html)

    (run_dir / "nodeids.txt").write_text("\n".join(nodeids) + "\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(
        f"source command : {src_cmd}\n"
        f"isolate_pytest : {' '.join(shlex.quote(a) for a in sys.argv)}\n"
        f"per-case flags : {' '.join(shlex.quote(f) for f in flags)}\n"
        f"cwd            : {cwd}\n"
        f"python         : {args.python}\n"
        f"cases          : {len(nodeids)} in {len(chunks)} process(es)\n", encoding="utf-8")

    emit = args.emit or ("bat" if IS_WINDOWS else "sh")
    written = []
    if emit in ("sh", "both"):
        p = run_dir / "run_isolated.sh"
        emit_sh(p, chunks, flags, args.python, cwd, run_dir, src_cmd, args.keep_html)
        written.append(p)
    if emit in ("bat", "both"):
        p = run_dir / "run_isolated.bat"
        emit_bat(p, chunks, flags, args.python, cwd, run_dir, src_cmd, args.keep_html)
        written.append(p)

    per_case_preview = ["pytest", "<node id>"] + flags + html_args(
        run_dir, "NNNN", "<case>", args.keep_html, sep=os.sep)
    print(f"\nBabelBrain root  : {root}")
    print(f"Each case runs   : " + " ".join(shlex.quote(c) for c in per_case_preview))
    print(f"Output directory : {run_dir}")
    print(f"Cases            : {len(nodeids)} ({len(chunks)} pytest process(es), {group} case(s) each)")
    print(f"Node ids         : {run_dir / 'nodeids.txt'}")
    for p in written:
        print(f"Runner script    : {p}")

    if not args.run:
        print("\nNothing was executed. Run the generated script, or re-invoke with --run.")
        return 0

    # ---- run
    summary_path = run_dir / "summary.csv"
    fields = ["index", "nodeid", "n_cases", "status", "returncode", "detail", "duration_s",
              "peak_rss_mb", "sys_avail_before_mb", "sys_avail_after_mb", "sys_avail_min_mb",
              "swap_used_after_mb", "log"]
    rows = []
    t_start = time.monotonic()
    avail_at_start, _ = sys_mem()
    stopped_early = False

    with open(summary_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        fh.flush()
        for i, chunk in enumerate(chunks, 1):
            try:
                row = run_chunk(i, len(chunks), chunk, flags, args.python, cwd, run_dir, args)
            except KeyboardInterrupt:
                print("\nInterrupted by user.")
                stopped_early = True
                break
            rows.append(row)
            writer.writerow(row)
            fh.flush()               # summary survives a crash of this driver too
            if args.stop_on_fail and row["status"] in BAD_STATUSES:
                print(f"\nStopping at case {i}: {row['status']} (--stop-on-fail)")
                stopped_early = True
                break
            if args.stop_on_crash and row["status"] in (STATUS_CRASH, STATUS_TIMEOUT, STATUS_MEMLIMIT):
                print(f"\nStopping at case {i}: {row['status']} (--stop-on-crash)")
                stopped_early = True
                break
            if args.cooldown:
                time.sleep(args.cooldown)

    # ---- report
    bad = [r for r in rows if r["status"] in BAD_STATUSES]
    if bad:
        failed_ids, seen_bad = [], set()
        for r in bad:
            for nid in str(r["nodeid"]).split(";"):
                if nid not in seen_bad:
                    seen_bad.add(nid)
                    failed_ids.append(nid)
        (run_dir / "failed_nodeids.txt").write_text("\n".join(failed_ids) + "\n", encoding="utf-8")

    total_time = time.monotonic() - t_start
    print("\n" + "=" * 100)
    print(f"ISOLATED RUN SUMMARY  ({len(rows)} of {len(chunks)} process(es) executed in {total_time / 60:.1f} min)")
    print("=" * 100)
    counts = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    for status, n in sorted(counts.items()):
        print(f"  {status:<14} {n}")
    if bad:
        print("\nCases that did not pass:")
        for r in bad:
            print(f"  [{r['index']:>4}] {r['status']:<12} rc={r['returncode']:<5} {r['nodeid']}")
            print(f"         {run_dir / r['log']}")
        print(f"\n  Rerun just these:\n"
              f"    python {tool_hint(root)} --run --from-file {run_dir / 'failed_nodeids.txt'}")

    peaks = [r["peak_rss_mb"] for r in rows if isinstance(r["peak_rss_mb"], float)]
    if peaks:
        worst = max(rows, key=lambda r: r["peak_rss_mb"] if isinstance(r["peak_rss_mb"], float) else -1)
        print(f"\nPeak RSS: max {max(peaks):.0f} MB, median {sorted(peaks)[len(peaks) // 2]:.0f} MB")
        print(f"  worst case: {worst['nodeid']}")
    avail_end, _ = sys_mem()
    if avail_at_start is not None and avail_end is not None:
        delta = avail_end - avail_at_start
        print(f"\nSystem free memory: {avail_at_start:.0f} MB at start -> {avail_end:.0f} MB at end "
              f"({delta:+.0f} MB).")
        print("  Every case ran in its own process, so a large drop here means memory is being retained "
              "outside the pytest process (GPU driver, page cache, other apps), not by a leak that process "
              "teardown would fix. Look at the sys_avail_* columns in summary.csv for the trend.")
    print(f"\nsummary.csv : {summary_path}")
    print(f"logs        : {run_dir / 'logs'}")

    if stopped_early:
        return 2
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
