"""Biomarker analysis pipeline: runs all cohort x estimand combinations.

Stages:
  1. ICI_LRs          — propensity score generation (all_ICI, first_line)
  2. generate_IPTW_df  — build IPTW datasets per cohort
  3. run_IPTW_analysis — Cox models for each cohort x estimand (ATT, ATE)

Usage:
  python run_biomarker_pipeline.py              # run all stages
  python run_biomarker_pipeline.py --stages 2 3 # skip stage 1
"""

import argparse
import subprocess
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

COHORTS = ['all_ICI', 'first_line']
ESTIMANDS = ['ATT', 'ATE']


def run_commands(cmds, stage_label):
    """Run a list of (label, cmd) tuples in parallel, wait for all, abort on failure."""
    procs = []
    for label, cmd in cmds:
        print(f"  Starting: {label}")
        proc = subprocess.Popen(cmd, cwd=SCRIPT_DIR)
        procs.append((label, proc))

    failed = []
    for label, proc in procs:
        proc.wait()
        if proc.returncode != 0:
            failed.append(label)
            print(f"  FAILED: {label} (exit code {proc.returncode})")
        else:
            print(f"  Done: {label}")

    if failed:
        print(f"\n{stage_label} failed for: {', '.join(failed)}")
        sys.exit(1)
    print(f"{stage_label} complete.\n")


def stage_1():
    """Generate propensity scores for each cohort."""
    print("=== Stage 1: Propensity score generation ===")
    cmds = [
        ('all_ICI propensity', [sys.executable, 'ICI_LRs_all_ICI.py']),
        ('first_line propensity', [sys.executable, 'ICI_LRs_first_line.py']),
    ]
    run_commands(cmds, 'Stage 1')


def stage_2():
    """Generate IPTW datasets for each cohort."""
    print("=== Stage 2: IPTW dataset generation ===")
    cmds = [
        (f'{cohort} IPTW dataset', [sys.executable, 'generate_IPTW_df.py', '--cohort', cohort])
        for cohort in COHORTS
    ]
    run_commands(cmds, 'Stage 2')


def stage_3():
    """Run IPTW Cox model analyses for all cohort x estimand combinations."""
    print("=== Stage 3: IPTW Cox model analysis ===")
    cmds = [
        (f'{cohort} x {estimand}',
         [sys.executable, 'run_IPTW_analysis.py', '--cohort', cohort, '--estimand', estimand])
        for cohort in COHORTS
        for estimand in ESTIMANDS
    ]
    run_commands(cmds, 'Stage 3')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run biomarker analysis pipeline.')
    parser.add_argument('--stages', nargs='+', type=int, default=[1, 2, 3],
                        choices=[1, 2, 3], help='Which stages to run (default: all)')
    args = parser.parse_args()

    stage_fns = {1: stage_1, 2: stage_2, 3: stage_3}

    print("=== Biomarker Analysis Pipeline ===")
    print(f"Stages: {args.stages}")
    print(f"Cohorts: {COHORTS}")
    print(f"Estimands: {ESTIMANDS}")
    print()

    for s in sorted(args.stages):
        stage_fns[s]()

    print("=== Pipeline complete ===")
