"""Extend the existing 1,600-sample Wood-only SPIP dataset to 6,000 samples.

Run from repository root:
  PYTHONPATH=. python -u data/generate_wood_only_6000.py

Copies existing samples unchanged, then generates missing grid realizations.
Writes a NEW output file and can resume after interruption.
"""
import json
import os
from pathlib import Path

from data.generate_wood_data import generate_wood_grid
from optimization.mip import solve_instance

GRID_SIZES = [(7, 7), (8, 8), (9, 9), (12, 12)]
PENALTY_MAX_VALUES = [5, 10]
ATTACK_BUDGETS = [5, 10]
REPS_PER_SETTING = 375  # 4 sizes x 2 penalty settings x 375 graphs x 2 budgets = 6000
BASE_SEED = 20_000_000
OLD_FILE = Path('data/training_data_shortest_path_wood_only.json')
NEW_FILE = Path('data/training_data_shortest_path_wood_only_6000.json')
SAVE_EVERY_GRAPHS = 10


def save_atomic(samples):
    temp = NEW_FILE.with_suffix('.json.tmp')
    with temp.open('w') as f:
        json.dump(samples, f)
    os.replace(temp, NEW_FILE)


def key(sample):
    return (int(sample['graph_seed']), int(sample['attack_budget']))


def expected_seed(size_index, penalty_index, rep):
    return BASE_SEED + size_index * 100_000 + penalty_index * 10_000 + rep


def validate_existing(samples):
    allowed = {}
    for size_index, (rows, cols) in enumerate(GRID_SIZES):
        for penalty_index, penalty_max in enumerate(PENALTY_MAX_VALUES):
            for rep in range(REPS_PER_SETTING):
                seed = expected_seed(size_index, penalty_index, rep)
                for budget in ATTACK_BUDGETS:
                    allowed[(seed, budget)] = (rows, cols, penalty_max)
    seen = set()
    for sample in samples:
        k = key(sample)
        if k in seen:
            raise ValueError(f'Duplicate sample key: {k}')
        if k not in allowed:
            raise ValueError(f'Unexpected sample key {k}; check seed formula')
        rows, cols, penalty_max = allowed[k]
        if (int(sample['rows']), int(sample['cols']), int(sample['penalty_max'])) != (rows, cols, penalty_max):
            raise ValueError(f'Configuration mismatch for {k}')
        if sample.get('path_length') is None or float(sample['path_length']) <= 0:
            raise ValueError(f'Invalid objective for {k}')
        seen.add(k)
    return seen


def main():
    NEW_FILE.parent.mkdir(parents=True, exist_ok=True)
    if NEW_FILE.exists():
        with NEW_FILE.open() as f:
            samples = json.load(f)
        print(f'Resuming {NEW_FILE} with {len(samples)} samples', flush=True)
    else:
        if not OLD_FILE.exists():
            raise FileNotFoundError(f'Missing original dataset: {OLD_FILE}')
        with OLD_FILE.open() as f:
            samples = json.load(f)
        print(f'Copying {len(samples)} original samples without re-solving', flush=True)
        validate_existing(samples)
        save_atomic(samples)
    done = validate_existing(samples)
    skipped = 0
    new_graphs_since_save = 0

    for size_index, (rows, cols) in enumerate(GRID_SIZES):
        for penalty_index, penalty_max in enumerate(PENALTY_MAX_VALUES):
            for rep in range(REPS_PER_SETTING):
                seed = expected_seed(size_index, penalty_index, rep)
                missing = [k for k in ATTACK_BUDGETS if (seed, k) not in done]
                if not missing:
                    continue
                G, s, t, density = generate_wood_grid(
                    rows=rows, cols=cols, cost_max=10,
                    delay_max=penalty_max, resource_max=1, seed=seed)
                for budget in missing:
                    sample = solve_instance(G=G, s=s, t=t, density=density,
                        attack_limit=budget, problem_type='shortest_path', flow_demand=1)
                    if sample is None:
                        skipped += 1
                        print(f'SKIPPED size={rows}x{cols} pmax={penalty_max} K={budget} seed={seed}', flush=True)
                        continue
                    sample.update({'graph_seed': seed, 'replication': rep,
                        'attack_budget': budget, 'problem_type': 'shortest_path',
                        'topology': 'wood', 'rows': rows, 'cols': cols,
                        'penalty_max': penalty_max})
                    samples.append(sample)
                    done.add((seed, budget))
                    print(f'Solved {len(samples)}/6000 size={rows}x{cols} pmax={penalty_max} '
                          f'K={budget} seed={seed} objective={sample["path_length"]:.3f} '
                          f'mip_solve_time={sample.get("mip_solve_time", float("nan")):.4f}s', flush=True)
                new_graphs_since_save += 1
                if new_graphs_since_save >= SAVE_EVERY_GRAPHS:
                    save_atomic(samples)
                    new_graphs_since_save = 0
                    print(f'Checkpointed {len(samples)}/6000', flush=True)
    save_atomic(samples)
    print(f'Completed: {len(samples)}/6000 samples, skipped={skipped}, path={NEW_FILE}', flush=True)
    if len(samples) != 6000:
        raise RuntimeError(f'Expected 6000 solved samples; found {len(samples)}')


if __name__ == '__main__':
    main()
