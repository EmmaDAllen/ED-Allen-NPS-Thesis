"""Audit SPIP evaluation CSVs and build a JMP-ready master table.

Run from the repository root:
    python audit_thesis_results.py

Dependencies: pandas (pip install pandas)
Original files in results/ are never changed.
"""
from itertools import product
from pathlib import Path
import re
import pandas as pd

RESULTS_DIR = Path('results')
OUTPUT_DIR = Path('analysis')
EXPECTED_SEEDS = (1, 2, 3, 4, 5)
MODELS = ('edge_transformer', 'tropical_v2', 'gnn', 'tropical', 'transformer')
TRAINING_DISTRIBUTIONS = (
    'mixed_55_15_filtered_reps100',
    'mixed_55_15_reps100',
    'mixed_60_10_reps100',
    'mixed_filtered_graphloss',
    'mixed_filtered',
    'onein_filtered',
    'onein_reps100',
)
EVALUATIONS = ('wood_exp', 'value_ood', 'ood_size', 'id_new', 'external', 'wood')
PREFIX = 'evaluation_results_'



def parse_filename(path):
    """Parse final SPIP experiments and classify historical files."""
    name = path.stem

    if not name.startswith(PREFIX):
        return None, 'unexpected prefix'

    rest = name[len(PREFIX):]

    model = next(
        (x for x in MODELS if rest.startswith(x + '_')),
        None
    )
    if model is None:
        return None, 'unrecognized model'

    rest = rest[len(model) + 1:]

    if not rest.startswith('shortest_path_'):
        return None, 'not a shortest_path result'

    rest = rest[len('shortest_path_'):]

    evaluation = next(
        (x for x in EVALUATIONS if rest.startswith(x + '_')),
        None
    )
    if evaluation is None:
        return None, 'unrecognized evaluation'

    rest = rest[len(evaluation) + 1:]

    # Remove candidate configuration first.
    candidate_config = 'default'
    if rest.endswith('_candidate4'):
        rest = rest[:-len('_candidate4')]
        candidate_config = 'candidate4'

    # Evaluation condition.
    condition = 'unspecified'
    for suffix, label in (
        ('_eval_unfiltered', 'unfiltered'),
        ('_eval_filtered', 'filtered')
    ):
        if rest.endswith(suffix):
            rest = rest[:-len(suffix)]
            condition = label
            break

    # Match known distributions before interpreting seed suffixes.
    distribution = None
    seed = 1

    for dist in TRAINING_DISTRIBUTIONS:
        if rest == dist:
            distribution = dist
            break

        match = re.fullmatch(
            re.escape(dist) + r'_v([1-9][0-9]*)',
            rest
        )
        if match:
            distribution = dist
            seed = int(match.group(1))
            break

    if distribution is None:
        return None, 'historical_or_unclassified: ' + rest

    experiment_group = (
        'primary'
        if model in ('tropical_v2', 'edge_transformer', 'gnn')
        else 'supplementary'
    )

    return dict(
        model=model,
        training_distribution=distribution,
        training_seed=seed,
        evaluation=evaluation,
        evaluation_condition=condition,
        candidate_config=candidate_config,
        experiment_group=experiment_group
    ), None



def main():
    if not RESULTS_DIR.is_dir():
        raise SystemExit('Cannot find results/. Run this script from the thesis repository root.')
    OUTPUT_DIR.mkdir(exist_ok=True)
    # Search both final and historical evaluation folders.
    final_dir = RESULTS_DIR / 'final_results'
    old_dir = RESULTS_DIR / 'old_results'

    paths = sorted(
        list(final_dir.glob('evaluation_results_*.csv')) +
        list(old_dir.glob('evaluation_results_*.csv')))

    inventory, problems, frames = [], [], []
    reference_columns = None
    for path in paths:
        meta, reason = parse_filename(path)
        if reason:
            inventory.append({'filename': path.name, 'result_folder': path.parent.name,
            'status': 'not_imported', 'reason': reason})
            continue
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception as exc:
            inventory.append({'filename': path.name, 'result_folder': path.parent.name, 'status': 'read_error', 'reason': str(exc)})
            problems.append({'filename': path.name, 'result_folder': path.parent.name, 'issue': 'read_error', 'details': str(exc)})
            continue
        columns = set(df.columns)
        if reference_columns is None:
            reference_columns = columns
        if columns != reference_columns:
            problems.append({'filename': path.name, 'issue': 'column_mismatch',
                             'details': 'missing=' + str(sorted(reference_columns - columns)) +
                                        '; extra=' + str(sorted(columns - reference_columns))})
        if df.empty:
            problems.append({'filename': path.name, 'issue': 'empty_csv', 'details': 'zero data rows'})
        for col, expected in (('model_type', meta['model']), ('eval_mode', meta['evaluation'])):
            if col in df.columns:
                actual = set(df[col].dropna().astype(str).unique())
                if actual and actual != {expected}:
                    problems.append({'filename': path.name, 'issue': 'filename_content_mismatch',
                                     'details': f'{col}: filename={expected}, values={sorted(actual)}'})
        # Key used only within each CSV: graph_seed and attack_limit are
        # expected to distinguish graph-budget observations for these datasets.
        key = ['graph_seed', 'attack_limit']
        if all(k in df.columns for k in key):
            # Some datasets may legitimately reuse a seed for distinct OD pairs.
            for extra in ('source_original', 'sink_original', 'source', 'sink', 'replication',
                          'wood_rows', 'wood_cols', 'network_name'):
                if extra in df.columns and df[extra].notna().any():
                    key.append(extra)
            dup_count = int(df.duplicated(subset=key).sum())
            if dup_count:
                problems.append({'filename': path.name, 'issue': 'possible_duplicate_rows',
                                 'details': f'{dup_count} duplicates on {key}'})

        inventory.append({'filename': path.name, 'result_folder': path.parent.name, **meta,
        'status': 'imported', 'reason': '','rows': len(df),'columns': len(df.columns)})

        for k, v in meta.items():
            df[k if k not in df.columns else 'file_' + k] = v
        df['source_file'] = path.name
        df['result_folder'] = path.parent.name
        frames.append(df)

    inv = pd.DataFrame(inventory)
    inv.to_csv(OUTPUT_DIR / 'file_inventory.csv', index=False)
    pd.DataFrame(problems, columns=['filename', 'issue', 'details']).to_csv(
        OUTPUT_DIR / 'data_quality_issues.csv', index=False)

    # A seed is expected only where a model/distribution/evaluation/condition
    # combination exists. This avoids inventing experimental configurations.
    # Only primary robustness experiments require five seeds.
    # Check all expected final five-seed experiments.
    from itertools import product

    missing = []

    imported = inv[inv['status'].eq('imported')].copy()

    final_models = ('tropical_v2', 'edge_transformer', 'gnn')

    expected_evaluations = [
        ('id_new', 'unfiltered'),
        ('value_ood', 'unfiltered'),
        ('ood_size', 'unfiltered'),
        ('id_new', 'filtered'),
        ('value_ood', 'filtered'),
        ('ood_size', 'filtered'),
        ('wood', 'unspecified'),
        ('external', 'unspecified'),
    ]

    primary = imported[
        (imported['model'].isin(final_models)) &
        (imported['candidate_config'] == 'default')
    ]

    robustness_distributions = tuple(dist for dist in TRAINING_DISTRIBUTIONS if dist != 'mixed_filtered_graphloss')

    for model, distribution, seed, eval_pair in product(
        final_models,
        robustness_distributions,
        EXPECTED_SEEDS,
        expected_evaluations):

        evaluation, condition = eval_pair

        match = primary[
            (primary['model'] == model) &
            (primary['training_distribution'] == distribution) &
            (primary['training_seed'] == seed) &
            (primary['evaluation'] == evaluation) &
            (primary['evaluation_condition'] == condition)
        ]

        if match.empty:
            missing.append({
                'model': model,
                'training_distribution': distribution,
                'training_seed': seed,
                'evaluation': evaluation,
                'evaluation_condition': condition,
                'issue': 'missing_expected_file'
            })

    pd.DataFrame(
        missing,
        columns=[
            'model',
            'training_distribution',
            'training_seed',
            'evaluation',
            'evaluation_condition',
            'issue'
        ]
    ).to_csv(
        OUTPUT_DIR / 'missing_seeds.csv',
        index=False
    )



    if frames:
        master = pd.concat(frames, ignore_index=True, sort=False)
        # Extra metadata columns are placed first for convenient JMP grouping.
        metadata = [
        'model',
        'training_distribution',
        'training_seed',
        'evaluation',
        'evaluation_condition',
        'result_folder',
        'source_file']
        master = master[metadata + [c for c in master.columns if c not in metadata]]
        master.to_csv(OUTPUT_DIR / 'jmp_master_results_with_graphloss.csv',
    index=False
)
        print(f'Imported {len(frames):,} files and {len(master):,} observations.')
    else:
        print('No recognized SPIP evaluation files imported.')
    print(f'Files found: {len(paths):,}')
    print(f'Unimported files: {int((inv.status != "imported").sum()) if not inv.empty else 0:,}')
    print(f'Missing seed entries (review): {len(missing):,}')
    print(f'Data quality flags (review): {len(problems):,}')
    print(f'Reports saved to: {OUTPUT_DIR.resolve()}')
    print('Review file_inventory.csv, missing_seeds.csv, and data_quality_issues.csv before using the master file.')


if __name__ == '__main__':
    main()
