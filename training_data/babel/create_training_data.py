import argparse
import os
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process arguments.')
    parser.add_argument('--input_file_dir', type=str,
                        default='/projects/babel/babel-outputs/2026jul22/sapbert-training-data',
                        help='input file directory to concatenate with input_file_list')
    parser.add_argument('--input_file_list', type=list, default=[
        'umls.txt', 'OrganismTaxon.txt', 'MolecularActivity.txt', 'BiologicalProcess.txt', 'Pathway.txt', 'AnatomicalEntity.txt',
        'Cell.txt', 'PhenotypicFeature.txt', 'Disease.txt', 'GeneProteinConflated.txt', 'CellLine.txt', 'MacromolecularComplex.txt',
        'Gene.txt', 'Protein.txt', 'DrugChemicalConflated.txt', 'GrossAnatomicalStructure.txt', 'CellularComponent.txt', 'GeneFamily.txt'
    ], help='input file list to process')
    parser.add_argument('--output_dir', type=str,
                        default='/projects/ner/software/sapbert/sapbert/data/babel/2026jul22',
                        help='output path for putting processed sapbert training data')
    parser.add_argument('--dedup_scope', type=str, choices=['none', 'file', 'global'], default='file',
                        help="scope for removing duplicate (NAME1, NAME2) synonym pairs: "
                             "'none' = don't dedup pairs (only the existing same-name filter runs), "
                             "'file' = drop duplicate pairs within each file (default), "
                             "'global' = drop duplicate pairs across all files (first occurrence wins)")
    args = parser.parse_args()
    input_file_dir = args.input_file_dir
    input_file_list = args.input_file_list
    output_dir = args.output_dir
    dedup_scope = args.dedup_scope

    # only used when dedup_scope == 'global': persists across the file loop so a pair
    # seen in an earlier file is dropped from every later file it appears in.
    # NOTE: processing order in input_file_list matters for 'global' scope, since the
    # FIRST file to contain a pair is the one that keeps it; later files have it dropped.
    seen_pairs_global = set()

    for f in input_file_list:
        print(f'processing {f}', flush=True)
        # the columns of the input data are biolink curie || id || name || name1 || name2 where name is the
        # canonical label to create name to id pairs for sapbert predictions
        df = pd.read_csv(os.path.join(input_file_dir, f), sep='\|\|', header=None,
                         usecols=[1, 3, 4], names=["ID", "NAME1", "NAME2"], dtype=str, engine='python')
        # the two statements below will not be needed here next time when babel input data is updated since
        # they will be taken care of on the babel side
        df['NAME1'] = df['NAME1'].apply(lambda x: str(x).strip().lower())
        df['NAME2'] = df['NAME2'].apply(lambda x: str(x).strip().lower())
        # filter out those rows where the synonym pairs are the same with case-insensitive comparison
        df = df[df.NAME1 != df.NAME2]

        # remove duplicate synonym pairs, scoped to this file or across all files.
        # (a, b) and (b, a) are treated as the same pair (order-insensitive) since this
        # data is used for SapBERT fine-tuning, where the pair direction doesn't matter.
        if dedup_scope in ('file', 'global'):
            before = len(df)
            seen_pairs_local = set()  # always reset per file; used directly for 'file' scope
            keep_mask = []
            for n1, n2 in zip(df['NAME1'], df['NAME2']):
                pair_key = frozenset((n1, n2))
                seen_set = seen_pairs_global if dedup_scope == 'global' else seen_pairs_local
                if pair_key in seen_set:
                    keep_mask.append(False)
                else:
                    seen_set.add(pair_key)
                    keep_mask.append(True)
            df = df[keep_mask]
            scope_desc = 'across all files so far' if dedup_scope == 'global' else 'within this file'
            print(f'  dropped {before - len(df)} duplicate pairs ({scope_desc})', flush=True)

        if df.empty:
            print(f'data {f} is empty after filtering out synonym pairs with the same names', flush=True)
        else:
            # since pandas does not support multiple character separator, cannot directly use to_csv to
            # write data frame to csv with separator ||
            row_series = df[df.columns].astype(str).apply(lambda x: '||'.join(x), axis=1)
            row_series.to_csv(os.path.join(output_dir, f),
                              header=False, sep='\t', index=False)
