import pandas as pd
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process arguments.')
    parser.add_argument('--input_file_dir', type=str,
                        # default='/projects/babel/babel-outputs/2024jan9/sapbert-training-data',
                        default='/home/hongyi/sapbert/data/babel/input',
                        help='input file directory to concatenate with input_file_list')
    # parser.add_argument('--input_file_list', type=list, default=[
    #     'AnatomicalEntity.txt', 'GrossAnatomicalStructure.txt', 'Cell.txt',
    #     'CellularComponent.txt', 'BiologicalProcess.txt', 'Pathway.txt', 'MolecularActivity.txt',
    #     'PhenotypicFeature.txt', 'Disease.txt', 'umls.txt', 'OrganismTaxon.txt',
    #     'Gene.txt', 'Protein.txt', 'DrugChemicalConflated.txt', 'GeneFamily.txt'
    # ], help='input file list to process')
    parser.add_argument('--input_file_list', type=list, default=[
        'GeneFamily.txt'
    ], help='input file list to process')
    parser.add_argument('--output_dir', type=str,
                        default='/projects/ner/software/sapbert/sapbert/data/babel/2024_v2',
                        help='output path for putting processed sapbert training data')

    args = parser.parse_args()
    input_file_dir = args.input_file_dir
    input_file_list = args.input_file_list
    output_dir = args.output_dir

    for f in input_file_list:
        # the columns of the input data are biolink curie || id || name || name1 || name2 where name is the
        # canonical label to create name to id pairs for sapbert predictions
        df = pd.read_csv(os.path.join(input_file_dir, f), sep='\|\|', header=None,
                         usecols=[1, 3, 4], names=["ID", "NAME1", "NAME2"], engine='python')
        df['NAME1'] = df['NAME1'].apply(lambda x: x.strip().lower())
        df['NAME2'] = df['NAME2'].apply(lambda x: x.strip().lower())

        # filter out those rows where the synonym pairs are the same with case-insensitive comparison
        df = df[df.NAME1 != df.NAME2]

        # since pandas does not support multiple character separator, cannot directly use to_csv to
        # write data frame to csv with separator ||
        row_series = df[df.columns].astype(str).apply(lambda x: '||'.join(x), axis=1)
        row_series.to_csv(os.path.join(output_dir, f),
                          header=False, sep='\t', index=False)
