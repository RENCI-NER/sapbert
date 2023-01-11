import pandas as pd
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process arguments.')
    parser.add_argument('--input_file_dir', type=str,
                        default='/projects/babel/sapbert-training/2022dec2-2-lc/compendia/',
                        help='input file directory to concatenate with input_file_list')
    parser.add_argument('--input_file_list', type=list, default=[
        'AnatomicalEntity.txt', 'GrossAnatomicalStructure.txt', 'ComplexMolecularMixture.txt',
        'ChemicalMixture.txt', 'Polypeptide.txt', 'Cell.txt',
        'CellularComponent.txt', 'BiologicalProcess.txt', 'Pathway.txt', 'MolecularActivity.txt',
        'PhenotypicFeature.txt', 'Disease.txt', 'umls.txt', 'OrganismTaxon.txt', 'ChemicalEntity.txt',
        'MolecularMixture.txt', 'Gene.txt', 'SmallMolecule.txt', 'Protein.txt'
    ], help='input file list to process')
    parser.add_argument('--output_dir', type=str,
                        default='/projects/ner/software/sapbert/sapbert/data/babel',
                        help='output path for putting processed sapbert training data')

    args = parser.parse_args()
    input_file_dir = args.input_file_dir
    input_file_list = args.input_file_list
    output_dir = args.output_dir

    for f in input_file_list:
        df = pd.read_csv(os.path.join(input_file_dir, f), sep='\|\|', header=None, usecols=[1, 2, 3])
        # since pandas does not support multiple character separator, cannot directly use to_csv to
        # write data frame to csv with separator ||
        row_series = df[df.columns].astype(str).apply(lambda x: '||'.join(x), axis=1)
        row_series.to_csv(os.path.join(output_dir, f),
                          header=False, sep='\t', index=False)
