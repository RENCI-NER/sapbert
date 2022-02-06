from tqdm import tqdm
import pandas as pd
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process arguments.')
    parser.add_argument('--input_file', type=str, default='../data/pubmed_split_11_prediction_output.csv',
                        help='input file for comparison between prediction columns')

    args = parser.parse_args()
    input_file = args.input_file

    df = pd.read_csv(input_file, usecols=['Labelled MESH ID', 'Predicted MESH ID'])
    df['Prediction_agreement'] = df.apply(lambda x: x['Predicted MESH ID'] in x['Labelled MESH ID'], axis=1)
    agree_df = df[df.Prediction_agreement == True]
    disagree_df = df[df.Prediction_agreement == False]
    print(f'agreement percentage: {len(agree_df)/len(df)}')
