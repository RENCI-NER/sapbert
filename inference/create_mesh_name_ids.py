from mesh import MESH
import pandas as pd
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process arguments.')
    parser.add_argument('--MESH_FILE_PATH', type=str, default='../data/mesh_desc_2022.xml',
                        help='MESH terms and ids downloaded from '
                             'https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/')
    parser.add_argument('--OUTPUT_FILE', type=str, default='../data/pubmed_mesh_name_ids.csv',
                        help='MESH input file names and ids to be mapped to')

    
    args = parser.parse_args()
    MESH_FILE_PATH = args.MESH_FILE_PATH
    OUTPUT_FILE = args.OUTPUT_FILE

    mesh = MESH(MESH_FILE_PATH)
    mesh.load_mesh()
    all_names = [name.strip('\n') for name in mesh.names]
    all_ids = mesh.ids
    
    df = pd.DataFrame(list(zip(all_names, all_ids)), columns=['Name', 'ID'])
    df.to_csv(OUTPUT_FILE, index=False)
