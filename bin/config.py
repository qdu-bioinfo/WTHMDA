import argparse

def parse_args():
    parser = argparse.ArgumentParser(description = 'WTHMDA')
    
    ## Data settings
    parser.add_argument('--mis', type=str, default='../data/example/microbe_I_S.csv')
    parser.add_argument('--dis', type=str, default='../data/example/disease_I_S.csv')
    parser.add_argument('--mlist', type=str, default='../data/example/microbe_list.xlsx')
    parser.add_argument('--train', type=str, default='../data/example/train_edge.csv')
    parser.add_argument('--test', type=str, default='../data/example/test_edge.csv')
    parser.add_argument('--o', type=str, default='../result/')   
    
    
    
    return parser.parse_args()