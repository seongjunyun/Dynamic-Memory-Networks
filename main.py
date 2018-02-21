import os
import argparse
from solver import Solver
import numpy as np

def main(config):

    # Create directories if not exist
    if not os.path.exists(config.model_save_path):
        os.makedirs(config.model_save_path)

    for i in np.arange(5,21):
        config.qa_type = str(i)

        # Solver
        solver = Solver(config)

        if config.mode == 'train':
            solver.train()
        elif config.mode == 'test':
            solver.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #Dataset
    parser.add_argument('--qa_type', type=str, default='1')

    #Model hyper-parameters
    parser.add_argument('--hidden_size', type=int, default=100)
    parser.add_argument('--input_size', type=int, default=100)

    # Hyper-parameters
    parser.add_argument('--lr', type=float, default=0.001)

    # Training settings
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--num_epochs', type=int, default=70)

    # Test settings
    parser.add_argument('--test_model', type=str, default='best_model_1.pth')

    # Path
    parser.add_argument('--model_save_path', type=str, default='./models')

    # Step size
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--model_save_step', type=int, default=50)
    
    # main
    parser.add_argument('--mode', type=str, default='train', choices=['train','test'])

    config = parser.parse_args()
    print(config)
    main(config)
