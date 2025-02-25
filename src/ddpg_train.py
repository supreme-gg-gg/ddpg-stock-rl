"""
Use DDPG to train a stock trader based on a window of history price
"""

from __future__ import print_function, division

import numpy as np
import argparse
import pprint

from model.ddpg.actor import StockActor, StockActorPVM
from model.ddpg.critic import StockCritic, StockCriticPVM
from model.ddpg.ddpg import DDPGAgent
from model.ddpg.ornstein_uhlenbeck import OrnsteinUhlenbeckActionNoise

from environment.portfolio import PortfolioEnv
from utils.data import read_stock_history, obs_normalizer
from utils.helpers import get_model_path, get_result_path

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Provide arguments for training different DDPG models')

    parser.add_argument('--debug', '-d', help='print debug statement', default=False)
    parser.add_argument('--predictor_type', '-p', help='cnn or lstm predictor', required=True)
    parser.add_argument('--window_length', '-w', help='observation window length', required=True)
    parser.add_argument('--batch_norm', '-b', help='whether to use batch normalization', required=True)
    parser.add_argument('--pvm', '-m', help='whether to use PVM', required=True)

    args = vars(parser.parse_args())

    pprint.pprint(args)

    if args['debug'] == 'True':
        DEBUG = True
    else:
        DEBUG = False

    # load the data
    history, abbreviation = read_stock_history(filepath='utils/datasets/stocks_history_target.h5')
    history = history[:, :, :4]
    target_stocks = abbreviation
    num_training_time = 1095
    window_length = int(args['window_length'])
    nb_classes = len(target_stocks) + 1
    pvm = args['pvm']

    # get target history
    target_history = np.empty(shape=(len(target_stocks), num_training_time, history.shape[2]))
    for i, stock in enumerate(target_stocks):
        target_history[i] = history[abbreviation.index(stock), :num_training_time, :]
    print(target_history.shape)

    # setup environment
    env = PortfolioEnv(target_history, target_stocks, steps=1000, window_length=window_length, pvm=pvm)

    # do some setup and checking
    action_dim = [nb_classes]
    state_dim = [nb_classes, window_length]
    batch_size = 64
    action_bound = 1.
    tau = 1e-3
    assert args['predictor_type'] in ['cnn', 'lstm'], 'Predictor must be either cnn or lstm'
    predictor_type = args['predictor_type']
    if args['batch_norm'] == 'True':
        use_batch_norm = True
    elif args['batch_norm'] == 'False':
        use_batch_norm = False
    else:
        raise ValueError('Unknown batch norm argument')
    model_save_path = get_model_path(window_length, predictor_type, use_batch_norm)
    summary_path = get_result_path(window_length, predictor_type, use_batch_norm)
    
    # create actor critic noise and agent
    actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

    if pvm:
        actor = StockActorPVM(state_dim, action_dim, action_bound, 1e-4, tau, batch_size,
                            predictor_type, use_batch_norm)
        critic = StockCriticPVM(state_dim=state_dim, action_dim=action_dim, learning_rate=1e-3, tau=1e-3,
                             predictor_type=predictor_type, use_batch_norm=use_batch_norm)
    else:
        actor = StockActor(state_dim, action_dim, action_bound, 1e-4, tau, batch_size,
                            predictor_type, use_batch_norm)
        critic = StockCritic(state_dim=state_dim, action_dim=action_dim, learning_rate=1e-3, tau=1e-3,
                                predictor_type=predictor_type, use_batch_norm=use_batch_norm)
    
    # Initalize the model (no need to load weight unless using checkpoints)
    ddpg_model = DDPGAgent(env, actor, critic, 0.95, actor_noise, obs_normalizer=obs_normalizer,
                        config_file='config/stock.json', model_save_path=model_save_path,
                        summary_path=summary_path)
    ddpg_model.train() # this saves automatically

    # TODO: Check if action_processor is necessary, it is NOT used in the original code but
    # seems to have support built-in in the networks.

