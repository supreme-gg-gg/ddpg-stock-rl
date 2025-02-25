"""
Helper functions migrated from the original stock_trading.py in src.
This helps modularise usage to group all helper functions under utils.
"""

import numpy as np

def get_model_path(window_length, predictor_type, use_batch_norm, pvm=False):
    if use_batch_norm:
        batch_norm_str = 'batch_norm'
    else:
        batch_norm_str = 'no_batch_norm'
    if pvm:
        pvm_str = 'pvm'
    else:
        pvm_str = 'no_pvm'
    return 'weights/stock/{}/window_{}/{}/{}/checkpoint.ckpt'.format(predictor_type, window_length, batch_norm_str, pvm_str)


def get_result_path(window_length, predictor_type, use_batch_norm, pvm=False):
    if use_batch_norm:
        batch_norm_str = 'batch_norm'
    else:
        batch_norm_str = 'no_batch_norm'
    if pvm:
        pvm_str = 'pvm'
    else:
        pvm_str = 'no_pvm'
    return 'results/stock/{}/window_{}/{}/{}/'.format(predictor_type, window_length, batch_norm_str, pvm_str)

# NOTE: This function is deprecated, variable scope is no longer used in PyTorch, this is old TensorFlow code
# def get_variable_scope(window_length, predictor_type, use_batch_norm):
#     if use_batch_norm:
#         batch_norm_str = 'batch_norm'
#     else:
#         batch_norm_str = 'no_batch_norm'
#     return '{}_window_{}_{}'.format(predictor_type, window_length, batch_norm_str)

def test_model(env, model):
    observation, info = env.reset()
    done = False
    while not done:
        action = model.predict_single(observation)
        observation, _, done, _ = env.step(action)
    env.render()

def test_model_pvm(env, model):
    (observation, prev_weights), info = env.reset()
    done = False
    while not done:
        action = model.predict_single(observation, prev_weights)
        (observation, prev_weights), _, done, info = env.step(action)
    env.render()

def test_model_multiple(env, models):
    observation, info = env.reset()
    done = False
    while not done:
        actions = []
        for model in models:
            actions.append(model.predict_single(observation))
        actions = np.array(actions)
        observation, _, done, info = env.step(actions)
    env.render()

def peek_stock_data():
    """you should have h5py installed to preview the data"""
    # Open the H5 file
    import h5py
    with h5py.File('utils/datasets/stocks_history_target_2.h5', 'r') as f:
        # Print the keys in the file
        print("Keys in the H5 file:", list(f.keys()))
        
        # Get the stock data and abbreviations
        stock_data = f['history'][:]  # Assuming 'stock_data' is the dataset name
        stock_abbr = f['abbreviation'][:] # Assuming 'abbreviation' is the dataset name
        
        # Convert byte strings to regular strings if needed
        stock_abbr = [abbr.decode('utf-8') if isinstance(abbr, bytes) else abbr for abbr in stock_abbr]
        
        # Print the shape of the data
        print("\nData shape:", stock_data.shape)
        print("Stock abbreviations:", stock_abbr)
        
        # Show first few entries
        print("\nFirst few entries of the stock data:")
        # Assuming the data structure is [stocks, time_steps, features]
        for i in range(min(5, len(stock_abbr))):
            print(f"\n{stock_abbr[i]}:")
            print(stock_data[i, :5])  # First 5 time steps