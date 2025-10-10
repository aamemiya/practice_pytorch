import copy
import random
import numpy as np
np.random.seed(1)
import pandas as pd
import os
import sys
import shutil
import argparse
import re
import helperfunctions as helpfunc
#import training_test as tntt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

parser = argparse.ArgumentParser(description = 'Optuna Experiment Controller')

parser.add_argument("--id_exp", "-id", default = "test", type =str , help = "experiment name")

parser.add_argument("--use_optuna",  default = 0, type = int, choices = [0, 1], help = "Use Optuna or not")
parser.add_argument("--num_trials", "-nt",  default = 10, type = int, help = "Num of Optuna trials")
parser.add_argument("--optuna_sql", "-osql", default = "test", type = str, help = "Optuna Study name")

parser.add_argument("--t", default = "optimize", type = str, choices = ["optimize", "best", "p_best", "p_all"], help = "Choose between optimization or training best parameter")
parser.add_argument("--recurrent", "-rec", default = 1, type = int, choices = [0, 1], help = "Use recurrent layers or not")
parser.add_argument("--epochs", "-e",  default = 200, type = int, help = "Number of epochs for training")
parser.add_argument("--netcdf_dataset", "-ncdf_loc", default = "../DATA/test/test_obs/test_da/assim.nc", type = str, help = "Location of the netCDF dataset")
parser.add_argument("--locality", "-l",  default = 1, type = int, help = "Locality size (including the main variable)")
parser.add_argument("--degree", "-d",  default = 1, type = int, help = "To make a polynomial input")
parser.add_argument("--normalized", "-norm",  default = 1, type = int, choices = [0, 1], help = "Use normalized dataset for training.")
parser.add_argument("--af_mix", "-afm",  default = 0, type = int, choices = [0, 1], help = "Use analysis forecast mixed.")
parser.add_argument("--time_splits", "-ts",  default = 5, type = int, help = "Num of RNN timesteps")
parser.add_argument("--train_batch", "-tb", default = 16384, type = int, help = "Training batch size")
parser.add_argument("--val_batch", "-vb", default = 16384, type = int, help = "Validation batch size")
parser.add_argument("--num_batches", "-nbs",  default = 1, type = int, help = "Number of training batch per epoch")
parser.add_argument("--network_type", "-ntyp",  default = "LSTM", type = str, help = "Network type")
parser.add_argument("--use_spread", "-sprd", default = 0, type =int , help = "Use ensemble spread inverse as sample_weight")
parser.add_argument("--list_layers", "-ll", default = "list.txt", type =str , help = "name of layer list file")
args = parser.parse_args()

if (args.locality > 1) and (args.degree > 1):
    parser.error('If degree is > 1 then locality has to be 1.')

def my_config(trial):
    #Parameter List
    plist = {}
    
    #Network related settings
    plist['make_recurrent'] = args.recurrent 
    plist['NN_type'] = args.network_type
    plist['use_sprd'] = args.use_spread

    if args.recurrent:
        plist['time_splits'] = args.time_splits
        print('\nNetwork is recurrent\n')
        plist['num_rnn_layers'] = 3
        plist['RNN_output'] = []
        plist['RNN_output'].append(10)
        plist['RNN_output'].append(8)
        plist['RNN_output'].append(4)
        plist['num_dense_layers'] = 1
        plist['dense_output'] = []
        plist['dense_output'].append(9)
        for i in range(plist['num_dense_layers'] - 1):
            plist['dense_output'].append(4)
    else:
        plist['time_splits'] = 1 
        print('\nNetwork is only dense\n')
        plist['num_rnn_layers'] = 0
        plist['RNN_output'] = []
        plist['num_dense_layers'] = 3
        plist['dense_output'] = []
        plist['dense_output'].append(21)
        plist['dense_output'].append(15)
        plist['dense_output'].append(12)

    plist['dense_output'].append(1)


    plist['activation'] = 'relu'
    plist['d_activation'] = 'relu'
    plist['rec_activation'] = 'sigmoid'
    plist['l2_regu'] = 1e-5
    plist['l1_regu'] = 0.0
    plist['rnn_dropout'] = 0.0
    plist['rec_rnn_dropout'] = 0.0

    #Training related settings
    plist['max_checkpoint_keep'] = 3
    plist['log_freq'] = 1
    plist['early_stop_patience'] = 400
    plist['summery_freq'] = 1
    plist['global_epoch'] = 0
    plist['global_batch_size'] = args.train_batch  
    plist['global_batch_size_v'] = args.val_batch
    plist['val_size'] = 1 * plist['global_batch_size_v']
    plist['val_min'] = 1000

    plist['lr_decay_steps'] = 1000
    plist['lr_decay_rate'] = 0
    plist['grad_mellow'] = 1
    plist['learning_rate'] = 1.0e-3
    plist['learning_rate'] = plist['learning_rate'] * plist['global_batch_size'] / (128)
    
    #Dataset and directories related settings
    plist['netCDf_loc'] = args.netcdf_dataset
    plist['optuna_db'] = args.optuna_sql
    plist['xlocal'] = 3
    plist['locality'] = args.locality
    plist['degree'] = args.degree
    plist['normalized'] = args.normalized
    plist['anal_for_mix'] = args.af_mix

    plist['experiment_name'] = args.id_exp  

    plist['experiment_dir'] = os.getcwd() + '/n_experiments/' + plist['experiment_name'] 
    plist['checkpoint_dir'] = plist['experiment_dir'] + '/checkpoint'
    plist['log_dir'] = plist['experiment_dir'] + '/log'

    plist['pickle_name'] = plist['checkpoint_dir'] + '/params.pickle'

    if os.path.isfile(plist['pickle_name']):
        print('\nPickle file exists. Reading parameter list from it.\n')
        plist = helpfunc.read_pickle(plist['pickle_name'])
    else:
        print('\nCreating experiment_dir and the corresponding sub-directories.\n')
        os.makedirs(plist['log_dir'],exist_ok=True)
        os.makedirs(plist['checkpoint_dir'],exist_ok=True)

    plist['epochs'] = args.epochs
    plist['test_num_timesteps'] = 300
    plist['num_timesteps'] = int(((plist['global_batch_size'] * args.num_batches + plist['val_size']) * plist['time_splits'])/ 8 + 100)
    
    return plist

if __name__ == "__main__":
    plist = my_config([])
    tntt.traintest(copy.deepcopy(plist))

