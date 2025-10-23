import torch
from torch import nn
import numpy as np
from netCDF4 import Dataset
import time
import math
import os
import sys
import re

import helperfunctions as helpfunc
import network_arch as net

import datetime

from plotting_single import plot_history, sample_scatter, scatter_plot

from collections import OrderedDict

def train(plist, model, checkpoint, optimizer, train_dataloader, val_dataloader, a_f, s_f):
   
    ename = re.search('DATA/(.+?)/', plist['netCDf_loc'])
    rname = plist['experiment_name'] 

    def compute_loss(labels, predictions, accuracy):
       per_example_loss = nn.functional.mse_loss(predictions, labels, weight=accuracy)
       return per_example_loss 
 
    def compute_metric(labels, predictions, accuracy):
       per_example_metric = torch.sum(torch.square(torch.sub(labels*accuracy, predictions*accuracy)))
       return per_example_metric 


    def train_step(inputs):
       local_forecast, analysis, accuracy = inputs
 
      # print("CHECK forecast dtype")
      # print(local_forecast.dtype)
#       quit()
      #pred_analysis, _ = model(local_forecast, stat = [])
       pred_analysis = model(local_forecast.to(torch.float32))
       #Calculating relative loss
       try:
          loss = compute_loss(analysis[:, -1, :], pred_analysis, accuracy[:, -1, :]) * plist['grad_mellow'] * (1.0 / (plist['global_batch_size']))
       except:
          loss = compute_loss(analysis, pred_analysis, accuracy) * plist['grad_mellow'] * (1.0 / (plist['global_batch_size']))
       loss.backward()
       optimizer.step()
       optimizer.zero_grad()

       try:
          metric = (compute_metric(analysis[:, -1, :], pred_analysis, accuracy[:, -1, :])) * (1.0 / (plist['global_batch_size']))
       except:
          metric = (compute_metric(analysis, pred_analysis, accuracy)) * (1.0 / (plist['global_batch_size']))
  
      # print("return ")
      # print("lossn ",loss)
      # print("metric " ,metric)
      # quit()

       return loss, metric

    def val_step(inputs):
       local_forecast, analysis, accuracy = inputs
       #pred_analysis, _ = model(local_forecast, stat = [])
       pred_analysis = model(local_forecast.to(torch.float32))
#       print("val input",local_forecast[:])
#       print("val pred",pred_analysis[:])
#       print("pred max",torch.max(pred_analysis))
#       print("pred min",torch.min(pred_analysis))
       try:
          loss = compute_loss(analysis[:, -1, :], pred_analysis, accuracy[:, -1, :]) * plist['grad_mellow'] * (1.0 / (plist['global_batch_size_v']))
       except:
          loss = compute_loss(analysis, pred_analysis, accuracy) * plist['grad_mellow'] * (1.0 / (plist['global_batch_size_v']))
       try:
          metric = (compute_metric(analysis[:, -1, :], pred_analysis, accuracy[:, -1, :])) * (1.0 / (plist['global_batch_size_v']))
       except:
          metric = (compute_metric(analysis, pred_analysis, accuracy)) * (1.0 / (plist['global_batch_size_v']))
       return loss, metric 



    #Initialing training variables
    global_step = 0
    global_step_val = 0
    val_min = 0
    val_loss_min = plist['val_min']
    timer_tot = time.time()

    epoch_nums=[]
    losses=[]
    t_metrics=[]
    v_metrics=[]

    #Starting training
    epochs = plist['epochs']

    for epoch in range(epochs):

                    start_time = time.time()

                    plist['global_epoch'] += 1

#                    print('\nStart of epoch %d' %(plist['global_epoch']))
                    model.train()

                    # Iterate over the batches of the dataset.
                    for step, inputs in enumerate(train_dataloader):
                        global_step += 1

                        # Open a GradientTape to record the operations run
                        # during the forward pass, which enables autodifferentiation.
                        loss, t_metric = train_step(inputs)

                        #if (step % plist['log_freq']) == 0:
                        #    print('Training loss (for one batch) at step %s: %s' % (step+1, float(loss)))
                            
                    # Display metrics at the end of each epoch.
#                    print('\nTraining loss at epoch end {}'.format(loss))
#                    print('Training acc over epoch: %s ' % (float(t_metric)))
                    #print('Seen so far: %s samples\n' % ((global_step) * plist['global_batch_size']))
      
                    model.eval()

                    #Code for validation at the end of each epoch
                    for step_val, val_inputs in enumerate(val_dataloader):

                        global_step_val += 1

                        val_loss, v_metric = val_step(val_inputs)

                        #if (step_val % plist['log_freq']) == 0:
                        #    print('Validation loss (for one batch) at step {}: {}'.format(step_val+1, val_loss))
                            
#                    print('Validation acc over epoch: %s \n' % (float(v_metric)))
                    print('epoch %d : Train loss, acc, Validation acc %s %s %s' % (plist['global_epoch'], format(loss), float(t_metric), float(v_metric)))
                    epoch_nums.append(plist['global_epoch']) 
                    losses.append(loss)                
                    t_metrics.append(t_metric)                
                    v_metrics.append(v_metric)                
                        
                    if val_loss_min > v_metric:

                        val_loss_min = v_metric

        ###                checkpoint.epoch.assign_add(1)
###                        save_path = manager.save()
###                        print("Saved checkpoint for epoch {}: {}".format(checkpoint.epoch.numpy(), save_path))
###                        print("\nRMSE {}\n".format(v_metric.numpy()))
                        plist['val_min'] = val_loss_min

                    if math.isnan(v_metric):
                        print('Breaking out as the validation loss is nan')
                        break                

                    if (epoch > 19):
                        if not (epoch % plist['early_stop_patience']):
                            if not (val_min):
                                val_min = val_loss_min
                            else:
                                if val_min > val_loss_min:
                                    val_min = val_loss_min
                                else:
                                    print('Breaking loop as validation accuracy not improving')
                                    print("loss {}".format(loss.numpy()))
                                    break

  #          fig_hist=plot_history(epoch_nums,t_metrics,v_metrics,normalized=True)
  #          fig_hist.savefig('history.png')
  #          with open('result.txt','w') as tmpf :
  #             tmpf.write('epoch %d : Train loss, acc, Validation acc %s %s %s \n' % (plist['global_epoch'], format(loss), float(t_metric), float(v_metric)))
  #             print('Time for epoch (seconds): %s' %((time.time() - start_time)))
    
    print('\n Total training time (in minutes): {}'.format((time.time()-timer_tot)/60))

    helpfunc.write_pickle(plist, plist['pickle_name'])
    
    ismp=10
    smp_forecast, smp_analysis, smp_accuracy = next(iter(val_dataloader))
    print(smp_forecast.shape)
    print(smp_analysis.shape)
    pred_analysis=model(smp_forecast.to(torch.float32))
  #  print("s_f, a_f")
  #  print(s_f, a_f)
  #  quit()
    smp_forecast = smp_forecast * np.mean(s_f.numpy()) + np.mean(a_f.numpy())     

  #  print(smp_forecast.detach().numpy().shape)
  #  print(smp_forecast.detach().numpy())
  #  quit()

    scatter_plot(((smp_forecast-pred_analysis).detach().numpy(), smp_forecast.detach().numpy(), (smp_forecast-smp_analysis).detach().numpy()), 1, 'Corrected_forecast', 'Biased_Forecast', 'Analysis', "tmp_img")
   
  #  model.summary()

    return val_loss_min
    
def traintest(plist):
    if torch.accelerator.is_available() :
        device=torch.accelerator.current_accelerator().type
        print('\nGPU Available: {}\n'.format(device))
    else: 
        device="cpu"
    #Get dataset
    print("\nProcessing Dataset\n")

    forecast_dataset, analysis_dataset, accuracy_dataset, a_f, s_f = helpfunc.createdataset(plist)

    if plist['make_recurrent']:
        analysis_split = helpfunc.split_sequences(analysis_dataset[:,:,:], plist['time_splits'])
        analysis_split = np.transpose(analysis_split, (1,0,2,3))
        accuracy_split = helpfunc.split_sequences(accuracy_dataset[:,:,:], plist['time_splits'])
        accuracy_split = np.transpose(accuracy_split, (1,0,2,3))
        forecast_split = helpfunc.split_sequences(forecast_dataset[:,:,:], plist['time_splits'])
        forecast_split = np.transpose(forecast_split, (1,0,2,3))

        analysis_dataset = np.reshape(analysis_split, (analysis_split.shape[0]*analysis_split.shape[1], plist['time_splits'], 1))
        accuracy_dataset = np.reshape(accuracy_split, (accuracy_split.shape[0]*accuracy_split.shape[1], plist['time_splits'], 1))
        forecast_dataset = np.reshape(forecast_split, (forecast_split.shape[0]*forecast_split.shape[1], plist['time_splits'], plist['locality']))

    else:
        plist['time_splits'] = 1
        analysis_dataset = np.reshape(analysis_dataset, (analysis_dataset.shape[0]*analysis_dataset.shape[1], 1))
        accuracy_dataset = np.reshape(accuracy_dataset, (accuracy_dataset.shape[0]*accuracy_dataset.shape[1], 1))
        forecast_dataset = np.reshape(forecast_dataset, (forecast_dataset.shape[0]*forecast_dataset.shape[1], plist['locality'] * plist['degree']))

    print("ana",analysis_dataset.shape)

    # Torch Dataset with weights
    val_dataset=helpfunc.WeightedDataset(forecast_dataset[-plist['val_size']:],analysis_dataset[-plist['val_size']:],accuracy_dataset[-plist['val_size']:])
    train_dataset=helpfunc.WeightedDataset(forecast_dataset[:-plist['val_size']],analysis_dataset[:-plist['val_size']],accuracy_dataset[:-plist['val_size']])


    print("tra",len(train_dataset))
   # quit()


    ### CHECK
    indices=torch.randint(0,len(train_dataset),(300,))
    smp_analysis, smp_forecast, _ = train_dataset[indices]
#    sample_scatter(smp_forecast,smp_analysis)
#    quit()

    # Torch Dataloader 
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=plist['global_batch_size'], shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=plist['global_batch_size_v'], shuffle=True)
   

    print("data in dataset")
    print(len(train_dataset))
                   
#    print("data in dataloader")      
#    for step, (X,y,z) in enumerate(train_dataloader):
#       print(step)      
#       print("fcst",len(X))
#       print("ana",len(y))
#       print("acc",len(z))
#    quit()

    model = net.rnn_model(plist)

    print(f"Model structure: {model}\n\n")
    print(model)
 
#    #print(smp_analysis)
#    smp_input=torch.nn.functional.normalize(torch.from_numpy(smp_forecast).to(torch.float32))
#    tmpmean=np.mean(smp_forecast)
#    tmpstd=np.std(smp_forecast-tmpmean)
#    smp_input2=(smp_forecast-tmpmean)/tmpstd
#    print(smp_forecast)
#    print(smp_input)
#    print(tmpmean,tmpstd,smp_input2)
#    print(model(torch.from_numpy(smp_input2).to(torch.float32)))
#    quit()
#    for item in model.parameters():
#      print(item)
#    quit()

#    for name, param in model.parameters():
#       print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
#    quit()

    optimizer = torch.optim.Adam(model.parameters(), lr=plist["learning_rate"])

    #Defining Model compiling parameters
    if plist['lr_decay_rate']:
        torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=plist['lr_decay_rate'], last_epoch=-1)

    #Defining the checkpoint instance
    a_f = torch.tensor(a_f)
    s_f = torch.tensor(s_f)
    time_splits = plist['time_splits']
      
    others_dict = OrderedDict()
    others_dict['epoch'] = 0
    others_dict['time_splits'] = time_splits
    others_dict['a_f'] = a_f
    others_dict['s_f'] = s_f
 
#    #Creating summary writer
#    summary_writer = tf.summary.create_file_writer(logdir= plist['log_dir'])

    save_directory = plist['checkpoint_dir']

    #Checking if previous checkpoint exists
#    if manager.latest_checkpoint:
#        print("Restored from {}".format(manager.latest_checkpoint))
            
#        print('Starting training from a restored point... \n')
#        return train(plist, model, checkpoint, optimizer, train_dataloader, val_dataloader)        
#    else:
#        print("No checkpoint exists.")
#        
#        print('Initializing from scratch... \n')
#        return train(plist, model, checkpoint, optimizer, train_dataloader, val_dataloader)

    return train(plist, model,  plist['checkpoint_dir'], optimizer, train_dataloader, val_dataloader, a_f, s_f)        

