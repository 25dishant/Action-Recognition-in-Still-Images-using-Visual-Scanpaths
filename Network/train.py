import pdb
from matplotlib.pyplot import axis
from tqdm import tqdm
import torch
import gc
import os
from myutils import get_ap_score
import numpy as np
import logging
import time

# def train_model(model, device, optimizer, scheduler, train_data, val_data, save_dir, model_num, epochs, log_file):
#     """
#     Train a deep neural network model
    
#     Args:
#         model: pytorch model object
#         device: cuda or cpu
#         optimizer: pytorch optimizer object
#         scheduler: learning rate scheduler object that wraps the optimizer
#         train_dataloader: training  images dataloader
#         valid_dataloader: validation images dataloader
#         save_dir: Location to save model weights, plots and log_file
#         epochs: number of training epochs
#         log_file: text file instance to record training and validation history
        
#     Returns:
#         Training history and Validation history (loss and average precision)
#     """
    
#     tr_loss, tr_map = [], []
#     val_loss, val_map = [], []
#     best_val_map = 0.0


#     # lr_warmup = float(args.lr_warmup)  # avoid int division
#     # cosine lr annealing
#     # lr_annealing = CosineAnnealingSchedule(min_lr=args.min_lr, max_lr=args.max_lr, cycle_length=args.cycle_len)
#     # epoch_size = len(train_data._dataset)

#     # rcnn_cls_loss = mx.gluon.loss.SoftmaxCrossEntropyLoss()
#     # metrics = [mx.metric.Loss('RCNN_CrossEntropy'), ]

#     # rcnn_acc_metric = RCNNAccMetric()
#     # metrics2 = [rcnn_acc_metric, ]

    
#     # Each epoch has a training and validation phase
#     for epoch in range(epochs):
#         print("-------Epoch {}----------".format(epoch+1))
#         log_file.write("Epoch {} >>".format(epoch+1))
#         scheduler.step()
        
#         for phase in ['train', 'valid']:
#             running_loss = 0.0
#             losses = []
#             running_ap = 0.0
            
#             # criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')
#             rcnn_cls_loss = torch.nn.CrossEntropyLoss(reduction='sum')
#             m = torch.nn.Sigmoid()
            
#             if phase == 'train':
#                 model.train(True)  # Set model to training mode
                
#             for i, batch in enumerate(train_data):
#                 # metric_losses = [[] for _ in metrics]
#                 metric_losses = [torch.nn.CrossEntropyLoss()]
#                 # add_losses = [[] for _ in metrics2]
#                 # with torch.no_grad:
#                 for data, label, box in zip(*batch):
#                     # print('data: ',data)
#                     # print('label: ',label)
#                     # print('box: ',box)
#                     # return
#                     gt_label = label[:, :, 4:5].squeeze(axis=-1)
#                     gt_box = label[:, :, :4]
#                     cls_pred = model(data)
#                     # losses of rcnn
#                     rcnn_loss = rcnn_cls_loss(cls_pred, gt_label)
#                     # overall losses
#                     losses.append(rcnn_loss.sum())
#                     metric_losses[0].append(rcnn_loss.sum())
#                     # add_losses[0].append([[gt_label], [cls_pred]])
#                 # torch.autograd.backward(losses)
#                 for metric, record in zip(metric_losses):
#                     metric.update(0, record)
#                 # for metric, records in zip(metrics2, add_losses):
#                 #     for pred in records:
#                 #         metric.update(pred[0], pred[1])
#             # scheduler.step()


#             # if args.log_interval and not (i + 1) % args.log_interval:
#             #     # msg = ','.join(['{}={:.3f}'.format(*metric.get()) for metric in metrics])
#             #     msg = ','.join(['{}={:.3f}'.format(*metric.get()) for metric in metrics + metrics2])
#             #     log_file.info('[Epoch {}][Batch {}], Speed: {:.3f} samples/sec, {}'.format(
#             #         epoch, i, args.log_interval * batch_size/(time.time()-btic), msg))
#             #     btic = time.time()


#             # for data, target in tqdm(train_data):
#                 #print(data)
#                 target = target.float()
#                 data, target = data.to(device), target.to(device)
                
#                 # zero the parameter gradients
#                 optimizer.zero_grad()
                
#                 output = model(data)
                
#                 loss = rcnn_cls_loss(output, target)
                
#                 # Get metrics here
#                 running_loss += loss # sum up batch loss
#                 running_ap += get_ap_score(torch.Tensor.cpu(target).detach().numpy(), torch.Tensor.cpu(m(output)).detach().numpy()) 
            
#                 # Backpropagate the system the determine the gradients
#                 loss.backward()
                
#                 # Update the paramteres of the model
#                 optimizer.step()
        
#                 # clear variables
#                 del data, target, output
#                 gc.collect()
#                 torch.cuda.empty_cache()
                
#                 #print("loss = ", running_loss)
                    
#                 num_samples = float(len(train_data.dataset))
#                 tr_loss_ = running_loss.item()/num_samples
#                 tr_map_ = running_ap/num_samples
                
#                 print('train_loss: {:.4f}, train_avg_precision:{:.3f}'.format(
#                     tr_loss_, tr_map_))
                
#                 log_file.write('train_loss: {:.4f}, train_avg_precision:{:.3f}, '.format(
#                     tr_loss_, tr_map_))
                
#                 # Append the values to global arrays
#                 tr_loss.append(tr_loss_), tr_map.append(tr_map_)
                        
                        
#             else:
#                 model.train(False)  # Set model to evaluate mode
        
#                 # torch.no_grad is for memory savings
#                 with torch.no_grad():
#                     for data, target in tqdm(val_data):
#                         target = target.float()
#                         data, target = data.to(device), target.to(device)
#                         output = model(data)
                        
#                         loss = rcnn_cls_loss(output, target)
                        
#                         running_loss += loss # sum up batch loss
#                         running_ap += get_ap_score(torch.Tensor.cpu(target).detach().numpy(), torch.Tensor.cpu(m(output)).detach().numpy()) 
                        
#                         del data, target, output
#                         gc.collect()
#                         torch.cuda.empty_cache()

#                     num_samples = float(len(val_data.dataset))
#                     val_loss_ = running_loss.item()/num_samples
#                     val_map_ = running_ap/num_samples
                    
#                     # Append the values to global arrays
#                     val_loss.append(val_loss_), val_map.append(val_map_)
                
#                     print('val_loss: {:.4f}, val_avg_precision:{:.3f}'.format(
#                     val_loss_, val_map_))
                    
#                     log_file.write('val_loss: {:.4f}, val_avg_precision:{:.3f}\n'.format(
#                     val_loss_, val_map_))
                    
#                     # Save model using val_acc
#                     if val_map_ >= best_val_map:
#                         best_val_map = val_map_
#                         log_file.write("saving best weights...\n")
#                         torch.save(model.state_dict(), os.path.join(save_dir,"model-{}.pth".format(model_num)))
                    
#     return ([tr_loss, tr_map], [val_loss, val_map])


def train_model(model, device, optimizer, scheduler, train_data, val_data, save_dir, model_num, epochs, log_file, eval_metric):
    tr_loss, tr_map = [], []
    val_loss, val_map = [], []
    best_val_map = 0.0
    
    for epoch in range(epochs):
        print("-------Epoch {}----------".format(epoch+1))
        log_file.write("Epoch {} >>".format(epoch+1))
        scheduler.step()
        
        running_loss = 0.0
        total_loss = 0.0
        running_ap = 0.0
        
        criterion = torch.nn.CrossEntropyLoss(reduction='sum')
        m = torch.nn.Sigmoid()
        
        model.train(True)  # Set model to training mode
        
        for i, batch in enumerate(train_data):
            #print(data)
            # target = target.float()
            #pdb.set_trace()
            data = batch[0]
            label = batch[1]
            box = batch[2]
            img_id = batch[3][0]
            observer_id = batch[4][0]
            fixations = batch[5]
            gt_label = label[0, 0, 4:5].squeeze(axis=-1)
            # gt_label = label[:, :, 4:5].squeeze(axis=-1)
            # gt_label = label[:,:,5:].squeeze(axis=-1)
            # gt_label = gt_label.reshape(1,11)
            gt_label = gt_label.long()
            gt_label = gt_label.reshape(1)
            # gt_label = gt_label[:,:11]
            # print(gt_label)
            gt_box = label[:, :, :4]
            data, gt_label, gt_box, box, fixations = data.to(device), gt_label.to(device), gt_box.to(device), box.to(device), fixations.to(device)
            optimizer.zero_grad()
            cls_pred = model(data,gt_box,box,fixations)
            # print(cls_pred)
            loss = criterion(cls_pred, gt_label)
            #print("LOSS: ",loss)
            running_loss += loss.item() # sum up batch loss
            total_loss += loss
            # print("RUNNING LOSS: ",running_loss)
            #running_ap += get_ap_score(torch.Tensor.cpu(gt_label).detach().numpy(), torch.Tensor.cpu(m(cls_pred)).detach().numpy()) 
            loss.backward()
            optimizer.step()
            if i%100==99:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0
                
                #clear variables
                # del data, target, output
                # gc.collect()
                # torch.cuda.empty_cache()
        #pdb.set_trace()
        train_loss = total_loss/len(train_data)
        print("Train Loss: ", train_loss)
        #-------------------------Validation---------------------------#
        #pdb.set_trace()        
        correct = 0
        total = 0
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_data:
                cls_scores = []
                gt_classes = []
                data = batch[0]
                label = batch[1]
                box = batch[2]
                img_id = batch[3][0]
                observer_id = batch[4][0]
                fixations = batch[5]
                data, label, box, fixations = data.to(device), label.to(device), box.to(device), fixations.to(device)
                gt_box = label[:, :, :4]
                gt_label = label[0,0,4:5].squeeze(axis = -1)
                gt_label = gt_label.long()
                gt_label = gt_label.reshape(1)
                cls_score = model(data, gt_box, box, fixations)
                val_loss = criterion(cls_score, gt_label)
                total_val_loss += val_loss
                #cls_score = mx.nd.softmax(cls_score, axis=-1)
                m = torch.nn.Softmax(dim=1)
                cls_score = m(cls_score)
                cls_scores.append(cls_score[:, :])
                gt_classes.append(label[:, :, 5:])
            #pdb.set_trace()
            print("Total Validation Loss: ", total_val_loss/len(val_data))
            for score, gt_class in zip(cls_scores, gt_classes):
                eval_metric.update(score, gt_class)
            #pdb.set_trace()
            map_name, mean_ap = eval_metric.get()
            val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
            log_file.write('[Epoch {}] Validation: \n{}'.format(epoch, val_msg))
            current_map = float(mean_ap[-1])
            print("Current MAP: ", current_map)
            outputs_file_name = '{:s}_{:04d}_val_outputs.csv'.format("", epoch)
            eval_metric.save(file_name=outputs_file_name)
            # zero the parameter gradients
            
            # Get metrics here
        
            # Backpropagate the system the determine the gradients
            
            # Update the paramteres of the model
    
            
            
            #print("loss = ", running_loss)
            
        #num_samples = float(len(train_data))
        #tr_loss_ = running_loss.item()/num_samples
        #tr_map_ = running_ap/num_samples
        
        #print('train_loss: {:.4f}, train_avg_precision:{:.3f}'.format(
            #tr_loss_, tr_map_))
        
        #log_file.write('train_loss: {:.4f}, train_avg_precision:{:.3f}, '.format(
            #tr_loss_, tr_map_))
        
        # Append the values to global arrays
        #tr_loss.append(tr_loss_), tr_map.append(tr_map_)
                    
    #return ([tr_loss, tr_map], [val_loss, val_map])