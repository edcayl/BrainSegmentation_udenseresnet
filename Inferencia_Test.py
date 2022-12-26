# -*- coding: utf-8 -*-
"""
Created on Mon May 23 11:46:19 2022

@author: PC-FONDECYT-1
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May 19 16:20:17 2022

@author: PC-FONDECYT-1
"""

from config import *
import time
from loss_func import dice
import SimpleITK as sitk

import glob
import os
import numpy as np
import pathlib
from datetime import datetime
from os import listdir
from Transformacion_img import Obtener_bordes, Quitar_bordes, resize_ski, margenes
import h5py
#op_dir = './iSeg-2019-Testing-labels'




hora = datetime.now()
hora = hora.strftime(("%d-%m-%Y_%H-%M"))
directorio = "/home4/eduardo.cavieres/Resultados/Imgout_%s" %hora
rootdirec = pathlib.Path().resolve()
os.mkdir("%s" %directorio)
op_dir = directorio#"./%s" %directorio



def read_med_image(file_path, dtype):
    #----Transformar imagen------
    img_stk = sitk.ReadImage(file_path)
    img_np = sitk.GetArrayFromImage(img_stk)
    img_np = img_np.astype(dtype)
    return img_np, img_stk

def convert_label(label_img):
    label_processed=np.zeros(label_img.shape[0:]).astype(np.uint8)
    for i in range(label_img.shape[2]):
        label_slice=label_img[:, :, i]
        label_slice[label_slice == 1 ] = 1# 10] = 1
        label_slice[label_slice == 2 ] = 2 #150] = 2
        label_slice[label_slice == 4 ] = 3#250] = 3
        label_processed[:, :, i]=label_slice
    return label_processed

def convert_label_submit(label_img):
    label_processed=np.zeros(label_img.shape[0:]).astype(np.uint8)
    for i in range(label_img.shape[2]):
        label_slice=label_img[:, :, i]
        label_slice[label_slice == 1] = 1#10
        label_slice[label_slice == 2] = 2#150
        label_slice[label_slice == 3] = 4#250
        label_processed[:, :, i]=label_slice
    return label_processed

def get_seg(net, op_dir):
    xstep = 64
    ystep = 64
    zstep = 64
    root_path = '/home4/eduardo.cavieres/MICCAI_BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
    test_path = "/home4/eduardo.cavieres/data_test"#"/home4/eduardo.cavieres/data_test"
    onlyfiles = listdir(test_path)
    test_subj = []
    for n in onlyfiles:
        test_subj.append(int(n[12:-3]))    #obtener grupo sujetos test
    test_subj.sort()

    
    for subject_id in (test_subj): #range(1,21):
        #if subject_id > 10:
        subject_id = int(subject_id)
        
        time_start = time.perf_counter()
        ###
        if (subject_id) > 99: 
          sub = "%s" %subject_id#"BraTS20_Training_%d/BraTS20_Training_%d_" % (subject_id,subject_id) 
        elif (subject_id) > 9: 
          sub = "0%s" %subject_id#"BraTS20_Training_0%d/BraTS20_Training_0%d_" % (subject_id,subject_id)
        else:
          sub = "00%s" %subject_id#"BraTS20_Training_00%d/BraTS20_Training_00%d_" % (subject_id,subject_id)

        trainset_image = "BraTS20_Training_%s/BraTS20_Training_%s_" %(sub,sub)
        ft1 = os.path.join(root_path, trainset_image + 't1.nii.gz')
        #ft2 = os.path.join(root_path, sub + 't2.nii.gz')
        #ft1ce = os.path.join(root_path, sub + 't1ce.nii.gz')
        #flair = os.path.join(root_path, sub + 'flair.nii.gz')
        
        ###

        imT1, imT1_itk = read_med_image(ft1, dtype=np.float32) 
        #borde_x,borde_y,borde_z = Obtener_bordes(imT1)
        #imT1 = Quitar_bordes(imT1,borde_x,borde_y,borde_z)

        
        
        

                 
        img_h5 = h5py.File("/home4/eduardo.cavieres/data_test/train_brats_%s.h5" %(subject_id),"r")
        t1_h5 =  img_h5["data"][0,0,:,:,:]
        t1_h5 = np.transpose(t1_h5, (0,2,1))
        t2_h5 = img_h5["data"][0,1,:,:,:]
        t2_h5 = np.transpose(t2_h5, (0,2,1))
        t1ce_h5 = img_h5["data"][0,2,:,:,:]
        t1ce_h5 = np.transpose(t1ce_h5, (0,2,1))
        flair_h5 = img_h5["data"][0,3,:,:,:]
        flair_h5 = np.transpose(flair_h5, (0,2,1))
        seg_h5 = img_h5["label"][0,0,:,:,:]
        seg_h5 = np.transpose(seg_h5, (0,2,1))
        
        

        input1 = t1_h5[:, :, :, None]
        input2 = t2_h5[:, :, :, None]
        input3 = t1ce_h5[:, :, :, None]
        input4 = flair_h5[:, :, :, None]

        inputs = np.concatenate((input1, input2, input3, input4), axis=3)
        inputs = inputs[None, :, :, :, :]
        image = inputs.transpose(0, 4, 1, 3, 2)               
        image = torch.from_numpy(image).float().to(device)   

        _, _, C, H, W = image.shape
        deep_slices   = margenes(C,patches_size[0],100)#np.arange(0, C - patches_size[0] + xstep, xstep)
        height_slices = margenes(H,patches_size[1],100)#np.arange(0, H - patches_size[1] + ystep, ystep)
        width_slices  = margenes(W,patches_size[2],100)#np.arange(0, W - patches_size[2] + zstep, zstep)

        
        


        whole_pred = np.zeros((1,)+(num_classes,) + image.shape[2:])
        count_used = np.zeros((image.shape[2], image.shape[3], image.shape[4])) + 1e-5
        #print(np.shape(whole_pred))
        with torch.no_grad():             
            #outputs = net(image)
            
            for i in range(len(deep_slices)):
                for j in range(len(height_slices)):
                    for k in range(len(width_slices)):
                        deep = deep_slices[i]
                        height = height_slices[j]
                        width = width_slices[k]
                        image_crop = image[:, :, deep   : deep   + patches_size[0],
                                                    height : height +patches_size[1],
                                                    width  : width  + patches_size[2]]
                        #print(np.shape(image_crop))

                        outputs = net(image_crop)
                        whole_pred[slice(None), slice(None), deep: deep + patches_size[0],
                                    height: height + patches_size[1],
                                    width: width + patches_size[2]] += outputs.data.cpu().numpy()

                        count_used[deep: deep + patches_size[0],
                                    height: height + patches_size[1],
                                    width: width + patches_size[2]] += 1
            
            
        
        #whole_pred = whole_pred / count_used
        whole_pred = whole_pred[0, :, :, :, :]
        
        whole_pred = np.argmax(whole_pred, axis=0)
        
        whole_pred = whole_pred.transpose(0,2,1)
        #print(np.shape(whole_pred))
        #print(np.shape(t1_h5))
        #break
        time_elapsed = (time.perf_counter() - time_start)
        f= open("%s/output_time.txt" %(directorio),"a+")
        f.write("subject_id / time_elapsed \n")
        f.write("Subject_%d %.16f\n" % (subject_id, time_elapsed))
        print("Subject_%d %.16f\n" % (subject_id, time_elapsed))

        f_pred = os.path.join( op_dir, "subject-%d-label.hdr"  % subject_id )
        whole_pred = (t1_h5 != 0) * whole_pred
        whole_pred = convert_label_submit(whole_pred)
        whole_pred_itk = sitk.GetImageFromArray(whole_pred.astype(np.uint8))
        whole_pred_itk.SetSpacing(imT1_itk.GetSpacing())
        whole_pred_itk.SetDirection(imT1_itk.GetDirection())
        sitk.WriteImage(whole_pred_itk, f_pred)

if __name__ == '__main__':
    if not os.path.exists(op_dir):
        os.makedirs(op_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = DenseResNet(num_init_features=32, growth_rate=16, block_config=(4, 4, 4, 4),num_classes=4).to(device)
    
    model = "best"
    f= open("%s/output_time.txt" %(directorio),"a+")
    f.write("Model: %s \n" % (model))
    f.close()
    
    if model == "best":
        saved_state_dict = torch.load( './checkpoints/best_model.pth' )
    
    else:    
        model = str(model).zfill(5)
        saved_state_dict = torch.load( './checkpoints/model_epoch_'+ model +'.pth' )
    
    net.load_state_dict(saved_state_dict)
    net.eval()
    
    d = get_seg(net, op_dir)
    
    f_log = open("net_log.txt","a+") 
    f_log.write("%s \t ,test \n" %hora)
    f_log.close()
