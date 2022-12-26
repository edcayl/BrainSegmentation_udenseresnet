from medpy.io import load
import numpy as np
import os
import glob
import h5py
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from skimage.transform import resize
#from config import dlt_bgr, rs_ds
dlt_bgr = False
rs_ds = False
##########################Directorios################################
data_path 	= '/home4/eduardo.cavieres/MICCAI_BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'          # Path to iSeg2017 dataset (img and hdr files)
train_path 	= '/home4/eduardo.cavieres/data_train'                  # Path to save hdf5 data.
val_path 	= '/home4/eduardo.cavieres/data_val'                    # Path to save hdf5 data.
test_path   = '/home4/eduardo.cavieres/data_test'
#####################################################################

######################################################################


######################################################################

###################Separacion Sujetos Train/Test/Val#################
nrosujetos = 369
random = 0

data_info = pd.read_excel("/home4/eduardo.cavieres/Udense/Separacionsujetos.xlsx",skiprows=1,usecols="A,B,D")
X = data_info["Sujeto"]
Y = data_info["Izq"]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=74, random_state=None,stratify=Y)
x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=59, random_state=42,stratify=Y_train)

x_train = np.array(x_train)
x_val = np.array(x_val)
x_test = np.array(X_test)


s_train = len(x_train) 
s_validacion = len(x_val) 
s_test = len(x_test) 
#####################################################################

hora = datetime.now()
hora = hora.strftime(("%d-%m-%Y_%H-%M-%S"))
os.mkdir("/home4/eduardo.cavieres/Resultados/prep_datos_%s" %hora)
archivo = "/home4/eduardo.cavieres/Resultados/prep_datos_%s/dataset_%s.txt" %(hora,hora)
f_out = open(archivo, "a+")
f_out.write("nro sujetos:  %s  nro sujetos entrenamiento: %s nro sujetos validacion: %s  nro sujetos test: %s \n" %(nrosujetos, s_train, s_validacion, s_test))
f_salida = open("/home4/eduardo.cavieres/Resultados/prep_datos_%s/out_%s.txt" %(hora,hora),"a+")
f_mask = open("/home4/eduardo.cavieres/Resultados/prep_datos_%s/mask_size%s.txt" %(hora,hora),"a+")
# Ref1: https://github.com/zhengyang-wang/Unet_3D/tree/master/preprocessing
# Ref2: https://github.com/tbuikr/3D_DenseSeg/blob/master/prepare_hdf5_cutedge.py

def convert_label(label_img):
    '''
    function that converts 0, 10, 150, 250 to 0, 1, 2, 3 labels for BG, CSF, GM and WM
    '''
    label_processed = np.where(label_img==1, 1, label_img)
    label_processed = np.where(label_processed==2, 2, label_processed)
    label_processed = np.where(label_processed==4, 3, label_processed)
    return label_processed


def masking(IMref,IM):
    
    out_y = (np.sum(IMref,axis=0)).astype(int)
    out_y = (np.sum(out_y,axis=1) == 0).astype(int)
    #print(len(out_y))
    img_enmascarada = np.delete(IM, np.argwhere(out_y==1),1)
    
    
    out_x = (np.sum(IMref,axis=1)).astype(int)
    out_x = (np.sum(out_x,axis=0) == 0).astype(int)
    #print(len(out_y))
    img_enmascarada = np.delete(img_enmascarada, np.argwhere(out_x==1),2)
        
    out_z = (np.sum(IMref,axis=2)).astype(int)
    out_z = (np.sum(out_z,axis=1) == 0).astype(int)
    #print(len(out_y))
    img_enmascarada = np.delete(img_enmascarada, np.argwhere(out_z==1),0)
   
    
    if False:    
        out_x = (np.sum(IMref, axis=0) == 0).astype(int) 
        out_y = (np.sum(IMref, axis=1) == 0).astype(int) 
        out_z = (np.sum(IMref, axis=2) == 0).astype(int) 
        
        print(out_x.shape)
        
        img_enmascarada = np.delete(IM, np.argwhere(out_x==1), axis= 0)
        img_enmascarada = np.delete(IM, np.argwhere(out_y==1), axis=1)
        img_enmascarada = np.delete(IM, np.argwhere(out_z==1), axis=2)
    return(out_x,out_y,out_z)
    
def cut_edge(data, keep_margin):
    '''
    function that cuts zero edge
    '''
    D, H, W = data.shape
    D_s, D_e = 0, D - 1
    H_s, H_e = 0, H - 1
    W_s, W_e = 0, W - 1

    while D_s < D:
        if data[D_s].sum() != 0:
            break
        D_s += 1
    while D_e > D_s:
        if data[D_e].sum() != 0:
            break
        D_e -= 1
    while H_s < H:
        if data[:, H_s].sum() != 0:
            break
        H_s += 1
    while H_e > H_s:
        if data[:, H_e].sum() != 0:
            break
        H_e -= 1
    while W_s < W:
        if data[:, :, W_s].sum() != 0:
            break
        W_s += 1
    while W_e > W_s:
        if data[:, :, W_e].sum() != 0:
            break
        W_e -= 1

    if keep_margin != 0:
        D_s = max(0, D_s - keep_margin)
        D_e = min(D - 1, D_e + keep_margin)
        H_s = max(0, H_s - keep_margin)
        H_e = min(H - 1, H_e + keep_margin)
        W_s = max(0, W_s - keep_margin)
        W_e = min(W - 1, W_e + keep_margin)

    return int(D_s), int(D_e), int(H_s), int(H_e), int(W_s), int(W_e)

def build_h5_dataset(data_path):
    '''
    Build HDF5 Image Dataset.
    '''
    
#    vsuj = np.random.randint(1,nrosujetos,s_validacion)
    f_out.write("sujetos de entrenamiento: %s \n" %(x_train))	
    f_out.write("sujetos de validacion: %s  \n" %(x_val))
    f_out.write("sujetos de test: %s \n" %(x_test))
    f_out.close()
    filelist = glob.glob(os.path.join(train_path,"*"))
    for f in filelist:
    	os.remove(f)
    filelist = glob.glob(os.path.join(val_path,"*"))	
    for f in filelist:
    	os.remove(f)				
    filelist = glob.glob(os.path.join(test_path,"*"))	
    for f in filelist:
    	os.remove(f)


    for i in range(369):
        # Subject 9 for validation
        if (i+1) in x_val: # (i == 8):
             target_path = val_path
             group = "val"
        elif (i+1) in x_test:      
             target_path = test_path
             group= "test"
        else:
             target_path = train_path
             group = "train"
		

        if (i+1) > 99:
           subject_name = "BraTS20_Training_%d" % (i + 1)
        elif (i+1) > 9:
           subject_name = "BraTS20_Training_0%d" % (i + 1)
        else:
           subject_name = 'BraTS20_Training_00%d' % (i + 1)
	
        f_T1 = os.path.join("%s/%s/%s_" %(data_path,subject_name,subject_name) + 't1.nii.gz')
        img_T1, header_T1 = load(f_T1)
        f_T2 = os.path.join("%s/%s/%s_" %(data_path,subject_name, subject_name) + 't2.nii.gz')
        img_T2, header_T2 = load(f_T2)
        f_Flair = os.path.join("%s/%s/%s_" %(data_path,subject_name, subject_name) + 'flair.nii.gz')
        img_Flair, header_Flair = load(f_Flair)
        f_T1ce = ("%s/%s/%s_" %(data_path,subject_name, subject_name) + 't1ce.nii.gz')
        img_T1ce, header_T1ce = load(f_T1ce)
        f_l = os.path.join("%s/%s/%s_" %(data_path,subject_name, subject_name) + 'seg.nii.gz')
        labels, header_label = load(f_l)

        

        #################################
        
        if True:
  #            print(f_T1)
    
            
            out_x_t1 ,out_y_t1 ,out_z_t1 = masking(img_T1, img_T1)
            #f_T2 = masking(img_T1, img_T2)
            #inputs_tmp_T1ce = masking(img_T1, img_T1ce)
            #f_Flair = masking(img_T1, img_Flair)
            #print(img_T1.shape)
            
            #labels = masking(img_T1, labels)
        
        


####################################################
 
       	inputs_T1 = img_T1.astype(np.float32)
       	inputs_T2 = img_T2.astype(np.float32)
       	inputs_T1ce = img_T1ce.astype(np.float32)
       	inputs_Flair = img_Flair.astype(np.float32)
        labels = labels.astype(np.uint8)
        labels=convert_label(labels)
        mask=labels>0
        # Normalization
        inputs_T1_norm = (inputs_T1 - inputs_T1[mask].mean()) / inputs_T1[mask].std()
        inputs_T2_norm = (inputs_T2 - inputs_T2[mask].mean()) / inputs_T2[mask].std()
        inputs_T1ce_norm = (inputs_T1ce - inputs_T1ce[mask].mean()) / inputs_T1ce[mask].std()
        inputs_Flair_norm = (inputs_Flair - inputs_Flair[mask].mean()) / inputs_Flair[mask].std()
       
        #enmascarar
        if False:
            print(inputs_T1_norm.shape)
    
            
            inputs_tmp_T1 = masking(inputs_T1_norm, inputs_T1_norm)
            inputs_tmp_T2 = masking(inputs_T1_norm, inputs_T2_norm)
            inputs_tmp_T1ce = masking(inputs_T1_norm, inputs_T1ce_norm)
            inputs_tmp_Flair = masking(inputs_T1_norm, inputs_Flair_norm)
            print(inputs_tmp_T1.shape)
            
            labels_tmp = masking(inputs_T1_norm, labels)
        
    #########################
        if dlt_bgr:
                inputs_tmp_T1 = np.delete(inputs_T1_norm, np.argwhere(out_y_t1==1),1)
                inputs_tmp_T1 = np.delete(inputs_tmp_T1, np.argwhere(out_x_t1==1),2)
                inputs_tmp_T1 = np.delete(inputs_tmp_T1, np.argwhere(out_z_t1==1),0) 
                shape_mask = inputs_tmp_T1.shape

                inputs_tmp_T2 = np.delete(inputs_T2_norm, np.argwhere(out_y_t1==1),1)
                inputs_tmp_T2 = np.delete(inputs_tmp_T2, np.argwhere(out_x_t1==1),2)
                inputs_tmp_T2 = np.delete(inputs_tmp_T2, np.argwhere(out_z_t1==1),0) 

                inputs_tmp_T1ce = np.delete(inputs_T1ce_norm, np.argwhere(out_y_t1==1),1)
                inputs_tmp_T1ce = np.delete(inputs_tmp_T1ce, np.argwhere(out_x_t1==1),2)
                inputs_tmp_T1ce = np.delete(inputs_tmp_T1ce, np.argwhere(out_z_t1==1),0)

                inputs_tmp_Flair = np.delete(inputs_Flair_norm, np.argwhere(out_y_t1==1),1)
                inputs_tmp_Flair = np.delete(inputs_tmp_Flair, np.argwhere(out_x_t1==1),2)
                inputs_tmp_Flair = np.delete(inputs_tmp_Flair, np.argwhere(out_z_t1==1),0)

                labels_tmp = np.delete(labels, np.argwhere(out_y_t1==1),1)
                labels_tmp = np.delete(labels_tmp, np.argwhere(out_x_t1==1),2)
                labels_tmp = np.delete(labels_tmp, np.argwhere(out_z_t1==1),0)       
        else:
                inputs_tmp_T1 = inputs_T1_norm
                inputs_tmp_T2 = inputs_T2_norm
                inputs_tmp_T1ce = inputs_T1ce_norm
                inputs_tmp_Flair = inputs_Flair_norm
                labels_tmp = labels
        img_in_size = (240,240,155)
       ##############resize#################
        if rs_ds:
        	inputs_tmp_T1 = resize(inputs_tmp_T1, img_in_size, order=1, mode='reflect', cval=0, clip=True, preserve_range=True, anti_aliasing=False, anti_aliasing_sigma=False)
        	inputs_tmp_T2 = resize(inputs_tmp_T2, img_in_size, order=1, mode='reflect', cval=0, clip=True, preserve_range=True, anti_aliasing=False, anti_aliasing_sigma=False)
       		inputs_tmp_T1ce = resize(inputs_tmp_T1ce, img_in_size, order=1, mode='reflect', cval=0, clip=True, preserve_range=True, anti_aliasing=False, anti_aliasing_sigma=False)
        	inputs_tmp_Flair = resize(inputs_tmp_Flair, img_in_size, order=1, mode='reflect', cval=0, clip=True, preserve_range=True, anti_aliasing=False, anti_aliasing_sigma=False)
        	labels_tmp = resize( labels_tmp, img_in_size, order=0, mode='reflect', cval=0, clip=True, preserve_range=True, anti_aliasing=False, anti_aliasing_sigma=False)
    #########################    
        inputs_tmp_T1 = inputs_tmp_T1[:, :, :, None]
        inputs_tmp_T2 = inputs_tmp_T2[:, :, :, None]
        inputs_tmp_T1ce = inputs_tmp_T1ce[:, :, :, None]
        inputs_tmp_Flair =  inputs_tmp_Flair[:, :, :, None]
        labels_tmp =  labels_tmp[:, :, :, None]
           
 ##############################################
        
###################################################           
        inputs = np.concatenate((inputs_tmp_T1, inputs_tmp_T2, inputs_tmp_T1ce, inputs_tmp_Flair), axis=3)
        
        
        # Cut edge


        print ('Subject:', i+1, 'Input:', inputs.shape, 'Labels:', labels_tmp.shape, 'group:', group)
        i_correct = i + 1 
        f_salida.write('Subject: %s ; Input: %s ; Labels: %s ; group: %s \n' %(i_correct,inputs.shape,labels_tmp.shape,group))
        #f_mask.write('Subject: %s ; shape: %s ; group: %s \n' %(i_correct,shape_mask,group))
        f_salida.flush()
        f_mask.flush()
        
        
        inputs_caffe = inputs[None, :, :, :, :]
        labels_caffe = labels_tmp[None, :, :, :, :]
        inputs_caffe = inputs_caffe.transpose(0, 4, 3, 1, 2)
        labels_caffe = labels_caffe.transpose(0, 4, 3, 1, 2)

        with h5py.File(os.path.join(target_path, 'train_brats_%s.h5' % (i+1)), 'w') as f:
            f['data'] = inputs_caffe  # c x d x h x w
            f['label'] = labels_caffe
            
    f_salida.close()
    f_out.close() 
    f_mask.close
    
    f_log = open("net_log.txt","a+") 
    f_log.write("%s \t ,generate_h5 \n" %hora)
    f_log.close()
          

if __name__ == '__main__':
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(val_path):
        os.makedirs(val_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    build_h5_dataset(data_path)
