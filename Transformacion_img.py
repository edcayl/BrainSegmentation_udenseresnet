# -*- coding: utf-8 -*-
"""
Created on Wed May  4 10:47:22 2022

@author: PC-FONDECYT-1
"""
import numpy as np
from skimage.transform import resize

def Obtener_bordes(IMref):
    
    out_y = (np.sum(IMref,axis=0)).astype(int)
    out_y = (np.sum(out_y,axis=1) == 0).astype(int)
     
    out_x = (np.sum(IMref,axis=1)).astype(int)
    out_x = (np.sum(out_x,axis=0) == 0).astype(int)
            
    out_z = (np.sum(IMref,axis=2)).astype(int)
    out_z = (np.sum(out_z,axis=1) == 0).astype(int)   
    
    return(out_x,out_y,out_z)

def Quitar_bordes(IM,axx,axy,axz):
    img_sinbordes = np.delete(IM, np.argwhere(axy==1),1)
    img_sinbordes = np.delete(img_sinbordes, np.argwhere(axx==1),2)
    img_sinbordes = np.delete(img_sinbordes, np.argwhere(axz==1),0) 
    
    return(img_sinbordes)


def resize_ski(IM, orden=0,tamano = (64,64,64),al=None,als=None):
          img_transformada = resize(IM, tamano, order=0, mode='reflect', cval=0, clip=True, preserve_range=True, anti_aliasing=False, anti_aliasing_sigma=False)
          
          return(img_transformada)
      
def margenes(img_size,patch_size,overlap):
    if overlap == 0:
        pasos = np.arange(0,int((img_size)-int(patch_size))+1,patch_size)
        
    else:
        last_step = 0
        pasos = [0]
        while(last_step + patch_size)<=(img_size):
            last_step = (last_step+patch_size)-overlap
            #print(last_step)
            if last_step <=(img_size-patch_size):
                pasos.append(last_step)
        ultimo_paso = pasos[len(pasos)-1]        
        if (ultimo_paso+patch_size) < img_size:
                pasos.append(img_size-patch_size)                                     
    return(pasos)               