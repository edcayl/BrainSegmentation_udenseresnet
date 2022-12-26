from config import *
import torch.utils.data as dataloader
from dataloader import H5Dataset
import torch.optim as optim
from loss_func import combined_loss, dice_loss, dice, eval_metrics
import time
from datetime import datetime
import numpy as np
from Transformacion_img import margenes
from tqdm import tqdm


#--------------------------Funciones---------------------------
def get_whole_pred(img_val,tget_val):
    C, H, W = img_in_size
    #print(np.shape(img_val))
    whole_pred = np.zeros((1,) +  img_in_size)
    whole_val = np.zeros((1,)+(num_classes,) +  img_in_size)
    count_used = np.zeros((img_in_size)) + 1e-5

    deep_slices =  margenes(int(C),int(patches_size[0]),1)
    height_slices =  margenes(int(H),int(patches_size[1]),1)
    width_slices =  margenes(int(W),int(patches_size[2]),1)
    #print(deep_slices)
    #print(height_slices)
    #print(width_slices)

    whole_loss = []    
    for i in (deep_slices):
        for j in (height_slices):
            for k in (width_slices):
                deep = i
                height = j
                width = k
                image_crop = img_val[ :,:, deep   : deep   + patches_size[0],
                                            height : height + patches_size[1],
                                            width  : width  + patches_size[2]]
                
                tget_crop = tget_val[:, deep   : deep   + patches_size[0],
                                            height : height + patches_size[1],
                                            width  : width  + patches_size[2]]
                model.eval()
                images_val = image_crop.to(device)
             
                outputs_val = model(images_val)
                tget_crop =  tget_crop.to(device)
                val_loss = loss_criteria(outputs_val,tget_crop)
                whole_loss.append(val_loss.item())

                whole_val[slice(None),slice(None),deep: deep + patches_size[0],
                            height: height + patches_size[1],
                            width: width + patches_size[2]] = outputs_val.data.cpu().numpy()
                


    
    #whole_val = whole_val / count_used
    whole_val = whole_val[0,:,:, :, :]
    whole_val = np.argmax(whole_val, axis=0)  
    val_loss = np.mean(whole_loss)                

    return whole_val,val_loss



# --------------------------CUDA check-----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -------------init Seg---------------
model = DenseResNet(num_init_features=32, growth_rate=16, block_config=(4, 4, 4, 4), drop_rate=0.2, num_classes=num_classes).to(device)
# --------------Loss---------------------------
  #----Loss Weight--------

if loss_weights is not None:
 pesos = torch.from_numpy(np.array(loss_weights)).float().to(device)
  #------------------------
 loss_criteria = combined_loss({'batch_dice': False, 'smooth': 1e-5, 'do_bg': False, 'square': False}, {'weight': pesos}).cuda()

else:
 loss_criteria = combined_loss({'batch_dice': False, 'smooth': 1e-5, 'do_bg': False, 'square': False}, {}).cuda()
optimizer = optim.Adam(model.parameters(), lr=lr_init, weight_decay=6e-4, betas=(0.97, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_stepsize, gamma=0.1)
# --------------Start Training and Validation ---------------------------

if __name__ == '__main__':
    checkpt = './checkpoints'
    best_loss = float("inf")           #####mejor lost del entrenamiento, valor previsional inicial muy alto
    best_epoch = 0
    
    if not os.path.exists(checkpt):
        os.makedirs(checkpt)
    #-----------------------Training--------------------------------------
    mri_data_train = H5Dataset("/home4/eduardo.cavieres/data_train", mode='train')
    trainloader = dataloader.DataLoader(mri_data_train, batch_size=batch_train, shuffle=True)
    mri_data_val = H5Dataset("/home4/eduardo.cavieres/data_val", mode='val')
    valloader = dataloader.DataLoader(mri_data_val, batch_size=1, shuffle=False)
    
    hora = datetime.now()
    hora = hora.strftime(("%d-%m-%Y_%H-%M-%S"))	
    
    dir_train = '/home4/eduardo.cavieres/Resultados/Entrenamiento_%s' %hora
    if not os.path.exists(dir_train):
        os.makedirs(dir_train)
    
    
    f1 = open('%s/_loss_%s.txt' %(dir_train,hora), 'a+')
    f2 = open('%s/_out_%s.txt' %(dir_train,hora) , 'a+')
    fv = open('%s/_DetallesVal_%s.txt' %(dir_train,hora) , 'a+')
    #print('              Clock |        LR |   Epoch |           Loss |        Val DSC |\n')
    
    f2.write( 'Fecha inicio: %s \n' %hora )
    f2.write( 'Train Config: \n' )
    f2.write( 'num_epoch = %s ; lr_init = %s ; lr_stepsize = %s ; \n batch_train = %s ; num_classes= %s ; img_in_size = %s \n \n' %(num_epoch, lr_init, lr_stepsize, batch_train, num_classes, img_in_size))
    f2.write( 'Patch_Size = %s ; Overlap = %s  \n' %(patches_size,solapado))
    f2.write( '              Clock |        LR |   Epoch |           Loss |        Val DSC |        net DSC |        pe DSC |        et DSC |\n' )
    fv.write("num_epoch,, net_dsc, pe_dsc, et_dsc, avr_dsc \n") 
    fv.flush()
    fm = open('%s/_measurements_%s.txt' %(dir_train,hora), 'a+')
    #----------------margenes en cada direccion-----------------------------   
    patches_num = np.array(img_in_size)/np.array(patches_size)
    x_pos =  margenes(int(img_in_size[0]),int(patches_size[0]),solapado)
    x_pos = iter(x_pos)
    y_pos =  margenes(int(img_in_size[1]),int(patches_size[1]),solapado)
    y_pos = iter(y_pos)
    z_pos =  margenes(int(img_in_size[2]),int(patches_size[2]),solapado)
    z_pos = iter(z_pos)
    indexx = next(x_pos)
    indexy = next(y_pos)
    indexz = next(z_pos)
    move = "init"
    #------------------------------------------------------------------------
    
    for epoch in range (num_epoch + 1):
        #---------------movimiento del patch----------------------
        if move == "z":
            indexz = next(z_pos)
            if indexz == (img_in_size[2]-patches_size[2]):
                move = "x"
                x_pos =  margenes(int(img_in_size[0]),int(patches_size[0]),solapado)
                x_pos = iter(x_pos)
                y_pos =  margenes(int(img_in_size[1]),int(patches_size[1]),solapado)
                y_pos = iter(y_pos)
                z_pos =  margenes(int(img_in_size[2]),int(patches_size[2]),solapado)
                z_pos = iter(z_pos)
                indexx = next(x_pos)
                indexy = next(y_pos)
                indexz = next(z_pos)
                
                
        if move == "y":
            indexy = next(y_pos)
            if indexy == (img_in_size[1]-patches_size[1]):
                move = "z"    
        if move == "x":
            indexx = next(x_pos)
            if indexx == (img_in_size[0]-patches_size[0]):
                move = "y"
        if move == "init":
            move = "x"        
                
        
        mri_data_train = H5Dataset("/home4/eduardo.cavieres/data_train", mode='train',pos_x=indexx,pos_y=indexy,pos_z=indexz)
        trainloader = dataloader.DataLoader(mri_data_train, batch_size=batch_train, shuffle=True)
        mri_data_val = H5Dataset("/home4/eduardo.cavieres/data_val", mode='val',pos_x=indexx,pos_y=indexy,pos_z=indexz)
        valloader = dataloader.DataLoader(mri_data_val, batch_size=1, shuffle=False)
        #---------------------------------------------------------------------------------

        running_loss = []
        model.train()
        for i, data in enumerate(trainloader):
        #for i, data in enumerate(tqdm((trainloader), total=len(trainloader),desc="Training... ",position = 0,leave=True)):

            images, targets = data
            images = images.to(device)
            targets = targets.to(device)           
            optimizer.zero_grad()
            outputs = model(images)
            loss_seg = loss_criteria(outputs, targets)
            running_loss.append(loss_seg.item())
            f1.write( '%d %f\n' % (epoch, loss_seg.item()) )
            
 
            loss_seg.backward()
            optimizer.step()
            
        running_loss = np.mean(running_loss)
        scheduler.step()
        # -----------------------Validation------------------------------------
        # no update parameter gradients during validation
        with torch.no_grad():
            et_eval = []
            tc_eval = []
            wt_eval = []
            if whole_val== False:
                fv.write("%s, ," %epoch)
                fv.flush()
                 
                #for data_val in (tqdm((valloader), total=len(valloader),desc="Validating... ",position = 0,leave=True)):
                for data_val in valloader:
                    images_val, targets_val = data_val
                    model.eval()
                    images_val = images_val.to(device)
                    targets_val = targets_val.to(device)
                    outputs_val = model(images_val)
                    _, predicted = torch.max(outputs_val.data, 1)
                    # ----------Compute dice-----------
                    predicted_val = predicted.data.cpu().numpy()
                    targets_val = targets_val.data.cpu().numpy()
                    dsc = []
                    
                    for i in range(1, num_classes):  # ignore Background 0
                        dsc_i,_,_,_= (dice(predicted_val, targets_val, i))
                        dsc.append(dsc_i)
                        fv.write("%s, " %dsc_i)
                        
                        
                    dsc_comp = dsc
                    dsc = np.mean(dsc)
                    fv.write("%s , ," %dsc_i)
                    
                    et_val,tc_val,wt_val = eval_metrics(predicted_val, targets_val)
                    et_eval.append(et_val)
                    tc_eval(tc_val)
                    wt_eval(wt_val)
                    
            else:
               fv.write("%s, ," %epoch)
               fv.flush()
               rval_loss = []
               #ifor data_val in (tqdm((valloader), total=len(valloader),desc="Validating... ",position = 0,leave=True)):                
               for data_val in valloader:
                   images_val, targets_val = data_val
                   predicted_val,validation_loss = get_whole_pred(images_val, targets_val)
                   rval_loss.append(validation_loss)
                   targets_val = targets_val.data.cpu().numpy()
                   t_v =targets_val[0,:,:,:]
                   targets_val = targets_val[0,:,:,:]
                   dsc = []
                   #print(np.shape(predicted_val))
                  # print(np.shape(targets_val))
                   for i in range(1, num_classes):  # ignore Background 0
                       dsc_i,_,im1_px,im2_px= (dice(predicted_val, targets_val, i))
                       fv.write("%s, " %dsc_i)
                       
                       if (im1_px + im2_px) > 0:    ###ignorar valores nan
                           dsc.append(dsc_i)
                           
                       
                        
                   dsc_comp = dsc
                   dsc = np.mean(dsc)
                   fv.write("%s , ," %dsc)
                   et_val,tc_val,wt_val = eval_metrics(predicted_val, targets_val)
                   et_eval.append(et_val)
                   tc_eval.append(tc_val)
                   wt_eval.append(wt_val)
                   
            et_eval = np.mean(et_val)
            tc_eval = np.mean(tc_val)
            wt_eval = np.mean(wt_val)
            rval_loss = np.mean(rval_loss)      

        #-------------------Debug-------------------------
        for param_group in optimizer.param_groups:
            #print('%19.7f | %0.7f | %7d | %14.9f | %14.9f |' % (\
            #        time.perf_counter(), param_group['lr'], epoch, loss_seg.item(), dsc))
            print('     LR |   Epoch |  Train Loss |  Val Loss|  Val DSC |')
            print('%10.7f | %7d |  %12.9f | %12.9f | %12.9f | \n' % (\
                param_group['lr'], epoch, running_loss, rval_loss, dsc))
            f2.write('%19.7f | %0.7f | %7d | %14.9f | %14.9f | %14.9f | %14.9f | %14.9f |\n' % (\
                    time.perf_counter(), param_group['lr'], epoch, loss_seg.item(), dsc,dsc_comp[0],dsc_comp[1],dsc_comp[1]))
            f2.flush() 
            f1.flush()
            fv.write(", ,%s  \n" %dsc)
            fv.flush()
            fm.write("%s,%s,%s \n " %(et_eval,tc_eval,wt_eval))
            fm.flush()

        #Save checkpoint
        if (epoch % 100) == 0 or epoch == (num_epoch - 1) or (epoch % 1000) == 0:
            torch.save(model.state_dict(), './checkpoints/' + '%s_%s.pth' % (checkpoint_name, str(epoch).zfill(5)))
        if  loss_seg.item() < best_loss:
            torch.save(model.state_dict(), './checkpoints/' + 'best_model.pth')
            best_epoch = str(epoch).zfill(5)
    
    f2.flush()
    f1.write("Best| %s | %s | \n" %(best_epoch,best_loss))
    f1.close()
    f3 = open("./checkpoints/last_train.txt" , "w")
    f3.write("inicio_entrenamiento:%s \n" %hora)
    
    f_log = open("net_log.txt","a+") 
    f_log.write("%s \t ,train_start \n" %hora)

    hora = datetime.now()
    hora = hora.strftime(("%d-%m-%Y_%H-%M-%S"))	
    
    f3.write("fin_entrenamiento:%s \n" %hora)
    f3.write("Mejor epoch: %s , valor loss: %s \n" %(best_epoch,best_loss))
    f3.close()
    f2.write( 'Fecha fin: %s \n' %hora )
    f2.close()
    
    f_log.write("%s \t ,train_end \n" %hora)
    f_log.close()
    fv.close()

