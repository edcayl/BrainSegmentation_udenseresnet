from config import *
import h5py
import torch.utils.data as data
import glob

class H5Dataset(data.Dataset):

    def __init__(self, root_path, crop_size=img_in_size, mode='train',pos_x=0,pos_y=0,pos_z=0):
        self.hdf5_list = [x for x in glob.glob(os.path.join(root_path, '*.h5'))]
        self.crop_size = crop_size
        self.patches_size = patches_size
        self.mode = mode
        self.indx_x = pos_x
        self.indx_y = pos_y
        self.indx_z = pos_z
        if (self.mode == 'train'):
            self.hdf5_list =self.hdf5_list + self.hdf5_list + self.hdf5_list + self.hdf5_list


    def __getitem__(self, index):
        h5_file = h5py.File(self.hdf5_list[index],"r")
        self.data = h5_file.get('data')
        self.label = h5_file.get('label')      
        self.label=self.label[:,0,...]        
        _, _, C, H, W = self.data.shape
        if (self.mode=='train'):
            cx = self.indx_x
            cy = self.indx_y
            cz = self.indx_z
            
            self.data_crop  = self.data [:, :, cx: cx + self.patches_size[0], cy: cy + self.patches_size[1], cz: cz + self.patches_size[2]]
            self.label_crop = self.label[:,  cx: cx + self.patches_size[0], cy: cy + self.patches_size[1], cz: cz + self.patches_size[2]]
            
            return (torch.from_numpy(self.data_crop[0,:,:,:,:]).float(),
                    torch.from_numpy(self.label_crop[0,:,:,:]).long())
            
            
            
        elif (self.mode == 'val'):
            cx = self.indx_x
            cy = self.indx_y
            cz = self.indx_z
            
            if whole_val == False:
                 self.label_crop = self.label[:,  cx: cx + self.patches_size[0], cy: cy + self.patches_size[1], cz: cz + self.patches_size[2]]
                 self.data_crop  = self.data [:, :, cx: cx + self.patches_size[0], cy: cy + self.patches_size[1], cz: cz + self.patches_size[2]]
                 return (torch.from_numpy(self.data_crop[0,:,:,:,:]).float(),
                         torch.from_numpy(self.label_crop[0,:,:,:]).long())
            
            
            else:
                 self.label_crop = self.label
                 #self.data_crop  = self.data [:, :, cx: cx + self.patches_size[0], cy: cy + self.patches_size[1], cz: cz + self.patches_size[2]]
                 self.data_crop = self.data
                 return (torch.from_numpy(self.data_crop[0,:,:,:,:]).float(),
                         (self.label_crop[0,:,:,:]))
#        print(C, H, W)
#        print(np.shape(self.data_crop))
#        print(np.shape(self.label_crop))
#        print(cx)
#        print(cx + self.patches_size[0])


    def __len__(self):
        return len(self.hdf5_list)
