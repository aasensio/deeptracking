import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import h5py
from torch.autograd import Variable
import time
from tqdm import tqdm
import model
import argparse
import platform
import astropy.units as u
import sys
import os

class dataset_h5(torch.utils.data.Dataset):
    def __init__(self, in_file, n_training, batch_size, training=True):
        super(dataset_h5, self).__init__()
        
        print("Reading {0}".format(in_file))
        self.file = h5py.File(in_file, 'r')
        self.n_images, _, self.nx, self.ny = self.file['images'].shape
        
        if (training):
            print("Reading training images...")
        else:
            print("Reading validation images...")

        self.images = self.file['images']
                
        if (n_training == -1):
            self.n_training = (self.n_images // batch_size) * batch_size
        else:        
            self.n_training = n_training
                                
        print('Number of used images : {0}/{1}'.format(self.n_training, self.n_images))

    def __getitem__(self, index):

        ims = self.images[index,:,:,:]
        
        minim = np.min(ims, axis=(1,2))
        maxim = np.max(ims, axis=(1,2))

        ims = (ims - minim[:,None,None]) / (maxim[:,None,None] - minim[:,None,None])
        
        return ims.astype('float32')

    def __len__(self):
        return self.n_training

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename+'.best')

class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt( diff * diff + self.eps )
        loss = torch.sum(error) 
        return loss 
        

class optical_flow(object):
    def __init__(self, batch_size, n_training=-1, n_testing=-1, resume=None):
        self.cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda else "cpu")
        self.batch_size = batch_size
        self.n_training = n_training
        self.n_testing = n_testing
        
        # torch.backends.cudnn.benchmark = True

        computer_name = platform.node()

        if (computer_name == 'deimos'):
            self.train_file = '/scratch/aasensio/deep_learning/deepvel/training.h5'
            self.test_file = '/scratch/aasensio/deep_learning/deepvel/validation.h5'
        

        kwargs = {'num_workers': 2, 'pin_memory': True} if self.cuda else {}
               
        self.model = model.network(n_pixel=64, device=self.device).to(self.device)
                
        self.train_loader = torch.utils.data.DataLoader(dataset_h5(self.train_file, self.n_training, self.batch_size, training=True),
                                                        batch_size=self.batch_size, shuffle=True, **kwargs)

        self.test_loader = torch.utils.data.DataLoader(dataset_h5(self.test_file, self.n_testing, self.batch_size, training=False),
                                                       batch_size=self.batch_size, shuffle=True, **kwargs)

    def init_optimize(self, epochs, lr, resume, weight_decay):

        self.lr = lr
        self.weight_decay = weight_decay        
        print('Learning rate : {0}'.format(lr))
        self.n_epochs = epochs

        # Check for resuming
        if (len(parsed['resume']) != 0):
            self.out_name = resume
            self.file_mode = 'a'
        else:
            current_time = time.strftime("%Y-%m-%d-%H:%M:%S")
            self.out_name = 'trained/{0}'.format(current_time)

            # Copy model
            shutil.copyfile(model.__file__, '{0}_model.py'.format(self.out_name))
            shutil.copyfile('{0}/{1}'.format(os.path.dirname(os.path.abspath(__file__)), __file__), '{0}_trainer.py'.format(self.out_name))
            self.file_mode = 'w'

            f = open('{0}_hyper.dat'.format(self.out_name), 'w')
            f.write('Learning_rate       Weight_decay     \n')
            f.write('{0}    {1}'.format(self.lr, self.weight_decay))
            f.close()

        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        if (len(parsed['resume']) != 0):
            print("=> loading checkpoint '{}'".format(self.out_name))
            if (self.cuda):
                checkpoint = torch.load('{0}.pth'.format(self.out_name))
            else:
                checkpoint = torch.load('{0}.pth'.format(self.out_name), map_location=lambda storage, loc: storage)
            
            self.model.load_state_dict(checkpoint['state_dict'])        
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            
            # Modify the learning rate with respect to that read from the file
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            print("=> loaded checkpoint '{}'".format(self.out_name))

        self.loss_fn = nn.MSELoss().to(self.device)

        self.loss_charb = L1_Charbonnier_loss().to(self.device)

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.5)

    def optimize(self):
        self.loss = []
        self.loss_val = []
        best_loss = -1e10

        trainF = open('{0}.loss.csv'.format(self.out_name), self.file_mode)

        print('Model : {0}'.format(self.out_name))

        for epoch in range(1, self.n_epochs + 1):
            self.scheduler.step()
            self.train(epoch)
            self.test(epoch)

            trainF.write('{},{},{}\n'.format(
                epoch, self.loss[-1], self.loss_val[-1]))
            trainF.flush()

            is_best = self.loss_val[-1] < best_loss
            best_loss = min(self.loss_val[-1], best_loss)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'best_loss': best_loss,
                'optimizer': self.optimizer.state_dict(),
            }, is_best, filename='{0}.pth'.format(self.out_name))

        trainF.close()

    def get_lr(self, init_value, final_value, beta):
        self.model.train()

        num = len(self.train_loader)-1
        mult = (final_value / init_value) ** (1/num)
        lr = init_value
        losses = []
        lrs = []
        avg_loss = 0.
        best_loss = 0.
        batch_num = 0
        self.optimizer.param_groups[0]['lr'] = lr
        t = tqdm(self.train_loader)
        
        f = open('lr_optimization.txt', 'w')
        for batch_idx, (im_focus_defocus, im_original, zernike_target, minim, maxim) in enumerate(t):

            batch_num += 1
            
            im_focus_defocus, im_original, zernike_target = im_focus_defocus.to(self.device), im_original.to(self.device), zernike_target.to(self.device)
            minim, maxim = minim.to(self.device), maxim.to(self.device)
            
            self.optimizer.zero_grad()
            output_focus, output_defocus, wavefront, wavefront_target = self.model(im_focus_defocus, im_original, zernike_target)

            output_focus = (output_focus - minim[:,0,None,None]) / (maxim[:,0,None,None] - minim[:,0,None,None])
            output_defocus = (output_defocus - minim[:,1,None,None]) / (maxim[:,1,None,None] - minim[:,1,None,None])
            
            loss = self.loss_fn(output_focus, im_focus_defocus[:,0,:,:])
            loss += self.loss_fn(output_defocus, im_focus_defocus[:,1,:,:])
            loss += 5e-2*self.loss_fn(wavefront, wavefront_target)

            avg_loss = beta * avg_loss + (1-beta) *loss.item()
            smoothed_loss = avg_loss / (1 - beta**batch_num)
            #Stop if the loss is exploding
            if batch_num > 1 and smoothed_loss > 4 * best_loss:
                f.close()
                return
            #Record the best loss
            if smoothed_loss < best_loss or batch_num==1:
                best_loss = smoothed_loss
            #Store the values
            losses.append(smoothed_loss)
            lrs.append(lr)

            loss.backward()

            self.optimizer.step()

            lr *= mult
            self.optimizer.param_groups[0]['lr'] = lr

            t.set_postfix(loss=smoothed_loss, lr=lr)

            f.write('{0}  {1}\n'.format(lr, smoothed_loss))
        
        f.close()

    def smooth_loss(self, flow):
        dx = (flow[:,:,0:-2,:] - flow[:,:,1:-1,:])**2
        dy = (flow[:,:,:,0:-2] - flow[:,:,:,1:-1])**2
        return (dx.sum() + dy.sum()) / (self.batch_size * 64**2)

    def train(self, epoch):
        self.model.train()
        print("Epoch {0}/{1}".format(epoch, self.n_epochs))
        t = tqdm(self.train_loader)
        loss_data_avg = 0.0
        loss_smooth_avg = 0.0
        loss_avg = 0.0
        n = 1
        
        for param_group in self.optimizer.param_groups:
            current_lr = param_group['lr']

        for batch_idx, ims in enumerate(t):
            ims = ims.to(self.device)
            
            self.optimizer.zero_grad()
            out_forward, out_backward, flow_forward, flow_backward = self.model(ims)
            
            loss_data = 1e-5*self.loss_charb(out_forward, ims[:,1:2,:,:]) + 1e-5*self.loss_charb(out_backward, ims[:,0:1,:,:])
            loss_smooth = self.smooth_loss(flow_forward) + self.smooth_loss(flow_backward)

            loss = loss_data + loss_smooth

            loss_data_avg += (loss_data.item() - loss_data_avg) / n
            loss_smooth_avg += (loss_smooth.item() - loss_smooth_avg) / n
            loss_avg += (loss.item() - loss_avg) / n
            n += 1

            loss.backward()

            self.optimizer.step()
            
            t.set_postfix(loss_data=loss_data_avg, loss_smooth=loss_smooth_avg, lr=current_lr)
            
        self.loss.append(loss_avg)

    def test(self, epoch):
        self.model.eval()
        t = tqdm(self.test_loader)
        n = 1
        loss_data_avg = 0.0
        loss_smooth_avg = 0.0
        loss_avg = 0.0

        with torch.no_grad():
            for batch_idx, ims in enumerate(t):
                ims = ims.to(self.device)
                        
                out_forward, out_backward, flow_forward, flow_backward = self.model(ims)
                
                loss_data = 1e-5*self.loss_charb(out_forward, ims[:,1:2,:,:]) + 1e-5*self.loss_charb(out_backward, ims[:,0:1,:,:])
                loss_smooth = self.smooth_loss(flow_forward) + self.smooth_loss(flow_backward)

                loss = loss_data + loss_smooth

                loss_data_avg += (loss_data.item() - loss_data_avg) / n
                loss_smooth_avg += (loss_smooth.item() - loss_smooth_avg) / n
                loss_avg += (loss.item() - loss_avg) / n
                n += 1

                t.set_postfix(loss_data=loss_data_avg, loss_smooth=loss_smooth_avg)
            
        self.loss_val.append(loss_avg)

if (__name__ == '__main__'):
    parser = argparse.ArgumentParser(description='Train neural network')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
    parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float,
                    metavar='LR', help='Learning rate')
    parser.add_argument('--wd', '--weigth-decay', default=0.0, type=float,
                    metavar='WD', help='Weigth decay')    
    
    parsed = vars(parser.parse_args())

    deepvel = optical_flow(batch_size=64)

    deepvel.init_optimize(100, lr=parsed['lr'], resume=parsed['resume'], weight_decay=parsed['wd'])
    deepvel.optimize()