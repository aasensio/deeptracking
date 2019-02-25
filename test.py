import numpy as np
import matplotlib.pyplot as pl
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable
import h5py
import shutil
from tqdm import tqdm
import matplotlib.animation as animation
from astropy.io import fits
import glob
import os
from skimage.feature import register_translation

import model

class optical_flow(object):
    def __init__(self, n_pixel=256, checkpoint=None):
        self.cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda else "cpu")
        self.n_pixel = n_pixel
        
        self.model = model.network(n_pixel=n_pixel, device=self.device).to(self.device)

        if (checkpoint is None):
            files = glob.glob('trained/*.pth')
            self.checkpoint = max(files, key=os.path.getctime)
            
        else:
            self.checkpoint = '{0}.pth'.format(checkpoint)
                        
        print("=> loading checkpoint '{}'".format(self.checkpoint))
        if (self.cuda):
            checkpoint = torch.load(self.checkpoint)
        else:
            checkpoint = torch.load(self.checkpoint, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(checkpoint['state_dict'])        
        print("=> loaded checkpoint '{}'".format(self.checkpoint))

    def test(self):
        pl.close('all')

        self.model.eval()

        steps = (slice(None,None,2),slice(None,None,2))

        f0 = fits.open('/net/nas/proyectos/fis/aasensio/deep_learning/deepvel_jess/CaK/destretched_02520.fits')
        f1 = fits.open('/net/nas/proyectos/fis/aasensio/deep_learning/deepvel_jess/CaK/destretched_02534.fits')

        ims = np.zeros((1,2,self.n_pixel,self.n_pixel))
        ims[0,0,:,:] = f0[0].data[512:512+self.n_pixel,512:512+self.n_pixel]
        ims[0,1,:,:] = f1[0].data[512:512+self.n_pixel,512:512+self.n_pixel]

        minim = np.min(ims, axis=(2,3))
        maxim = np.max(ims, axis=(2,3))

        ims = (ims - minim[:,:,None,None]) / (maxim[:,:,None,None] - minim[:,:,None,None])

        ims = torch.from_numpy(ims.astype('float32'))
        ims = ims.to(self.device)     
        
        out_forward, out_backward, flow_forward, flow_backward = self.model(ims)

        output = out_forward.cpu().data.numpy()
        flow = flow_forward.cpu().data.numpy()  

        flowx = flow[0,0,:,:]
        flowy = flow[0,1,:,:]

        ims = ims.cpu().data.numpy()        
        
        f, ax = pl.subplots(nrows=2, ncols=4, figsize=(14,6))        
        ax[0,0].imshow(ims[0,0,:,:])
        ax[0,1].imshow(ims[0,1,:,:])
        ax[0,2].imshow(flow[0,0,:,:])
        ax[0,3].imshow(flow[0,1,:,:])

        ax[1,0].imshow(output[0,0,:,:])
        ax[1,1].imshow(ims[0,0,:,:]-output[0,0,:,:])
        ax[1,2].imshow(ims[0,1,:,:]-output[0,0,:,:])        
        
        ax[0,0].set_title('Input 1')
        ax[0,1].set_title('Input 2')
        ax[0,2].set_title('Flow x')
        ax[0,3].set_title('Flow y')

        ax[1,0].set_title('NN')
        ax[1,1].set_title('NN-I1 {0}'.format(np.std(ims[0,0,:,:]-output[0,0,:,:])))
        ax[1,2].set_title('NN-I2 {0}'.format(np.std(ims[0,1,:,:]-output[0,0,:,:])))

        f, ax = pl.subplots()
        x = np.arange(self.n_pixel)
        y = np.arange(self.n_pixel)
        X, Y = np.meshgrid(x, y)
        ax.imshow(ims[0,0,:,:])
        Q = ax.quiver(X[steps], Y[steps], 0.5*self.n_pixel*flowx[steps], 0.5*self.n_pixel*flowy[steps], scale=10, units='inches', color='yellow')
        qk = ax.quiverkey(Q, 0.9, 0.9, 1, r'$2 \frac{m}{s}$', labelpos='E',  coordinates='figure', color='k')
        

        pl.show()
        stop()

    def updatefig(self, *args):
        f0 = fits.open(self.files[self.loop])
        f1 = fits.open(self.files[self.loop+1])

        with torch.no_grad():

            ims = np.zeros((1,2,self.n_pixel,self.n_pixel))
            ims[0,0,:,:] = f0[0].data[self.origin:self.origin+self.n_pixel,self.origin:self.origin+self.n_pixel]
            ims[0,1,:,:] = f1[0].data[self.origin:self.origin+self.n_pixel,self.origin:self.origin+self.n_pixel]

            minim = np.min(ims, axis=(2,3))
            maxim = np.max(ims, axis=(2,3))

            ims = (ims - minim[:,:,None,None]) / (maxim[:,:,None,None] - minim[:,:,None,None])

            shift, error, diffphase = register_translation(self.reference, ims[0,1,:,:])
            shift = [int(f) for f in shift]                            
            ims[0,1,:,:] = np.roll(ims[0,1,:,:], shift, axis=(0,1))

            shift, error, diffphase = register_translation(self.reference, ims[0,0,:,:])
            shift = [int(f) for f in shift]                            
            ims[0,0,:,:] = np.roll(ims[0,0,:,:], shift, axis=(0,1))

            ims = torch.from_numpy(ims.astype('float32'))
            ims = ims.to(self.device)     
            
            out_forward, flow_forward = self.model(ims, backward=False)

            output = out_forward.cpu().data.numpy()
            flow = flow_forward.cpu().data.numpy()  

            ims = ims.cpu().data.numpy()

            flowx = flow[0,0,:,:]
            flowy = flow[0,1,:,:]

        f0.close()
        f1.close()

        flowx *= self.scale
        flowy *= self.scale
        
        self.im1.set_array(np.flip(ims[0,0,:,:], axis=0))
        self.im2.set_array(np.flip(ims[0,1,:,:], axis=0))
        self.flowx.set_array(np.flip(flow[0,0,:,:], axis=0))
        self.flowy.set_array(np.flip(flow[0,1,:,:], axis=0))
        self.Q.set_UVC(self.n_pixel*flowx[self.steps], self.n_pixel*flowy[self.steps])

        self.loop += 1
        self.pbar.update(1)

        return self.im1, self.im2, self.flowx, self.flowy

    def movie(self):

        self.origin = 512
        self.model.eval()

        self.scale = 0.18

        self.files = glob.glob('/net/nas/proyectos/fis/aasensio/deep_learning/deepvel_jess/CaK/destretched_*.fits')
        self.files.sort()

        self.n_frames = len(self.files) - 2

        self.loop = 0

        f0 = fits.open(self.files[self.loop])
        f1 = fits.open(self.files[self.loop+1])

        self.reference = f0[0].data[self.origin:self.origin+self.n_pixel,self.origin:self.origin+self.n_pixel]

        with torch.no_grad():

            ims = np.zeros((1,2,self.n_pixel,self.n_pixel))
            ims[0,0,:,:] = f0[0].data[self.origin:self.origin+self.n_pixel,self.origin:self.origin+self.n_pixel]
            ims[0,1,:,:] = f1[0].data[self.origin:self.origin+self.n_pixel,self.origin:self.origin+self.n_pixel]

            minim = np.min(ims, axis=(2,3))
            maxim = np.max(ims, axis=(2,3))

            ims = (ims - minim[:,:,None,None]) / (maxim[:,:,None,None] - minim[:,:,None,None])

            shift, error, diffphase = register_translation(self.reference, ims[0,1,:,:])
            shift = [int(f) for f in shift]                            
            ims[0,1,:,:] = np.roll(ims[0,1,:,:], shift, axis=(0,1))

            shift, error, diffphase = register_translation(self.reference, ims[0,0,:,:])
            shift = [int(f) for f in shift]                            
            ims[0,0,:,:] = np.roll(ims[0,0,:,:], shift, axis=(0,1))

            ims = torch.from_numpy(ims.astype('float32'))
            ims = ims.to(self.device)     
            
            out_forward, flow_forward = self.model(ims, backward=False)

            output = out_forward.cpu().data.numpy()
            flow = flow_forward.cpu().data.numpy()  

            flowx = flow[0,0,:,:]
            flowy = flow[0,1,:,:]

            ims = ims.cpu().data.numpy()

        f0.close()
        f1.close()

        x = np.arange(self.n_pixel)
        y = np.arange(self.n_pixel)
        X, Y = np.meshgrid(x, y)
        X = X * self.scale
        Y = Y * self.scale

        flowx *= self.scale
        flowy *= self.scale

        self.steps = (slice(None,None,2),slice(None,None,2))

        f, ax = pl.subplots(nrows=2, ncols=2, figsize=(12,10))
        self.im1 = ax[0,0].imshow(np.flip(ims[0,0,:,:], axis=0), extent=[0,self.n_pixel*self.scale,0,self.n_pixel*self.scale])
        self.Q = ax[0,0].quiver(X[self.steps], Y[self.steps], self.n_pixel*flowx[self.steps], self.n_pixel*flowy[self.steps], scale=10, units='inches', headwidth=3, headlength=3, color='yellow')
        self.im2 = ax[0,1].imshow(np.flip(ims[0,1,:,:], axis=0), extent=[0,self.n_pixel*self.scale,0,self.n_pixel*self.scale])
        self.flowx = ax[1,0].imshow(np.flip(flowx, axis=0), extent=[0,self.n_pixel*self.scale,0,self.n_pixel*self.scale])
        self.flowy = ax[1,1].imshow(np.flip(flowy, axis=0), extent=[0,self.n_pixel*self.scale,0,self.n_pixel*self.scale])
        qk = ax[0,0].quiverkey(self.Q, 0.9, 0.9, 1, r'$2 \frac{m}{s}$', labelpos='E',  coordinates='figure', color='k')
        
        ax[0,0].set_title('Input 1')
        ax[0,1].set_title('Input 2')
        ax[1,0].set_title('Flow x')
        ax[1,1].set_title('Flow y')

        self.pbar = tqdm(total=self.n_frames)

        self.loop += 1

        frames = self.n_frames
        #frames = 20

        ani = animation.FuncAnimation(f, self.updatefig, interval=100, blit=True, frames=frames-2)

        ani.save('CaK.mp4')                

        self.pbar.close()

        

optical_flow_network = optical_flow(n_pixel=128)
#optical_flow_network.test()
optical_flow_network.movie()
