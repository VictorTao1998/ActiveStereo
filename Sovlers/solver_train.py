import torch
import os
import time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from datasets.messytable import MessytableDataset
from Models import get_model
from Losses import get_losses
from Metrics.metrics import epe_metric
from Metrics.metrics import tripe_metric
from tensorboardX import SummaryWriter
from utils.util import *
import pdb

class TrainSolver(object):

    def __init__(self, args, config):

        self.config = config
        self.args = args


        self.max_disp = self.config.ARGS.MAX_DISP
        self.loss_name = "XTLoss"

        train_dataset = MessytableDataset(config.SPLIT.TRAIN, gaussian_blur=False, color_jitter=False, debug=args.debug, sub=600)
        val_dataset = MessytableDataset(config.SPLIT.VAL, gaussian_blur=False, color_jitter=False, debug=args.debug, sub=100, isVal=True)

        self.TrainImgLoader = torch.utils.data.DataLoader(train_dataset, batch_size=config.SOLVER.BATCH_SIZE,
                                                     shuffle=True, num_workers=config.SOLVER.NUM_WORKER, drop_last=True)

        self.ValImgLoader = torch.utils.data.DataLoader(val_dataset, batch_size=config.SOLVER.BATCH_SIZE,
                                                   shuffle=False, num_workers=config.SOLVER.NUM_WORKER, drop_last=False)


        self.model = get_model(self.config)

        self.crit = get_losses(self.loss_name, max_disp=self.max_disp)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.SOLVER.LR_CASCADE)

        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[2,3,4], gamma=0.5)

        self.writer = SummaryWriter(args.logdir)

        self.global_step = 0
        self.epoch = 0
        self.best_epe = 9999999999

    def save_checkpoint(self, best=False):

        ckpt_root = os.path.join(self.args.logdir, 'checkpoints')

        if not os.path.exists(ckpt_root):
            os.makedirs(ckpt_root) 
        
        ckpt_name = 'ep_{:d}.pth'.format(self.epoch)

        if best:
            ckpt_name = 'best_epe_{:f}.pth'.format(self.best_epe)

        states = {
            'epoch': self.epoch,
            'best_epe': self.best_epe,
            'global_step': self.global_step,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict()
        }
        ckpt_full = os.path.join(ckpt_root, ckpt_name)
        
        torch.save(states, ckpt_full)
    
    def load_checkpoint(self):

        ckpt_root = os.path.join(self.args.loadmodel)

        states = torch.load(ckpt_full, map_location=lambda storage, loc: storage)

        self.epoch = states['epoch']
        self.best_epe = states['best_epe']
        self.global_step = states['global_step']
        self.model.load_state_dict(states['model_state'])
        self.optimizer.load_state_dict(states['optimizer_state'])
        self.scheduler.load_state_dict(states['scheduler_state'])

    def run(self):
        self.model = nn.DataParallel(self.model)
        self.model.cuda()
        
        print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in self.model.parameters()])))

        if self.args.loadmodel:
            self.load_checkpoint()
            print('[{:d}] Model loaded.'.format(self.args.loadmodel))
        
        for epoch in range(self.config.SOLVER.EPOCHS):
            self.model.train()
            self.epoch = epoch
            for i, data_batch in enumerate(self.TrainImgLoader):
                start_time = time.time()
                
                self.model.train()
                imgL, imgR, disp_L = data_batch['img_sim_L'], data_batch['img_sim_R'], data_batch['img_disp_l']
                imgL, imgR, disp_L = imgL.cuda(), imgR.cuda(), disp_L.cuda()

                disp_L = F.interpolate(disp_L, scale_factor=0.5, mode='nearest',
                             recompute_scale_factor=False)  # [bs, 1, H, W]
                
                self.optimizer.zero_grad()
                #pdb.set_trace()
                disp_pred_left = self.model(imgL, imgR)
                
                #pdb.set_trace()

                loss = self.crit(imgL, imgR, disp_pred_left)
                loss.backward()
                self.optimizer.step()
                
                elapsed = time.time() - start_time
                #print(disp_L.shape, disp_pred_left.shape)
                train_EPE_left = epe_metric(disp_L, disp_pred_left, self.max_disp)
                train_3PE_left = tripe_metric(disp_L, disp_pred_left, self.max_disp)

                
                print(
                    'Epoch[{:d}/{:d}] iter[{:d}/{:d}] Train Loss = {:.6f}, EPE = {:.3f} px, 3PE = {:.3f}%, time = {:.3f}s.'.format(
                        self.epoch, self.config.SOLVER.EPOCHS,
                        i, len(self.TrainImgLoader),
                        loss.item(),
                        train_EPE_left, 
                        train_3PE_left * 100,
                        elapsed
                    )
                )
                #print(imgL.shape, disp_pred_left.shape, disp_L.shape)
                if self.global_step % self.args.summary_freq == 0:
                    scalar_output = {'reproj_loss': loss.item(), 'EPE': train_EPE_left, 'bad3': train_3PE_left}
                    save_scalars(self.writer, 'train', scalar_output, self.global_step)

                    save_images(self.writer, 'train', {'img_L':[imgL.detach().cpu()]}, self.global_step)   
                    save_images(self.writer, 'train', {'img_R':[imgR.detach().cpu()]}, self.global_step)
                    save_images(self.writer, 'train', {'disp_gt':[disp_L.detach().cpu()]}, self.global_step)   
                    save_images(self.writer, 'train', {'disp_pred':[disp_pred_left.detach().cpu()]}, self.global_step)

                self.global_step += 1

            self.scheduler.step()

            start_time = time.time()
            self.model.eval()
            with torch.no_grad():
                
                val_EPE_metric_left = 0.0
                val_TriPE_metric_left = 0.0
                N_total = 0.0
                
                for i, val_batch in enumerate(self.ValImgLoader):
                    imgL, imgR, disp_L = val_batch['img_sim_L'], val_batch['img_sim_R'], val_batch['img_disp_l']
                    imgL, imgR, disp_L = imgL.cuda(), imgR.cuda(), disp_L.cuda()
                    
                    disp_L = F.interpolate(disp_L, scale_factor=0.5, mode='nearest',
                             recompute_scale_factor=False)  # [bs, 1, H, W]

                    N_curr = imgL.shape[0]
                    
                    #print(imgL.shape, imgR.shape)
                    disp_pred_left = self.model(imgL, imgR)
                    
                    val_EPE_metric_left += epe_metric(disp_L, disp_pred_left, self.max_disp) * N_curr 
                    val_TriPE_metric_left += tripe_metric(disp_L, disp_pred_left, self.max_disp) * N_curr

                    N_total += N_curr

                    scalar_output = {'reproj_loss': loss.item(), 'EPE': train_EPE_left, 'bad3': train_3PE_left}
                    save_scalars(self.writer, 'validation', scalar_output, self.global_step)

                    save_images(self.writer, 'validation', {'img_L':[imgL.detach().cpu()]}, self.global_step)   
                    save_images(self.writer, 'validation', {'img_R':[imgR.detach().cpu()]}, self.global_step)
                    save_images(self.writer, 'validation', {'disp_gt':[disp_L.detach().cpu()]}, self.global_step)   
                    save_images(self.writer, 'validation', {'disp_pred':[disp_pred_left.detach().cpu()]}, self.global_step)
                
                val_EPE_metric_left /= N_total
                val_TriPE_metric_left /= N_total
                

                elapsed = time.time() - start_time
                print(
                    'Epoch:[{:d}/{:d}] Validation : EPE = {:.6f} px, 3PE = {:.3f} %, time = {:.3f} s.'.format(
                        self.epoch, self.config.SOLVER.EPOCHS,
                        val_EPE_metric_left, 
                        val_TriPE_metric_left * 100, 
                        elapsed / N_total
                    )
                )

                self.save_checkpoint()
                print('')
                print('Epoch[{:d}] Model saved.'.format(self.epoch))

                if val_EPE_metric_left < self.best_epe:
                    self.best_epe = val_EPE_metric_left
                    self.save_checkpoint(best=True)
                    print('')
                    print('Best Epoch[{:d}] Model saved.'.format(self.epoch))
            



