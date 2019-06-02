import json
import os
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt


class LogReport():
    def __init__(self, log_dir, log_name='log'):
        self.log_dir = log_dir
        self.log_name = log_name
        self.log_ = []

    def __call__(self, log):
        self.log_.append(log)
        with open(os.path.join(self.log_dir, self.log_name), 'w', encoding='UTF-8') as f:
            json.dump(self.log_, f, indent=4)
    
    def save_lossgraph(self):
        epoch = []
        gen_loss = []
        dis_loss = []

        for l in self.log_:
            epoch.append(l['epoch'])
            gen_loss.append(l['gen/loss'])
            dis_loss.append(l['dis/loss'])

        epoch = np.asarray(epoch)
        gen_loss = np.asarray(gen_loss)
        dis_loss = np.asarray(dis_loss)

        plt.plot(epoch, gen_loss)
        plt.xlabel('epoch')
        plt.ylabel('loss_gen')
        plt.savefig(os.path.join(self.log_dir, 'lossgraph_gen.pdf'))
        plt.close()

        plt.plot(epoch, dis_loss)
        plt.xlabel('epoch')
        plt.ylabel('loss_dis')
        plt.savefig(os.path.join(self.log_dir, 'lossgraph_dis.pdf'))
        plt.close()


class TestReport():
    def __init__(self, log_dir, log_name='log_test'):
        self.log_dir = log_dir
        self.log_name = log_name
        self.log_ = []

    def __call__(self, log):
        self.log_.append(log)
        with open(os.path.join(self.log_dir, self.log_name), 'w', encoding='UTF-8') as f:
            json.dump(self.log_, f, indent=4)
    
    def save_lossgraph(self):
        epoch = []
        mse = []
        psnr = []
        
        for l in self.log_:
            epoch.append(l['epoch'])
            mse.append(l['mse'])
            psnr.append(l['psnr'])

        epoch = np.asarray(epoch)
        mse = np.asarray(mse)
        psnr = np.asarray(psnr)

        plt.plot(epoch, mse)
        plt.xlabel('epoch')
        plt.ylabel('mse')
        plt.savefig(os.path.join(self.log_dir, 'graph_mse.pdf'))
        plt.close()

        plt.plot(epoch, psnr)
        plt.xlabel('epoch')
        plt.ylabel('psnr')
        plt.savefig(os.path.join(self.log_dir, 'graph_psnr.pdf'))
        plt.close()
