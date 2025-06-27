import os.path as osp
import cpfssr.archs
import cpfssr.data
import cpfssr.models
import cpfssr.losses
import basicsr.metrics
from basicsr.train import train_pipeline

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
