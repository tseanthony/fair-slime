import numpy as np
from visdom import Visdom

from sslime.core.config import config as cfg


class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, config, env='main'):
        if config:
            self.viz = Visdom(**config)
        else:
            self.viz = None
        self.env = env
        self.plots = {}

    def plot(self, var_name, split_name, x, y, title=None):
        if not self.viz: return
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(
                X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title or var_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(
                X=np.array([x]), Y=np.array([y]), env=self.env,
                win=self.plots[var_name], name=split_name, update = 'append')


def get_visdom_plotter():
    return VisdomLinePlotter(cfg.VISDOM.CONFIG, cfg.VISDOM.ENV)
