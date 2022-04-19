from torch.utils.tensorboard import SummaryWriter

class Metric():
  def __init__(self,name):
    self.min = 1
    self.max = 0
    self.loss = 0
    self.sum = 0
    self.count = 0
    self.ave = 0
    self.index = 0
    self.writer = SummaryWriter(name)
    self.name = name

  def step(self,loss):
    self.sum += loss
    self.count += 1
    self.index += 1
    self.ave = self.sum/self.count
    self.max = max(self.max,loss)
    self.min = min(self.min,loss)
    self.loss = loss

  def reset(self):
    self.loss = 0
    self.sum = 0
    self.count = 0
    self.ave = 0

  def log(self,tag):
    self.writer.add_scalar(tag,self.ave,self.index)
