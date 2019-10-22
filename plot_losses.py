import matplotlib.pyplot as plt
import os
import shutil
import csv
import numpy as np

def mkdir(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)


losses_path = '/data/cifar10/exp/losses.csv'
plot_dir = './plots'
mkdir(plot_dir)

losses = []
with open(losses_path) as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    headers = next(spamreader, None)
    data = [list(map(float, row)) for row in csv.reader(csvfile)]

data = np.asarray(data)
assert len(headers) == data.shape[1]
for idx, loss_name in enumerate(headers):
    loss_data = data[:,idx]
    plt.plot(loss_data, label=loss_name)
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.title('GAN loss')

plt.legend()
plt.savefig(plot_dir + '/' + 'gan_loss.png')
