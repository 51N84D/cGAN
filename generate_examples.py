import torch
import numpy as np
from models.cgan import cGAN 
import shutil
import os
import cv2
from torch.autograd import Variable
from PIL import Image

def mkdir(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)



def generate_examples(train_data, specs, save_dir, num_examples=10):

    for i in range(specs.shape[0]):
        mkdir(save_dir + str(i))
        audio = np.expand_dims(specs[i], axis = 0)
        #audio = np.tile(audio, (num_examples,1, 1))

        audio = torch.FloatTensor(audio).to(device)
        for j in range(num_examples):
            noise = Variable(FloatTensor(np.random.normal(0, 1, (1, 128))))
            image_outputs = model(noise, audio.unsqueeze(1))
            imgs = (1+image_outputs.cpu().detach().numpy())*255.0/2.0
            img = np.transpose(np.squeeze(imgs), (2,1,0)).astype(np.uint8)
            im = Image.fromarray(img)
            im.save(save_dir + str(i) + '/' + str(j) + '.png')
        #Save train_example
        train_example = np.squeeze(train_data[10])
        train_example = (2*train_example / 255.0) - 1.0
        train_example = ((1+train_example)*255/2).astype(np.uint8)
        im = Image.fromarray(train_example)
        im.save(save_dir + str(i) + '/' + 'example' + '.png')

        
def main():
    #Go through each audio class, and generate a few examples
    data_dir = './data/'
    spec_file = 'specs.npy'
    train_data_path = '/data/cifar10/image_numpy/train_images.npy'
    descriptions_file = './data/descriptions.txt'
    save_dir = './examples/'
    mkdir(save_dir)
    epoch = 50

    specs = np.load(data_dir + spec_file)
    n = specs.shape[0]

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    model = cGAN().to(device)
    checkpoint_path = '/data/cifar10/exp/checkpoints/model_state.' + str(epoch) + '.pth'
    model.load_state_dict(torch.load(checkpoint_path)['g_state_dict'])

    model.eval()

    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    train_data = np.load(train_data_path)
    generate_examples(train_data, specs, save_dir)


if __name__== "__main__":
    main()
