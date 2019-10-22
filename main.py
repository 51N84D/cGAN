import dataloaders
import argparse
import numpy as np
import torch
from torch import optim, nn
import time
import librosa
from util import *
import shutil
import os
import csv
import models.cifar as models
from models.cgan import cGAN, Discriminator
from torch.autograd import Variable
import re

def mkdir(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

def sorted_alphanumeric(data):

    '''Given input list as data, returns the list sorted in an alphanumeric manner'''

    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(data, key=alphanum_key)

def load_checkpoint(model_state, filename='checkpoint.pth.tar'):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 1

    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        g_model.load_state_dict(checkpoint['g_state_dict'])
        g_optimizer.load_state_dict(checkpoint['g_optimizer'])

        d_model.load_state_dict(checkpoint['d_state_dict'])
        d_optimizer.load_state_dict(checkpoint['d_optimizer']) 
        
        model_state['g_model'] = g_model
        model_state['d_model'] = d_model
        model_state['g_optimizer'] = g_optimizer
        model_state['d_optimizer'] = d_optimizer

        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model_state, start_epoch

def accuracy(output, target):
    """Computes the accuracy for multiple binary predictions"""
    pred = output >= 0.5
    truth = target >= 0.5
    acc = pred.eq(truth).sum() / target.numel()
    return acc

def tocuda(x):
    if not args.no_cuda:
        return x.cuda()
    return x

# Training settings
parser = argparse.ArgumentParser(description='Observer Network')
parser.add_argument('--train-batch', default=100, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', type=float, default=0.0002, metavar='LR',
                    help='learning rate (default: 0.0002)')
parser.add_argument('--schedule', type=int, nargs='+', default=[20, 50, 120, 180],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.5, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--image-data', type=str, default='/data/cifar10/image_numpy/',
                    help='path to image data')
parser.add_argument('--audio-data', type=str, default='./data/specs.npy',
                    help='path to audio data')
parser.add_argument('--no_cuda', type=int, default=0,
                    help='(do not) use gpu')
parser.add_argument('--log_interval', type=int, default=1,
                    help='print statistics')
parser.add_argument('--exp_dir', type=str, default='/data/cifar10/exp',
                    help='experiment directory')
parser.add_argument('--new_exp', type=int, default=0,
                    help='make new dir (erases previous dir)')
parser.add_argument('--save_epochs', type=int, default=50,
                    help='save every n epochs')
parser.add_argument('--noise_size', type=int, default=128,
                    help='size of noise vector')
parser.add_argument('--resume', type=int, default=1,
                    help='resume training from latest epoch (in exp_dir)')
parser.add_argument('--beta', type=float, default=0.5,
                    help='beta for adam optimizer')
parser.add_argument('--beta1', type=float, default=0.999,
                    help='beta1 for adam optimizer')
parser.add_argument('--flip_iter', type=int, default=1000,
                    help='flip labels every n iterations')




args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
if args.new_exp or not os.path.isdir(args.exp_dir):
    mkdir(args.exp_dir)

exp_dir = args.exp_dir
save_epochs = args.save_epochs

train_image_path = args.image_data + 'train_images.npy'
train_labels_path = args.image_data + 'train_labels.npy'
test_image_path = args.image_data + 'test_images.npy'
test_labels_path = args.image_data + 'test_labels.npy'
audio_path = args.audio_data


train_paths = {}
train_paths['image'] = train_image_path
train_paths['labels'] = train_labels_path
train_paths['audio'] = audio_path


train_loader = torch.utils.data.DataLoader(
    dataloaders.ImageConceptDataset(train_paths),
    batch_size=args.train_batch, shuffle=True)

g_model = cGAN().to(device)
d_model = Discriminator().to(device)

g_optimizer = optim.Adam(g_model.parameters(), lr=args.lr, betas=(args.beta, args.beta1))
#d_optimizer = optim.Adam(d_model.parameters(), lr=args.lr, betas=(args.beta, args.beta1))
d_optimizer = optim.SGD(d_model.parameters(), lr=args.lr)

model_state = {}
model_state['g_model'] = g_model
model_state['d_model'] = d_model
model_state['g_optimizer'] = g_optimizer
model_state['d_optimizer'] = d_optimizer

start_epoch = 1
if args.resume and os.path.isdir(exp_dir + '/checkpoints/'):
    cpkts = sorted_alphanumeric(os.listdir(exp_dir + '/checkpoints/'))
    #Take the latest cpkt
    checkpoint_file = cpkts[-1]
    model_state, start_epoch = load_checkpoint(model_state, filename=exp_dir + '/checkpoints/' + checkpoint_file)

g_model = model_state['g_model']
d_model = model_state['d_model']
g_optimizer = model_state['g_optimizer']
d_optimizer = model_state['d_optimizer']

FloatTensor = torch.cuda.FloatTensor if not args.no_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if not args.no_cuda else torch.LongTensor

bce_loss = nn.MSELoss()

#--------------------------TRAINING/TESTING FUNCTIONS--------------------------
def train(epoch):
    g_losses = 0
    d_losses = 0
    real_losses = 0
    fake_losses = 0
    for batch_idx, (image_inputs,labels,audio) in enumerate(train_loader):
        image_inputs = image_inputs.to(device)
        labels = labels.to(device)
        audio = audio.to(device)

        valid = Variable(FloatTensor(image_inputs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(image_inputs.shape[0], 1).fill_(0.0), requires_grad=False)
        valid_smooth = Variable(FloatTensor(image_inputs.shape[0], 1).fill_(1.0), requires_grad=False)


        real_imgs = Variable(image_inputs.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))

        #instance noise:
        noise1 = Variable(tocuda(torch.Tensor(image_inputs.size()).normal_(0, 0.1 * (start_epoch + args.epochs - epoch) / (start_epoch + args.epochs))))
        noise2 = Variable(tocuda(torch.Tensor(image_inputs.size()).normal_(0, 0.1 * (start_epoch + args.epochs - epoch) / (start_epoch + args.epochs))))
        noise3 = Variable(tocuda(torch.Tensor(image_inputs.size()).normal_(0, 0.1 * (start_epoch + args.epochs - epoch) / (start_epoch + args.epochs))))

        #--------------GENERATOR---------------------
        g_optimizer.zero_grad()
        #sample noise:
        noise = Variable(FloatTensor(np.random.normal(0, 1, (image_inputs.shape[0], args.noise_size))))
        gen_imgs = g_model(noise, audio.unsqueeze(1))

        validity = d_model(gen_imgs, labels)
        g_loss = bce_loss(validity, valid)
        g_loss.backward()
        g_optimizer.step()

        g_losses += g_loss
        #--------------DISCRIMINATOR-----------------
        d_optimizer.zero_grad()

        # Loss for real images
        validity_real = d_model(real_imgs, labels)
        #if batch_idx / args.flip_iter == 0:
        #    d_real_loss = bce_loss(validity_real, fake)
        #else:
        d_real_loss = bce_loss(validity_real, valid_smooth)
        d_real_loss.backward()

        # Loss for fake images
        validity_fake = d_model(gen_imgs.detach(), labels)
        #if batch_idx / args.flip_iter == 0:
        #    d_fake_loss = bce_loss(validity_fake, valid)
        #else:
        d_fake_loss = bce_loss(validity_fake, fake)
        d_fake_loss.backward()

        fake_acc = float(torch.sum(validity_fake <= 0.5))/float(validity_fake.shape[0]) 
        real_acc = float(torch.sum(validity_real >= 0.5))/float(validity_real.shape[0]) 
       

        # Total discriminator loss
        d_loss = d_real_loss + d_fake_loss
        #d_loss.backward()
        d_optimizer.step()

        d_losses += d_loss
        real_losses += d_real_loss
        fake_losses += d_fake_loss

        if batch_idx % args.log_interval == 0:
           print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, start_epoch + args.epochs, batch_idx, len(train_loader), d_loss.item(), g_loss.item())
                )
    print(
        "Averages: [Epoch %d/%d] [D loss: %f] [G loss: %f]"
        % (epoch, args.epochs, d_losses/len(train_loader), g_losses/len(train_loader))
        )

    #Save loss
    with open(exp_dir + '/losses.csv', 'a') as loss_file:
        #First write headers:
        headers = ['d_loss_real', 'd_loss_fake', 'g_loss']
        writer = csv.writer(loss_file)
        if epoch == 1:
            writer.writerow(headers)
        #d_avg = d_losses.detach().cpu().numpy()/len(train_loader)
        real_losses_avg = real_losses.detach().cpu().numpy()/len(train_loader)
        fake_losses_avg = fake_losses.detach().cpu().numpy()/len(train_loader)
        g_avg = g_losses.detach().cpu().numpy()/len(train_loader)

        writer.writerow([real_losses_avg, fake_losses_avg, g_avg])
        #loss_file.write(str(train_loss / len(train_loader.dataset))+'\n')
    
    #Save state
    if epoch % save_epochs == 0:
        if not os.path.isdir(exp_dir + '/checkpoints/'):
            mkdir(exp_dir + '/checkpoints/')
 
        state = {'epoch': epoch + 1, 'g_state_dict': g_model.state_dict(),
             'g_optimizer': g_optimizer.state_dict(), 'd_state_dict': d_model.state_dict(),
             'd_optimizer': d_optimizer.state_dict()}
        torch.save(state, exp_dir + '/checkpoints/model_state.%d.pth' % (epoch))
        

#------------------------------------------------------------------------------
for epoch in range(start_epoch, start_epoch + args.epochs + 1):
    train(epoch)



