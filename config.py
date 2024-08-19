import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--epochs', type=int, default=20,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=0.,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--batch_size', type=int, default=5,
                    help='Number of samples in each batch.')
parser.add_argument('--lr_scheduler_step', type=int, default=5,
                    help='Number of steps for learning rate scheduler.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
