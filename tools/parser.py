import argparse
from multiprocessing import cpu_count


def parse_args(args=None):
    parser = argparse.ArgumentParser(description='GAN implementation')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log', action='store_false', default=True,
                        help='log the run')
    parser.add_argument('--has-nan', action='store_true', default=False,
                        help='data has nan values')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-frequency', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-frequency', type=int, default=50, metavar='N',
                        help='how many epochs to wait before saving the model (default: 50)')
    parser.add_argument('--test-frequency', type=int, default=5, metavar='N',
                        help='how often to test the model on the test dataset')
    parser.add_argument('--dataset', type=str, default='taal',
                        help='train and test dataset')
    parser.add_argument('--subfolder', type=str, default='unw_png',
                        help='which subfolder from the dataset to use')
    parser.add_argument('--num-channels', type=int, default=1, metavar='N',
                        help='number of channels (default: 1)')
    parser.add_argument('--beta-vae', action='store_true', help='Beta-VAE')
    parser.add_argument('--learn-variance', action='store_true', help='learn Variance')
    parser.add_argument('--result-dir', type=str, default='res',
                        help='directory where to save output images')
    parser.add_argument("-n_jobs", "--num-workers", default=cpu_count(), type=int,
                        help="Number of worker processes used to load data.")
    parser.add_argument("--lr", "--learning-rate", default=1e-3, type=float, help="Learning rate")
    parser.add_argument('--model-name', type=str,
                        help='name of the model to be loaded')
    parser.add_argument('--validate', action='store_true', help='Run test only')
    # parser.add_argument('--decoder_type', type=str, default='gaussian', help='save output images')
    parser.add_argument('--image-size', type=int, default=128, metavar='N',
                        help='size of the images')
    parser.add_argument('--latent-dim', type=int, default=64, metavar='N',
                        help='size of the images')
    parser.add_argument("--threshold-percentage", default=0.95, type=float,
                        help="Percentage for choosing the threshold")
    parser.add_argument("--cc", default=0.05, type=float,
                        help="Coherence threshold to filter interferograms based on based on")
    parser.add_argument('--model-type', type=str, default='na', help='model to use for training')
    parser.add_argument('--no-heatmap', action='store_true', default=False, help='disables heatmap during testing')
    return parser.parse_args(args)
