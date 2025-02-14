import os
import time

import numpy as np
import torch
import torchvision.utils

from tools.plotting_utils import save_insar, plot_img_to_file


class BasicTrainer:
    def __init__(self, args, train_loader, test_loader, logger=None):
        if torch.cuda.is_available() and not args.no_cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.logger = logger if logger is not None else {'info': ''}

        self.learn_variance = args.learn_variance
        self.beta_vae = args.beta_vae
        # self.batch_size = args.batch_size
        self.batch_size = 169
        self.epochs = args.epochs
        self.in_channels = args.num_channels
        self.out_channels = self.in_channels
        self.log_frequency = args.log_frequency
        self.save_frequency = args.save_frequency
        self.test_frequency = args.test_frequency
        self.result_dir = args.result_dir
        self.has_nan = args.has_nan
        self.image_size = args.image_size
        self.start_time = time.strftime('%y_%m_%d_%H_%M_%S')
        self.model_type = args.model_type
        self.no_heatmap = args.no_heatmap

        self.model = torch.nn.ModuleDict({})

        self.print_all_epochs = True
        self.end_epochs_to_print = 0
        self.name = 'model'

        norm_dist = torch.distributions.normal.Normal(self.image_size / 2, self.image_size / 6)
        a = torch.exp(norm_dist.log_prob(torch.arange(self.image_size))).view(-1, 1)
        b = torch.exp(norm_dist.log_prob(torch.arange(self.image_size))).view(1, -1)

        wmap = torch.matmul(a, b)
        wmap = wmap / wmap.sum()
        self.wmap = wmap.to(self.device)

        self.test_kernel_size = self.image_size
        self.test_stride = int(self.image_size / 4)

        self.threshold_percentage = 0.95

        self.prepared_for_train = False
        self.prepared_for_test = False

    def train(self):
        self.model.to(self.device)
        self.prepare_train()
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            num_iter = 0
            train_loss = 0

            for batch_idx, (data, path, labels) in enumerate(self.train_loader):
                data = data.to(self.device)
                labels = labels.to(self.device)

                loss = self.train_step(batch_idx, data, labels, path)

                num_iter += 1
                train_loss += loss.item()

            if self.print_all_epochs or epoch >= self.epochs + 1 - self.end_epochs_to_print:
                print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(self.train_loader.dataset)))
                # print("\033[F\033[K", end="")

            save_filename = self.name + '_net_' + str(epoch) + '_' + self.start_time + '.pt'
            save_path = os.path.join(self.result_dir, save_filename)
            # if epoch % self.save_frequency == 0 or epoch == self.epochs:
            #     print('Saving model at: ' + save_path)
            #     torch.save(self.model.state_dict(), save_path)
            #     self.logger['model_name'] = save_path

    def prepare_train(self):
        pass

    def train_step(self, batch_idx, data, labels, path) -> torch.Tensor:
        pass

    def get_threshold(self, load_loss_function=False,):
        self.model.to(self.device)
        self.model.eval()

        all_losses = torch.zeros(len(self.train_loader.dataset))
        all_maps = torch.zeros((len(self.train_loader.dataset), 512, 512))
        if load_loss_function:
            self.loss_function = get_loss_function(self.logger['test_loss_function'])
        else:
            self.loss_function = LatentMSE('mean')
            self.logger['test_loss_function'] = str(self.loss_function)
        self.prepare_test()

        with torch.no_grad():
            for index, (data, path, labels) in enumerate(self.train_loader):
                data = data.to(self.device)
                # zero = data.view(data.shape[-2], data.shape[-1])[10, 500].item()
                # # zero = data.view(data.shape[-2], data.shape[-1])[500, 10].item()
                zero = -1
                # data[data == -1] = zero

                labels = labels.to(self.device)

                weight_map = torch.zeros(data.shape[2], data.shape[3]).to(self.device) + 0.00001
                prob_map = torch.zeros(data.shape[2], data.shape[3]).to(self.device)

                if self.no_heatmap:
                    data = data.unfold(2, self.image_size, self.image_size)\
                            .unfold(3, self.image_size, self.image_size)\
                            .reshape(-1, self.in_channels, self.image_size, self.image_size)

                    loss, fake = self.test_step(index, data, labels, path)
                    loss = loss.mean().item()
                else:
                    himg = data.shape[2]
                    wimg = data.shape[3]
                    hgap = wgap = 32

                    hpatch = self.image_size
                    wpatch = self.image_size

                    # batch_size = 169
                    batch_size = self.batch_size
                    patches = torch.zeros(batch_size, self.in_channels, self.image_size, self.image_size).to(self.device)
                    patches_index = 0

                    ls = torch.zeros(169).to(self.device)
                    zps = torch.zeros(169).to(self.device)
                    ind = 0

                    for starty in np.concatenate((np.arange(0, himg - hpatch, hgap), np.array([himg - hpatch])), axis=0):
                        for startx in np.concatenate((np.arange(0, wimg - wpatch, wgap), np.array([wimg - wpatch])), axis=0):
                            patches[patches_index, :, :, :] = data[:, :, starty:starty + hpatch, startx:startx + wpatch]
                            patches_index += 1

                            if patches_index == batch_size:
                                patches_index = 0
                                loss, fake = self.test_step(index, patches, labels, path)
                                zero_percentage = (patches == zero).sum(dim=(1, 2, 3)) / (
                                            patches.shape[1] * patches.shape[2] * patches.shape[3])
                                for i in range(len(loss)):
                                    ls[ind] = loss[i]
                                    zps[ind] = zero_percentage[i]
                                    ind += 1
                    if patches_index > 0:
                        loss, fake = self.test_step(index, patches, labels, path)
                        zero_percentage = (patches == zero).sum(dim=(1, 2, 3)) / (
                                patches.shape[1] * patches.shape[2] * patches.shape[3])
                        for i in range(patches_index):
                            ls[ind] = loss[i]
                            zps[ind] = zero_percentage[i]
                            ind += 1

                    patches_index = 0
                    for starty in np.concatenate((np.arange(0, himg - hpatch, hgap), np.array([himg - hpatch])), axis=0):
                        for startx in np.concatenate((np.arange(0, wimg - wpatch, wgap), np.array([wimg - wpatch])), axis=0):
                            mask = zps[patches_index] < 0.6
                            if mask == 0:
                                patches_index += 1
                                continue

                            weight_map[starty: starty + hpatch, startx: startx + wpatch] += self.wmap
                            prob_map[starty: starty + hpatch, startx: startx + wpatch] += \
                                ((1 - zps[patches_index]) * ls[patches_index]) * self.wmap

                            patches_index += 1

                    prob_map /= weight_map
                    loss = prob_map.mean().item()

                all_losses[index] = loss
                all_maps[index] = prob_map

        all_losses, _ = torch.sort(all_losses)
        threshold = all_losses[int(self.threshold_percentage * len(all_losses))]

        return all_losses, threshold, all_maps

    def test(self, save_images=False, save_all_images=False, load_loss_function=False, test_limit=-1):
        self.model.to(self.device)
        self.model.eval()

        test_loss = 0
        all_losses = torch.zeros(len(self.test_loader.dataset))
        all_prints = np.empty(len(self.test_loader.dataset), dtype='S256')
        all_maps = torch.zeros((len(self.test_loader.dataset), 512, 512))

        if load_loss_function:
            self.loss_function = get_loss_function(self.logger['test_loss_function'])
        else:
            self.loss_function = LatentMSE('mean')
            self.logger['test_loss_function'] = str(self.loss_function)
        self.prepare_test()

        test_counter = 0
        with torch.no_grad():
            for index, (data, path, labels) in enumerate(self.test_loader):
                data = data.to(self.device)
                # zero = data.view(data.shape[-2], data.shape[-1])[10, 500].item()
                # zero = data.view(data.shape[-2], data.shape[-1])[500, 10].item()
                zero = -1
                # data[data == -1] = zero

                labels = labels.to(self.device)

                weight_map = torch.zeros(data.shape[2], data.shape[3]).to(self.device) + 0.00001
                prob_map = torch.zeros(data.shape[2], data.shape[3]).to(self.device)

                # x_folds = data.unfold(2, self.test_kernel_size, self.test_stride) \
                #     .unfold(3, self.test_kernel_size, self.test_stride)
                # weight_map_folds = weight_map.unfold(0, self.test_kernel_size, self.test_stride) \
                #     .unfold(1, self.test_kernel_size, self.test_stride)
                # prob_map_folds = prob_map.unfold(0, self.test_kernel_size, self.test_stride) \
                #     .unfold(1, self.test_kernel_size, self.test_stride)

                # for y in range(0, x_folds.shape[2]):
                #     x = x_folds[:, :, y, :, :, :].view(-1, self.in_channels, self.image_size, self.image_size)
                if self.no_heatmap:
                    data = data.unfold(2, self.image_size, self.image_size)\
                            .unfold(3, self.image_size, self.image_size)\
                            .reshape(-1, self.in_channels, self.image_size, self.image_size)

                    loss, fake = self.test_step(index, data, labels, path)
                    loss = loss.mean().item()
                else:
                    himg = data.shape[2]
                    wimg = data.shape[3]
                    hgap = wgap = 32

                    hpatch = self.image_size
                    wpatch = self.image_size

                    # batch_size = 169
                    batch_size = self.batch_size
                    patches = torch.zeros(batch_size, self.in_channels, self.image_size, self.image_size).to(self.device)
                    patches_index = 0

                    ls = torch.zeros(169).to(self.device)
                    zps = torch.zeros(169).to(self.device)
                    ind = 0

                    # print('now')
                    for starty in np.concatenate((np.arange(0, himg - hpatch, hgap), np.array([himg - hpatch])), axis=0):
                        for startx in np.concatenate((np.arange(0, wimg - wpatch, wgap), np.array([wimg - wpatch])), axis=0):
                            patches[patches_index, :, :, :] = data[:, :, starty:starty + hpatch, startx:startx + wpatch]
                            patches_index += 1

                            if patches_index == batch_size:
                                patches_index = 0
                                loss, fake = self.test_step(index, patches, labels, path)
                                zero_percentage = (patches == zero).sum(dim=(1, 2, 3)) / (
                                            patches.shape[1] * patches.shape[2] * patches.shape[3])
                                for i in range(len(loss)):
                                    ls[ind] = loss[i]
                                    zps[ind] = zero_percentage[i]
                                    ind += 1
                    if patches_index > 0:
                        loss, fake = self.test_step(index, patches, labels, path)
                        zero_percentage = (patches == zero).sum(dim=(1, 2, 3)) / (
                                patches.shape[1] * patches.shape[2] * patches.shape[3])
                        for i in range(patches_index):
                            ls[ind] = loss[i]
                            zps[ind] = zero_percentage[i]
                            ind += 1
                    # print(ind)

                    patches_index = 0
                    for starty in np.concatenate((np.arange(0, himg - hpatch, hgap), np.array([himg - hpatch])), axis=0):
                        for startx in np.concatenate((np.arange(0, wimg - wpatch, wgap), np.array([wimg - wpatch])), axis=0):
                            mask = zps[patches_index] < 0.6
                            if mask == 0:
                                patches_index += 1
                                continue

                            weight_map[starty: starty + hpatch, startx: startx + wpatch] += self.wmap
                            prob_map[starty: starty + hpatch, startx: startx + wpatch] += \
                                ((1 - zps[patches_index]) * ls[patches_index]) * self.wmap

                            patches_index += 1
                    # print('fin')
                        #     x = data[:, :, starty:starty + hpatch, startx:startx + wpatch]
                        # # print(weight_map_folds.shape)
                        #     zero_percentage = (x == zero).sum(dim=(1, 2, 3)) / (x.shape[1] * x.shape[2] * x.shape[3])
                        #     mask = zero_percentage < 0.6
                        #
                        #     if mask.sum() == 0:
                        #         continue
                        #     # x = x[mask, :, :, :]
                        #
                        #     loss, fake = self.test_step(index, x, labels, path)
                        #
                        #     weight_map[starty: starty + hpatch, startx: startx + wpatch] += self.wmap
                        #     prob_map[starty: starty + hpatch, startx: startx + wpatch] += \
                        #         ((1 - zero_percentage) * loss) * self.wmap


                        # weight_map_folds[y, mask, :, :] += self.wmap
                        # prob_map_folds[y, mask, :, :] += \
                        #     ((1 - zero_percentage[mask]) * loss).view(loss.shape[0], 1, 1) * self.wmap

                    prob_map /= weight_map
                    loss = prob_map.mean().item()
                    # loss = prob_map.max().item()

                sample_filename = path[0]
                if torch.sum(torch.isnan(torch.tensor(loss)) + torch.isinf(torch.tensor(loss))) > 0:
                    print(sample_filename)
                else:
                    test_loss += loss

                all_losses[index] = loss
                all_prints[index] = "sample {} with loss {}".format(sample_filename, loss)
                all_maps[index] = prob_map

                # data.view(data.shape[-2], data.shape[-1]),
                # data[data == zero] = -1
                # picture = torch.cat([((data + 1)/2).view(data.shape[-2], data.shape[-1]), prob_map]).detach().cpu().numpy()
                # torch.save(picture, self.result_dir + '/recons_test_' + sample_filename.split('/')[-1] + '.obj')
                # torchvision.utils.save_image(picture, self.result_dir + '/recons_test_' + sample_filename.split('/')[-1] + '.png')
                # plot_img_to_file(picture, self.result_dir + '/recons_test_' + sample_filename.split('/')[-1] + '.png', vmin=0, vmax=1)

                if (index == 0) and save_images or save_all_images:
                    comparison = torch.cat([data,
                                            fake.view(-1, self.in_channels,
                                                      self.image_size, self.image_size)])
                    save_insar(comparison.view(-1, self.image_size, self.image_size).detach().cpu().numpy(), path[0],
                               self.result_dir + '/recons_test_' + sample_filename.split('/')[-1] + '.png',
                               ncols=int(np.sqrt(data.shape[0])))

                test_counter += 1
                # if test_counter % 10 == 0:
                #     print(f"{test_counter} / {len(self.test_loader)}")
                if test_counter == test_limit:
                    break


        test_loss /= len(self.test_loader.dataset)
        print('====> Test set loss: {:f}'.format(test_loss))
        return all_losses, all_prints, all_maps

    def get_loss(self,  dataset):
        with torch.no_grad():
            for index, (data, path, labels) in enumerate(dataset):
                data = data.to(self.device)
                zero = data.view(data.shape[-2], data.shape[-1])[10, 500].item()
                # zero = data.view(data.shape[-2], data.shape[-1])[500, 10].item()
                # zero = -1
                data[data == -1] = zero

                labels = labels.to(self.device)

                weight_map = torch.zeros(data.shape[2], data.shape[3]).to(self.device) + 0.00001
                prob_map = torch.zeros(data.shape[2], data.shape[3]).to(self.device)

                # x_folds = data.unfold(2, self.test_kernel_size, self.test_stride) \
                #     .unfold(3, self.test_kernel_size, self.test_stride)
                # weight_map_folds = weight_map.unfold(0, self.test_kernel_size, self.test_stride) \
                #     .unfold(1, self.test_kernel_size, self.test_stride)
                # prob_map_folds = prob_map.unfold(0, self.test_kernel_size, self.test_stride) \
                #     .unfold(1, self.test_kernel_size, self.test_stride)

                # for y in range(0, x_folds.shape[2]):
                #     x = x_folds[:, :, y, :, :, :].view(-1, self.in_channels, self.image_size, self.image_size)
                if self.no_heatmap:
                    data = data.unfold(2, self.image_size, self.image_size)\
                            .unfold(3, self.image_size, self.image_size)\
                            .reshape(-1, self.in_channels, self.image_size, self.image_size)

                    loss, fake = self.test_step(index, data, labels, path)
                    loss = loss.mean().item()
                else:
                    himg = data.shape[2]
                    wimg = data.shape[3]
                    hgap = wgap = 32

                    hpatch = self.image_size
                    wpatch = self.image_size

                    # batch_size = 169
                    batch_size = self.batch_size
                    patches = torch.zeros(batch_size, self.in_channels, self.image_size, self.image_size).to(self.device)
                    patches_index = 0

                    ls = torch.zeros(169).to(self.device)
                    zps = torch.zeros(169).to(self.device)
                    ind = 0

                    # print('now')
                    for starty in np.concatenate((np.arange(0, himg - hpatch, hgap), np.array([himg - hpatch])), axis=0):
                        for startx in np.concatenate((np.arange(0, wimg - wpatch, wgap), np.array([wimg - wpatch])), axis=0):
                            patches[patches_index, :, :, :] = data[:, :, starty:starty + hpatch, startx:startx + wpatch]
                            patches_index += 1

                            if patches_index == batch_size:
                                patches_index = 0
                                loss, fake = self.test_step(index, patches, labels, path)
                                zero_percentage = (patches == zero).sum(dim=(1, 2, 3)) / (
                                            patches.shape[1] * patches.shape[2] * patches.shape[3])
                                for i in range(len(loss)):
                                    ls[ind] = loss[i]
                                    zps[ind] = zero_percentage[i]
                                    ind += 1
                    if patches_index > 0:
                        loss, fake = self.test_step(index, patches, labels, path)
                        zero_percentage = (patches == zero).sum(dim=(1, 2, 3)) / (
                                patches.shape[1] * patches.shape[2] * patches.shape[3])
                        for i in range(patches_index):
                            ls[ind] = loss[i]
                            zps[ind] = zero_percentage[i]
                            ind += 1
                    # print(ind)

                    patches_index = 0
                    for starty in np.concatenate((np.arange(0, himg - hpatch, hgap), np.array([himg - hpatch])), axis=0):
                        for startx in np.concatenate((np.arange(0, wimg - wpatch, wgap), np.array([wimg - wpatch])), axis=0):
                            mask = zps[patches_index] < 0.6
                            if mask == 0:
                                patches_index += 1
                                continue

                            weight_map[starty: starty + hpatch, startx: startx + wpatch] += self.wmap
                            prob_map[starty: starty + hpatch, startx: startx + wpatch] += \
                                ((1 - zps[patches_index]) * ls[patches_index]) * self.wmap

                            patches_index += 1
                    # print('fin')
                        #     x = data[:, :, starty:starty + hpatch, startx:startx + wpatch]
                        # # print(weight_map_folds.shape)
                        #     zero_percentage = (x == zero).sum(dim=(1, 2, 3)) / (x.shape[1] * x.shape[2] * x.shape[3])
                        #     mask = zero_percentage < 0.6
                        #
                        #     if mask.sum() == 0:
                        #         continue
                        #     # x = x[mask, :, :, :]
                        #
                        #     loss, fake = self.test_step(index, x, labels, path)
                        #
                        #     weight_map[starty: starty + hpatch, startx: startx + wpatch] += self.wmap
                        #     prob_map[starty: starty + hpatch, startx: startx + wpatch] += \
                        #         ((1 - zero_percentage) * loss) * self.wmap


                        # weight_map_folds[y, mask, :, :] += self.wmap
                        # prob_map_folds[y, mask, :, :] += \
                        #     ((1 - zero_percentage[mask]) * loss).view(loss.shape[0], 1, 1) * self.wmap

                    prob_map /= weight_map
                    loss = prob_map.mean().item()

        return loss, prob_map

    def prepare_test(self):
        pass

    def test_step(self, index, data, labels, path) -> (torch.Tensor, torch.Tensor):
        pass

    def load_model(self, model_name):
        state_dict = torch.load(model_name)
        self.model.load_state_dict(state_dict)


def get_loss_function(loss_function):
    if 'LatentMSE' in loss_function:
        string_of_args = loss_function.split('(')[1].split(')')[0]
        args = dict(e.split('=') for e in string_of_args.split(', '))
        return LatentMSE(**args)


class LatentKDE(torch.nn.Module):
    def __init__(self, kde_i, kde_o):
        super().__init__()
        self.kde_i = kde_i
        self.kde_o = kde_o

    def forward(self, latent_i, latent_o):
        return -0.7 * torch.mean(torch.tensor(self.kde_i.score_samples(latent_i.cpu().numpy()))) \
            - 0.3 * torch.mean(torch.tensor(self.kde_i.score_samples(latent_o.cpu().numpy())))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"


class MSE(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.mean(torch.square(x - y), dim=0, keepdim=True)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"


class LatentMSE(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        if reduction == 'mean':
            self.red = torch.mean
        elif reduction == 'sum':
            self.red = torch.sum
        elif reduction == 'max':
            self.red = torch.max
        else:
            raise ValueError(f'Unknown reduction {reduction}')
        self.reduction = reduction
        self.mse = torch.nn.MSELoss()

    def forward(self, latent_i, latent_o):
        return self.red(self.mse(latent_i, latent_o))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(reduction={self.reduction})"
