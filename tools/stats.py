import os

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
import numpy.ma as ma
from torch import tensor

from tools.auroc import AUROC
from tools.optimal_f1 import OptimalF1


def _normalize(a):
    return (a-np.nanmin(a))/(np.nanmax(a) - np.nanmin(a))


def _reverse_normalize(x, normal, abnormal):
    all_elem = np.concatenate((normal, abnormal))
    return x * (np.nanmax(all_elem) - np.nanmin(all_elem)) + np.nanmin(all_elem)


def _get_preds_and_target(normal, abnormal) -> (torch.Tensor, torch.Tensor):
    # positive class is outlier, so abnormal = 1
    # we don't classify the preds, otherwise the algorithm won't work (it will output a graph with 1 point)
    # instead we normalize the losses
    all_elem = np.concatenate((normal, abnormal))
    preds = _normalize(all_elem)
    target = np.concatenate((np.zeros(normal.size), np.ones(abnormal.size)))

    return tensor(preds), tensor(target)


def compute_threshold(normal, abnormal):
    preds, target = _get_preds_and_target(normal, abnormal)

    threshold = OptimalF1()
    threshold.update(preds, target)
    threshold.compute()

    return threshold.threshold.item()


def compute_real_threshold(normal, abnormal):
    preds, target = _get_preds_and_target(normal, abnormal)

    threshold = OptimalF1()
    threshold.update(preds, target)
    threshold.compute()

    th = threshold.threshold.item()
    return _reverse_normalize(th, normal, abnormal)


def plot_bins(normal, abnormal, plot_name):
    preds, _ = _get_preds_and_target(normal, abnormal)
    _range = (torch.min(preds).item(), torch.max(preds).item())
    counts_normal, bins_normal = np.histogram(preds[:normal.size], bins=20, range=_range)
    counts_abnormal, bins_abnormal = np.histogram(preds[normal.size:], bins=20, range=_range)

    indexes = np.zeros(len(bins_normal) - 1)

    count = 0
    y = bins_normal[0]
    for x in bins_normal[1:]:
        indexes[count] = (x + y) / 2
        y = x
        count += 1

    counts_normal = np.array(counts_normal) / np.array(counts_normal).sum()
    counts_abnormal = np.array(counts_abnormal) / np.array(counts_abnormal).sum()

    mask1 = ma.where(counts_normal >= counts_abnormal)
    mask2 = ma.where(counts_abnormal > counts_normal)

    plt.bar(indexes[mask1], counts_normal[mask1], label="normal", color='b', width=0.02)
    plt.bar(indexes, counts_abnormal, label="abnormal", color='r', width=0.02)
    plt.bar(indexes[mask2], counts_normal[mask2], color='b', width=0.02)

    plt.axvline(x=compute_threshold(normal, abnormal), label="threshold", c='y')

    plt.xlabel("score")
    plt.ylabel("normalized frequency")
    plt.legend()
    plt.savefig(plot_name, bbox_inches='tight')
    plt.close()


def plot_auroc(normal, abnormal, plot_name, print_results=False, logger=None, stats_obj=None):
    preds, target = _get_preds_and_target(normal, abnormal)

    auroc = AUROC()
    auroc.update(preds, target)
    fig, title = auroc.generate_figure(stats_obj)

    _, _, auc = auroc.get_stats()

    if print_results:
        print(f'AUC: {auc:0.2f}')

    if logger is not None:
        logger['AUROC'] = int(auc * 100)

    fig.savefig(plot_name, bbox_inches='tight')
    plt.close(fig)


def get_f1(normal, abnormal, print_results=False, logger=None, stats_obj=None):
    preds, target = _get_preds_and_target(normal, abnormal)

    f1 = OptimalF1()
    f1.update(preds, target)
    result = f1.compute().item()

    if stats_obj is not None:
        stats_obj['f1'] = result

    if print_results:
        print(f"F1 score: {result}")

    if logger is not None:
        logger['F1_score'] = int(result * 100)

    return result


def get_accuracy(normal, abnormal, print_results=False, logger=None):
    threshold = compute_threshold(normal, abnormal)
    preds, _ = _get_preds_and_target(normal, abnormal)
    preds = np.array(preds)

    normal_correct = normal.size - (preds[:normal.size] > threshold).sum()
    normal_acc = normal_correct / normal.size

    abnormal_correct = (preds[normal.size:] > threshold).sum()
    abnormal_acc = abnormal_correct / abnormal.size

    overall_accuracy = (normal_correct + abnormal_correct) / preds.size
    # print(preds[normal.size:] > threshold)
    if print_results:
        print(f"Normal acc: {normal_acc}, {normal_correct}/{normal.size}")
        print(f"Abnormal acc: {abnormal_acc}, {abnormal_correct}/{abnormal.size}")
        print(f"Overall accuracy: {overall_accuracy}")

    if logger is not None:
        logger['true_negatives'] = normal_correct
        logger['nr_of_negatives'] = normal.size
        logger['true_positives'] = abnormal_correct
        logger['nr_of_positives'] = abnormal.size
        logger['accuracy'] = overall_accuracy

    return overall_accuracy, normal_acc, abnormal_acc


class MultiPlotAUROC:
    def __init__(self, plot_name):
        # TODO add a line through the middle
        fig, axis = plt.subplots()
        self.fig = fig
        self.axis = axis
        self.plot_name = plot_name

        self.axis.plot(
            [0, 1],
            [0, 1],
            lw=2,
            linestyle="--",
            figure=self.fig,
        )

    def plot(self, normal, abnormal, plot_name):
        preds, target = _get_preds_and_target(normal, abnormal)
        auroc = AUROC()
        # update appeds the new results to the previous ones, so we need to create a new auroc object every time
        auroc.update(preds, target)
        x_vals, y_vals, auc = auroc.get_stats()

        self.axis.plot(
            x_vals,
            y_vals,
            figure=self.fig,
            lw=2,
            label=f"{plot_name}: {auc.detach().cpu():0.2f}",
        )

    def plot_precalculated(self, x_vals, y_vals, plot_name, color=None, linestyle=None,):
        if color == None:
            self.axis.plot(
                x_vals,
                y_vals,
                figure=self.fig,
                lw=2,
                label=f"{plot_name}",
            )
        else:
            self.axis.plot(
                x_vals,
                y_vals,
                figure=self.fig,
                lw=2,
                label=f"{plot_name}",
                linestyle=linestyle,
                c=color,
            )

    def close_plot(self):
        self.axis.set_xlim((0.0, 1.0))
        self.axis.set_ylim((0.0, 1.0))
        self.axis.set_xlabel("False Positive Rate")
        self.axis.set_ylabel("True Positive Rate")
        self.axis.legend(loc="lower right")
        self.axis.set_title("ROC")

        self.fig.savefig(self.plot_name, bbox_inches='tight')
        plt.close(self.fig)


class StatsSaver:
    def __init__(self, path_to_csv='res/stats.csv'):
        self.path_to_csv = path_to_csv
        if os.path.exists(self.path_to_csv):
            self.stats = pd.read_csv(self.path_to_csv, index_col=0)
        else:
            self.stats = pd.DataFrame()

    def get_column(self, nr_of_column):
        return self.stats.iloc[[nr_of_column]].to_dict(orient='records')[0]

    def get_model_logs(self, model_name):
        return self.stats.loc[self.stats['model_name'] == model_name].to_dict(orient='records')[0]

    def add_column(self, logger):
        new_row = pd.Series(logger)
        self.stats = pd.concat([self.stats, new_row.to_frame().T], ignore_index=True)
        # self.stats = pd.DataFrame(logger, index=[0])

    def close_stats(self):
        self.stats.to_csv(self.path_to_csv)


def _calculate_accuracy_with_threshold(positives, losses, prints, threshold):
    tp = 0
    allp = (losses > threshold).sum()
    for i in prints[losses > threshold]:
        for p in positives:
            if p in i:
                tp += 1
                break

    fn = 0
    alln = (losses < threshold).sum()
    for i in prints[losses < threshold]:
        for p in positives:
            if p in i:
                fn += 1
                break

    print(f'true positives: {tp}, false_positives: {allp - tp}, true negatives: {alln - fn}, false_negatives: {fn}')


def custom_auroc_no_heatmap(dataset, subfolder, stats_obj):
    positives = positives_dict[dataset][subfolder]
    positives = [str(p).split('.')[0] for p in positives]
    res = stats_obj

    high_percentage = 0.99
    percentage = 0.95

    normal_losses = res['normal_losses']
    abnormal_losses = res['abnormal_losses']
    train_losses = res['train_losses']

    normal_prints = res['normal_prints']
    abnormal_prints = res['abnormal_prints']

    if dataset in ['taal', 'agung']:
        normal_losses = np.concatenate((normal_losses, abnormal_losses))
        normal_prints = np.concatenate((normal_prints, abnormal_prints))

    normal_prints = [str(p).split('/')[-1].split('.')[0].split('geo')[0] for p in normal_prints]
    normal_prints = np.array(normal_prints)

    sorted_train_losses = np.sort(train_losses)
    threshold = sorted_train_losses[int(percentage * len(sorted_train_losses))]
    _calculate_accuracy_with_threshold(positives, normal_losses, normal_prints, threshold)
    threshold = sorted_train_losses[int(high_percentage * len(sorted_train_losses))]
    _calculate_accuracy_with_threshold(positives, normal_losses, normal_prints, threshold)

    ns = []
    abs = []
    for i in range(len(normal_prints)):
        normal = True
        for p in positives:
            if p in normal_prints[i]:
                normal = False
                break
        if normal:
            ns.append(normal_losses[i])
        else:
            abs.append(normal_losses[i])

    ns = np.array(ns)
    abs = np.array(abs)

    preds, target = _get_preds_and_target(ns, abs)
    auroc = AUROC()
    auroc.update(preds, target)
    _, _, auc = auroc.get_stats()
    print(f'AUC: {auc:0.3f}')

    get_accuracy(ns, abs, print_results=True)
    get_f1(ns, abs, print_results=True)

    print('Real threshold:', compute_real_threshold(ns, abs))


def custom_auroc(dataset, subfolder, stats_obj, mask=None, use_max=False):
    if mask is None:
        mask = torch.zeros((512,512))
    positives = positives_dict[dataset][subfolder]
    positives = [str(p).split('.')[0] for p in positives]
    res = stats_obj

    high_percentage = 0.99
    percentage = 0.95

    normal_maps = res['normal_maps']
    normal_maps[:, mask == 1] = 0
    if use_max:
        normal_losses = normal_maps.max(axis=(1, 2))
    else:
        normal_losses = normal_maps.mean(axis=(1, 2))

    abnormal_maps = res['abnormal_maps']
    abnormal_maps[:, mask == 1] = 0
    if use_max:
        abnormal_losses = abnormal_maps.max(axis=(1, 2))
    else:
        abnormal_losses = abnormal_maps.mean(axis=(1, 2))

    normal_prints = res['normal_prints']
    abnormal_prints = res['abnormal_prints']
    train_losses = res['train_losses']

    if 'train_maps' in res.keys():
        train_maps = res['train_maps']
        train_maps[:, mask == 1] = 0
        if use_max:
            train_losses = train_maps.max(axis=(1, 2))
        else:
            train_losses = train_maps.mean(axis=(1, 2))
        train_losses = np.sort(train_losses)

    if dataset in ['taal', 'agung']:
        normal_losses = np.concatenate((normal_losses, abnormal_losses))
        normal_prints = np.concatenate((normal_prints, abnormal_prints))

    normal_prints = [str(p).split('/')[-1].split('.')[0].split('geo')[0] for p in normal_prints]
    normal_prints = np.array(normal_prints)

    sorted_train_losses = np.sort(train_losses)
    threshold = sorted_train_losses[int(percentage * len(sorted_train_losses))]
    _calculate_accuracy_with_threshold(positives, normal_losses, normal_prints, threshold)
    threshold = sorted_train_losses[int(high_percentage * len(sorted_train_losses))]
    _calculate_accuracy_with_threshold(positives, normal_losses, normal_prints, threshold)

    ns = []
    abs = []
    for i in range(len(normal_prints)):
        normal = True
        for p in positives:
            if p in normal_prints[i]:
                normal = False
                break
        if normal:
            ns.append(normal_losses[i])
        else:
            abs.append(normal_losses[i])

    ns = np.array(ns)
    abs = np.array(abs)

    preds, target = _get_preds_and_target(ns, abs)
    auroc = AUROC()
    auroc.update(preds, target)
    _, _, auc = auroc.get_stats()
    print(f'AUC: {auc:0.3f}')

    get_accuracy(ns, abs, print_results=True)
    get_f1(ns, abs, print_results=True)

    print('Real threshold:', compute_real_threshold(ns, abs))



positives_lamongan_3D = [
    '20161229_20170323',
    '20180529_20180704',
    '20191027_20191120',
    '20180704_20180716',
    '20191015_20191120',
    '20161229_20170215',
    '20161111_20161229',
    '20191108_20191120',
    '20161229_20170122',
    '20161205_20161229',
    '20161229_20170311',

    '20160924_20161229',
    '20161018_20161229',
    '20181008_20181113',
    '20181020_20181113',
    '20181110_20181113',
    '20190921_20191120',
    '20191015_20191202',
    '20191027_20191202',
    '20191108_20191202',
]

positives_lamongan_105D = [
    '20190730_20190823',
    '20191010_20191127',
    '20191221_20200114',
    '20190730_20190904',
    '20191010_20191115',
    '20190730_20190811',
    '20191022_20191115',
    '20190718_20190730',
    '20190612_20190730',

    '20161212_20170210',
    '20170105_20170210',
    '20170129_20170222',
    '20171020_20171101',
    '20191022_20191127',
    '20200712_20200724',
    '20200724_20200805',
    '20210426_20210508',
    '20210508_20210520',
    '20220409_20220726',
    '20220515_20220726',
    '20220515_20220807',
    '20220702_20220706',
    '20220702_20220807',
    '20200714_20220726',
    '20220714_20220807',
    '20220714_20220831',
    '20220726_20220807',
    '20220726_20220819',
    '20220726_20220831',
    '20220726_20220921',
    '20220807_20220819',
    '20220807_20220831',
    '20220807_20220921',
    '20220807_20220924',
    '20220819_20220831',
    '20220819_20220912',
    '20220831_20220912',
    '20220831_20220924',
    '20220831_20221006',
    '20220831_20221018',
    '20220912_20220924',
    '20220912_20221006',
    '20220912_20221018',
    '20220912_20221030',
    '20221123_20230110',
    '20221205_20230110',
    '20221205_20230122',
    '20221217_20230110',
]

positives_lawu_3D = [
]

positives_lawu_127A = [
]

positives_casiri_54D = [
    '20180918_20181012',
    '20180918_20181024',
    '20180918_20181105',
    '20181012_20181024',
    '20181012_20181105',
    '20191012_20181111',
    '20181024_20181105',
    '20181024_20181111',
    '20181024_20181117',
    '20181105_20181111',
    '20181105_20181117',
    '20181105_20181123',
    '20181111_20181117',
    '20181111_20181129',
    '20181117_20181123',
    '20181117_20181129',
    '20181117_20181205',
    '20181123_20181129',
    '20181129_20181205',
    '20190404_20190410',
    '20190416_20190504',
    '20190422_20190504',
    '20190422_20190510',
    '20190428_20190504',
    '20190428_20190516',
    '20190504_20190510',
    '20190516_20190528',
    '20190516_20190609',
    '20190528_20190609',
    '20190609_20190615',
    '20190609_20190709',
    '20190709_20190715',
    '20190709_20190727',
    '20190715_20190721',
    '20190715_20190727',
    '20190715_20190808',
    '20190721_20190727',
    '20190721_20190808',
    '20190727_20190802',
    '20190727_20190808',
    '20190727_20190814',
    '20190802_20190808',
    '20190802_20190820',
    '20190808_20190814',
    '20190808_20190820',
    '20190808_20190826',
    '20190814_20190820',
    '20190814_20190901',
    '20190820_20190826',
    '20190820_20190901',
    '20190820_20190907',
    '20190826_20190901',
    '20190826_20190913',
    '20190901_20190907',
    '20190901_20190913',
    '20190901_20190919',
    '20190907_20190913',
    '20190907_20190925',
    '20190913_20190919',
    '20190913_20190925',
    '20190919_20190925',
    '20190925_20191007',
    '20191007_20191019',
    '20191019_20191031',
    '20191019_20191106',
    '20191025_20191031',
    '20191025_20191112',
    '20191031_20191106',
    '20191031_20191112',
    '20191031_20191118',
    '20191031_20191130',
    '20191106_20191112',
    '20191106_20191118',
    '20191106_20191206',
    '20191112_20191118',
    '20191112_20191130',
    '20191112_20191212',
    '20191130_20191206',
    '20191130_20191218',
    '20191206_20191212',
    '20191206_20191224',
    '20191212_20191218',
    '20191212_20191230',
    '20191218_20191224',
    '20191218_20200105',
    '20191218_20200117',
    '20191224_20191230',
    '20191224_20200123',
    '20191230_20200105',
    '20191230_20200117',
    '20191230_20200129',
    '20200105_20200123',
    '20200105_20200204',
    '20200117_20200123',
    '20200117_20200204',
    '20200123_20200129',
    '20200123_20200204',
    '20200123_20200210',
    '20200129_20200204',
    '20200129_20200216',
    '20200204_20200210',
    '20200204_20200222',
    '20200210_20200216',
    '20200210_20200228',
    '20200216_20200222',
    '20200216_20200305',
    '20200222_20200228',
    '20200222_20200311',
    '20200228_20200305',
    '20200228_20200311',
    '20200305_20200311',
    '20200305_20200901',
    '20200311_20200907',
    '20200317_20200901',
    '20200317_20200913',
    '20200317_20200919',
    '20200317_20200925',
    '20200323_20200907',
    '20200329_20200901',
    '20200404_20200410',
    '20200404_20200422',
    '20200410_20200416',
    '20200410_20200428',
    '20200416_20200422',
    '20200416_20200504',
    '20200416_20200510',
    '20200422_20200428',
    '20200422_20200510',
    '20200428_20200504',
    '20200428_20200510',
    '20200428_20200516',
    '20200428_20200528',
    '20200504_20200510',
    '20200510_20200516',
    '20200510_20200528',
    '20200510_20200609',
    '20200603_20200919',
    '20200609_20200727',
    '20200609_20200907',
    '20200609_20200913',
    '20200609_20200919',
    '20200609_20200925',
    '20200615_20200925',
    '20200621_20200727',
    '20200621_20200808',
    '20200621_20200907',
    '20200621_20200919',
    '20200627_20200802',
    '20200627_20200814',
    '20200627_20200820',
    '20200627_20200907',
    '20200627_20200919',
    '20200627_20200925',
    '20200627_20210306',
    '20200627_20210318',
    '20200627_20210610',
    '20200703_20200727',
    '20200703_20200808',
    '20200703_20200814',
    '20200703_20200820',
    '20200703_20200919',
    '20200709_20200727',
    '20200709_20200814',
    '20200709_20200820',
    '20200709_20200919',
    '20200709_20200925',
    '20200715_20200727',
    '20200715_20200802',
    '20200715_20200808',
    '20200715_20200820',
    '20200721_20200727',
    '20200721_20200802',
    '20200721_20200808',
    '20200721_20200826',
]

positives_agung = [
    '20180709_20180715',
    '20200526_20200607',
    '20180709_20180727',
    '20181007_20181019',
    '20180531_20180612',
    '20180419_20180501',
    '20190219_20190225',
    '20180916_20181004',
    '20181206_20181218',
    '20200106_20200118',
    '20180425_20180501',
    '20200121_20200127',
    '20170605_20170723',
    '20200526_20200619',
    '20200115_20200127',
    '20200226_20200303',
    '20180925_20181007',
    '20180802_20180820',
    '20200526_20200601',
    '20200619_20200725',
    '20181019_20181031',
    '20200115_20200121',
    '20170415_20170720',
    '20180703_20180715',
    '20191119_20191201',
    '20200514_20200526',
    '20180410_20180609',
    '20191225_20200130',
    '20200607_20200701',
    '20180215_20180221',
    '20180823_20180829',
    '20200426_20200526',
    '20190225_20190303',
    '20181115_20181121',
    '20180612_20180624',
    '20190126_20190219',
    '20190228_20190306',
    '20191201_20191225',
    '20180606_20180612',
    '20191014_20191026',
    '20190201_20190219',
    '20180829_20180910',
    '20200408_20200426',
    '20181121_20181215',
    '20200619_20200625',
    '20200420_20200426',
    '20200426_20200502',
    '20180320_20180407',
    '20200208_20200226',
    '20200520_20200613',
    '20190207_20190219',
    '20200508_20200607',
    '20180407_20180425',
    '20200514_20200613',
    '20180814_20180820',
    '20181206_20181230',
    '20200414_20200426',
    '20200208_20200220',
    '20180401_20180407',
    '20191014_20191201',
    '20200402_20200426',
    '20200420_20200502',
    '20190219_20190303',
    '20181004_20181022',
    '20200607_20200619',
    '20180425_20180507',
    '20180901_20180925',
    '20190204_20190306',
    '20200220_20200303',
    '20180724_20180805',
    '20181203_20181215',
    '20200426_20200508',
    '20200514_20200601',
    '20180401_20180413',
    '20200526_20200613',
    '20191026_20191213',
    '20190213_20190219',
    '20180913_20181007',
    '20200502_20200601',
    '20200520_20200607',
    '20200520_20200601',
    '20180116_20180122',
    '20191023_20191029',
    '20190219_20190309',
    '20180215_20180227',
    '20170629_20170723',
    '20191213_20200130',
    '20180504_20180528',
    '20200508_20200601',
    '20200514_20200607',
    '20200613_20200619',
    '20191119_20191225',
    '20200426_20200520',
    '20200607_20200613',
    '20200214_20200303',
    '20190520_20190613',
    '20190514_20190601',
    '20190713_20190719',
    '20170723_20170804',
    '20190520_20190607',
    '20190426_20190520',
    '20190520_20190526',
    '20190327_20190402',
    '20190420_20190514',
    '20170711_20170723',
    '20190601_20190613',
    '20190526_20190613',
    '20190601_20190625',

    '20171015_20171027',
    '20190803_20190815',
    '20190929_20191005',
    '20190502_20190526',
    '20190911_20190917',
    '20190806_20190812',
    '20190502_20190520',
    '20190719_20190806',
    '20170813_20170918',
    '20190514_20190526',
    '20190508_20190601',
    '20190908_20191014',
    '20190526_20190619',
    '20190911_20190923',
    '20190713_20190725',
    '20190607_20190619',
    '20171027_20171108',
    '20190526_20190607',
    '20170801_20170906',
    '20190508_20190526',
    '20170921_20171027',
    '20190601_20190607',
    '20190508_20190520',
    '20190812_20190830',
    '20170723_20170828',
    '20190321_20190402',
    '20190619_20190625',
    '20170723_20170816',
    '20190613_20190619',
    '20190514_20190520',
    '20190625_20190713',
]

positives_taal = ['20191218_20200117', '20200115_20200127',
                  '20191206_20200117', '20200111_20200204',
                  '20200111_20200123', '20191216_20200115',
                  '20191230_20200117', '20200111_20200129',
                  '20200117_20200129', '20200117_20200123',
                  '20200111_20200117', '20200115_20200202',
                  '20191228_20200115', '20200117_20200210',
                  '20200109_20200115', '20200117_20200204',
                  '20191204_20200115']

positives_dict = {
    'lamongan': {
        'gacos_3D-inp2': positives_lamongan_3D,
        'gacos_3D_remade-inp': positives_lamongan_3D,
        'gacos_105D-inp': positives_lamongan_105D,
        'gacos_105D_remade-inp': positives_lamongan_105D,
    },
    'agung': {
        'gacos_png_remade': positives_agung,
        'unw_png_remade': positives_agung,
        'unw_png_remade2': positives_agung,
    },
    'taal': {
        'unw_png_norm': positives_taal,
        'unw_png': positives_taal,
        'wrap': positives_taal,
        'wrap_png': positives_taal,
        'unw_remade_water': positives_taal,
    },
    'casiri': {
        'gacos_54D-inp': positives_casiri_54D,
        'gacos_54D_remade-inp': positives_casiri_54D,
    },
    'lawu': {
        'gacos_127A-inp': positives_lawu_127A,
        'gacos_3D-inp2': positives_lawu_3D,
        'gacos_3D_remade-inp': positives_lawu_3D,
        'gacos_127A_remade-inp': positives_lawu_127A,
    },
}
