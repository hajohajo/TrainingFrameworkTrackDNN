import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_curve, auc

binning_dict = {
    "trk_pt": np.linspace(0.0, 500.0, 101),
    "prediction": np.linspace(0.0, 1.0, 21),
    "trk_mva": np.linspace(0.0, 1.0, 21)
}

label_dict = {
    "trk_pt": "track p_{T}",
    "trk_mva": "BDT",
    "prediction": "DNN",
    "bkg": "fake",
    "sig": "signal"
}

color_dict = {"bkg": sns.xkcd_rgb["cerulean"], "sig": sns.xkcd_rgb["rouge"]}


def plot_xy(dataframe, x, y, class_index, result_dir, show_density=False, postfix=""):
    '''
    Plots y as a function of x using predefined binnings from the binning dict

    :param dataframe: Pandas dataframe with columns x, y and class_index columns
    :param x: name of the first variable to plot
    :param y: name of the second variable to plot
    :param class_index: name of the column that is 1 for signal and 0 for background
    :param result_dir: directory where to save the plots
    :param show_density: Whether to plot the density of signal and background under the plot
    :param name: label for the plot
    '''
    y_bin_width = binning_dict[y][1]-binning_dict[y][0]
    ymin = binning_dict[y][0]
    ymax = binning_dict[y][-1]+y_bin_width

    true_indices = (dataframe.loc[:, class_index] == 1)
    signal_x = dataframe[true_indices].loc[:, x]
    background_x = dataframe[~true_indices].loc[:, x]
    signal_y = dataframe[true_indices].loc[:, y]
    background_y = dataframe[~true_indices].loc[:, y]
    binning = binning_dict[x]
    bin_centers = binning[:-1]+(binning[1]-binning[0])/2.0

    sig_indices = np.digitize(signal_x, binning, right=True)
    sig_indices[sig_indices == 0] = 1
    sig_indices = sig_indices - 1
    bkg_indices = np.digitize(background_x, binning, right=True)
    bkg_indices[bkg_indices == 0] = 1
    bkg_indices = bkg_indices - 1

    sig_bin_means = -np.ones(len(binning)-1)
    sig_bin_std = np.zeros(len(binning)-1)
    bkg_bin_means = -np.ones(len(binning)-1)
    bkg_bin_std = np.zeros(len(binning)-1)

    for i in range(len(binning)-1):
        if (np.sum(sig_indices == i)) > 2:
            sig_mean, sig_std = norm.fit(signal_y[sig_indices == i])
            sig_bin_means[i] = sig_mean
            sig_bin_std[i] = sig_std

        if (np.sum(bkg_indices == i)) > 2:
            bkg_mean, bkg_std = norm.fit(background_y[bkg_indices == i])
            bkg_bin_means[i] = bkg_mean
            bkg_bin_std[i] = bkg_std


    sig_up = np.add(sig_bin_means, sig_bin_std)
    sig_up = np.clip(sig_up, 0.0, 1.0)
    sig_down = np.add(sig_bin_means, -sig_bin_std)
    sig_down = np.clip(sig_down, 0.0, 1.0)

    bkg_up = np.add(bkg_bin_means, bkg_bin_std)
    bkg_up = np.clip(bkg_up, 0.0, 1.0)
    bkg_down = np.add(bkg_bin_means, -bkg_bin_std)
    bkg_down = np.clip(bkg_down, 0.0, 1.0)

    non_zero_sig_bins = (sig_bin_means > -1)
    non_zero_bkg_bins = (bkg_bin_means > -1)

    if show_density:
        fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, sharex=True,
                                       gridspec_kw={'height_ratios': [2, 1]}, figsize=(8, 6))
        fig.suptitle(f"{y} mean and std. w.r.t {x}")

        for mean, up, down, centers, color in [(sig_bin_means[non_zero_sig_bins], sig_up[non_zero_sig_bins], sig_down[non_zero_sig_bins], bin_centers[non_zero_sig_bins], color_dict["sig"]),
                                               (bkg_bin_means[non_zero_bkg_bins], bkg_up[non_zero_bkg_bins], bkg_down[non_zero_bkg_bins], bin_centers[non_zero_bkg_bins], color_dict["bkg"])]:
            ax0.fill_between(centers, up, down, alpha=0.6, color=color)
            ax0.plot(centers, up, alpha=0.8, color=color)
            ax0.plot(centers, down, alpha=0.8, color=color)
            ax0.scatter(centers, mean, linewidths=1.0, color=color, edgecolors='k')
        ax0.set_xlim(binning[0], binning[-1])
        ax0.set_ylim(ymin, ymax)

        ax1.hist((background_x, signal_x), bins=binning, color=[color_dict["bkg"], color_dict["sig"]], stacked=True, edgecolor='k',
                 linewidth=1.0, alpha=0.7, label=[label_dict["bkg"], label_dict["sig"]])
        ax1.set_xlabel(x)
        ax1.set_ylabel("Number of tracks/bin")
        ax1.semilogy()
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        ax1.grid(False, axis='x')
        ax1.legend()

        plt.tight_layout(pad=2.2)
        fig.align_ylabels((ax0, ax1))
        plt.savefig(f"{result_dir}/{x}_vs_{y}_{postfix}.pdf")
        plt.clf()
        plt.close()

def plot_ROC_comparison(dataframe, prediction_1_label, prediction_2_label, truth_label, result_dir, postfix=""):
    """
    Produces ROC plot comparing two predictions

    :param dataframe: pandas dataframe containing at least prediction and truth labels
    :param prediction_1_label:
    :param prediction_2_label:
    :param truth_label:
    :param postfix:
    :return:
    """
    false_positive_rate_1, true_positive_rate_1, thresholds_1 = roc_curve(dataframe.loc[:, truth_label],
                                                                    dataframe.loc[:, prediction_1_label])
    false_positive_rate_2, true_positive_rate_2, thresholds_2 = roc_curve(dataframe.loc[:, truth_label],
                                                                    dataframe.loc[:, prediction_2_label])

    auc_1 = auc(false_positive_rate_1, true_positive_rate_1)
    auc_2 = auc(false_positive_rate_2, true_positive_rate_2)

    plt.plot(true_positive_rate_1, 1 - false_positive_rate_1, label="{} AUC = {:.3f}".format(label_dict[prediction_1_label], auc_1), color=sns.xkcd_rgb["emerald green"], linewidth=2)
    plt.plot(true_positive_rate_2, 1 - false_positive_rate_2, label="{} AUC = {:.3f}".format(label_dict[prediction_2_label], auc_2), color=sns.xkcd_rgb["dark lavender"], linewidth=2)

    plt.legend()
    plt.title("ROC curve")
    plt.ylabel("Fake rejection")
    plt.xlabel("True efficiency")
    plt.xlim(0.0, 1.05)
    plt.ylim(0.0, 1.05)
    plt.savefig(f"{result_dir}/ROC_{postfix}.pdf")
    plt.clf()
    plt.close()
