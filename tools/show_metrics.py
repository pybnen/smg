import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
import os.path

parser = argparse.ArgumentParser(description="Visualizes metrics.json of sacred run")
parser.add_argument("metrics_filepath", help="Path to metrics.json file")
args = parser.parse_args()


def moving_average(a: np.array, n: int=3) -> np.array:
    """Create moving average of sequence a over with window size n
    a: np.array, shape (x, )
    n: int
    return np.array containing moving average, shape (x - (n-1), )"""

    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def vis_metric(file_path, only_avg=False):
    with open(file_path, "r") as f:
        metrics_content = json.load(f)
    print(metrics_content.keys())
    if only_avg:
        n_cols = 1
        get_avg_idx = lambda i: i + 1
    else:
        n_cols = 2
        get_normal_idx = lambda i: 2 * i + 1
        get_avg_idx = lambda i: 2 * i + 2

    metric_names = ['train.losses/r_loss',
                    'train.losses/kl_loss',
                    'eval.losses/r_loss',
                    'eval.losses/kl_loss']
    n_metrics = len(metric_names)
    window_size = 50

    fig = plt.figure(figsize=(7.5 * n_cols, 6 * n_metrics))
    for i, name in enumerate(metric_names):
        if name not in metrics_content:
            continue

        metric = metrics_content[name]

        values = np.array(metric['values'])
        steps = np.array(metric['steps'])

        if not only_avg:
            # plot normal values
            ax = fig.add_subplot(n_metrics, n_cols, get_normal_idx(i))
            ax.plot(steps, values)

            ax.grid(True, axis='x')
            plt.xlabel('Update steps')
            plt.ylabel('Loss')
            plt.title(name)

        # plot moving average
        ax = fig.add_subplot(n_metrics, n_cols, get_avg_idx(i))
        ax.plot(steps[window_size - 1:], moving_average(values, window_size))

        ax.grid(True, axis='x')
        plt.xlabel('Update steps')
        plt.ylabel('Loss')
        plt.title(" {} moving avg (window size {})".format(name, window_size))
    #fig.tight_layout(h_pad=5.)
    plt.subplots_adjust(hspace=0.80)
    plt.show()


def main():
    if not os.path.isfile(args.metrics_filepath):
        print("Given path is not a file: '{}'".format(args.metrics_filepath))
        return
    vis_metric(args.metrics_filepath, only_avg=False)


if __name__ == "__main__":
    main()

