# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2007-2021 The scikit-learn developers.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

import math
from io import BytesIO
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from kumo_core.utils.names import get_table_name
from kumoapi.typing import Stype

from io_utils import fsutil, logging
from kumo.config.graph import TableMetadataConfig

logger = logging.getLogger(__name__)


def visualize_positional_encoding(pe: torch.Tensor, filepath: str):
    assert pe.ndim == 2
    import matplotlib.pyplot as plt

    # `matshow` always creates its own figure (ignoring any current
    # figure), so close the one it actually drew on rather than a
    # separately-created, never-drawn-on `plt.figure()`.
    cax = plt.matshow(pe.numpy())
    fig = cax.figure
    fig.colorbar(cax)
    fig.savefig(filepath)
    plt.close(fig)


def visualize_scalar_distribution(data, filepath: str,
                                  xlabel_name: str = "label",
                                  xlim: Optional[Tuple] = None):
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    matplotlib.use("Agg")

    if isinstance(data, torch.Tensor):
        data = data.numpy()

    fig = plt.figure()
    plt_ax = fig.add_subplot(111)

    if xlim is not None:
        plt_ax.set_xlim(xlim)

    sns.histplot(data, kde=True, ax=plt_ax)
    plt.xlabel(xlabel_name)
    plt.ylabel("Count")
    try:
        buffer = BytesIO()
        fig.savefig(buffer, format='png')
        fsutil.write_binary(filepath, buffer.getvalue())
    except FileNotFoundError:
        logger.warning(
            "Scalar distribution not saved because {} does not exist.".format(
                filepath))
    plt.close(fig)


def bar_plot(data, filepath, xlabel_name, ylabel_name, hue=None, color=None,
             xlabel_rotate=0, order_x=False, dpi: Optional[int] = None):
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    matplotlib.use("Agg")

    fig = plt.figure()
    plt_ax = fig.add_subplot(111)

    if not xlabel_rotate == 0:
        plt.xticks(rotation=xlabel_rotate, ha='right')

    if order_x:
        data.sort_values(by=[xlabel_name], ascending=True, inplace=True)

    sns.barplot(x=xlabel_name, y=ylabel_name, data=data, ax=plt_ax, hue=hue,
                color=color, dodge=False)
    fig.tight_layout()

    try:
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=dpi)
        fsutil.write_binary(filepath, buffer.getvalue())
    except FileNotFoundError:
        logger.warning(
            "Bar plot not saved because {} does not exist.".format(filepath))
    plt.close(fig)


def scatter_plot(data, filepath, xlabel_name, ylabel_name, xlabel_rotate=0):
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    matplotlib.use("Agg")

    fig = plt.figure()
    plt_ax = fig.add_subplot(111)

    if not xlabel_rotate == 0:
        plt.xticks(rotation=xlabel_rotate, ha='right')
    sns.scatterplot(x=xlabel_name, y=ylabel_name, data=data, ax=plt_ax)
    fig.tight_layout()

    try:
        buffer = BytesIO()
        fig.savefig(buffer, format='png')
        fsutil.write_binary(filepath, buffer.getvalue())
    except FileNotFoundError:
        logger.warning(
            "Scatter plot not saved because {} does not exist.".format(
                filepath))
    plt.close(fig)


def box_plot(
    data: pd.DataFrame,
    filepath: str,
    xlabel_name: str,
    ylabel_name: str,
    xlabel_rotate: int = 0,
    outliers: bool = False,
    whis: float = 1.5,
):
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    matplotlib.use("Agg")

    fig = plt.figure()
    plt_ax = fig.add_subplot(111)

    if not xlabel_rotate == 0:
        plt.xticks(rotation=xlabel_rotate, ha='right')
    # The argument whis specifies the proportion of the inter-quartile range
    # past the low and high quartiles to extend the plot whiskers.
    # Points outside this range will be identified as outliers.
    sns.boxplot(x=xlabel_name, y=ylabel_name, data=data, ax=plt_ax,
                showfliers=outliers, whis=whis)
    fig.tight_layout()
    # fig1 = plt.gcf()

    try:
        buffer = BytesIO()
        fig.savefig(buffer, format='png')
        fsutil.write_binary(filepath, buffer.getvalue())
    except FileNotFoundError:
        logger.warning(
            "Box plot not saved because {} does not exist.".format(filepath))
    plt.close(fig)


def binary_roc_curve_plot(y_true: np.array, y_score: np.array, filepath: str,
                          pos_label: int = 1):
    """
    Plot the ROC curve for binary classification. Adapted from scikit-learn's
    ``examples/model_selection/plot_roc.py`` at commit
    02b41de8fca096d92aea6a5336fce372b842c5c7.

    Args:
        y_true (np.array): Ground truth labels.
        y_score (np.array): Scores or probabilities for the positive class.
        filepath (str): Path to save the figure.
        pos_label (int, optional): Label for the positive class.
            Default is 1.
    """
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use("Agg")
    from sklearn.metrics import roc_auc_score, roc_curve

    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=pos_label)
    roc_auc = roc_auc_score(y_true, y_score)
    fig = plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2,
             label=f"ROC curve (area={roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    fig.tight_layout()

    try:
        buffer = BytesIO()
        fig.savefig(buffer, format='png')
        fsutil.write_binary(filepath, buffer.getvalue())
    except FileNotFoundError:
        logger.warning(
            "ROC curve not saved because {} does not exist.".format(filepath))
    plt.close(fig)


def binary_pr_curve_plot(y_true: np.array, y_score: np.array, filepath: str,
                         pos_label: int = 1, positive_rate: float = None):
    """
    Plot the PR curve for binary classification. Adapted from scikit-learn's
    ``examples/model_selection/plot_roc.py`` at commit
    02b41de8fca096d92aea6a5336fce372b842c5c7.

    Args:
        y_true (np.array): Ground truth labels.
        y_score (np.array): Probabilities for the positive class.
        filepath (str): Path to save the figure.
        pos_label (int, optional): Label for the positive class.
            Default is 1.
        positive_rate (float, optional): The ratio of the positive
            samples.
    """
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use("Agg")
    from sklearn.metrics import auc, precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(
        y_true, y_score, pos_label=pos_label)
    pr_auc = auc(recall, precision)
    fig = plt.figure()
    plt.plot(recall, precision, color="darkorange", lw=2,
             label=f"PR curve (area={pr_auc:.2f})")
    if positive_rate is not None:
        plt.plot([0, 1], [positive_rate, positive_rate], color="navy", lw=2,
                 linestyle="--", label="Random")
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision Recall Curve (PRC)")
    plt.legend(loc="lower right")
    fig.tight_layout()

    try:
        buffer = BytesIO()
        fig.savefig(buffer, format='png')
        fsutil.write_binary(filepath, buffer.getvalue())
    except FileNotFoundError:
        logger.warning(
            "PR curve not saved because {} does not exist.".format(filepath))
    plt.close(fig)


def binary_confusion_matrix_plot(y_true: np.array, y_pred: np.array,
                                 filepath: str, pos_label: int = 1):
    """
    Plot the confusion matrix for binary classification in heatmap. Modified
    from https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea

    Args:
        y_true (np.array): Ground truth labels.
        y_pred (np.array): Predicted targets.
        filepath (str): Path to save the figure.
        pos_label (int, optional): Label for the positive class.
            Default is 1.
    """
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    matplotlib.use("Agg")

    matrix = confusion_matrix(y_true, y_pred)
    num = matrix.sum()
    names = [
        'True Negative', 'False Positive', 'False Negative', 'True Positive'
    ]
    counts = [f"{value}" for value in matrix.flatten()]
    percentages = [f"{value:.2f}" for value in matrix.flatten() / num]

    num_positive = (y_true == pos_label).sum()
    num_negative = (y_true != pos_label).sum()
    num_truth = np.array(
        [num_negative, num_negative, num_positive, num_positive])
    labels = [
        f"{n}\n{c} / {t}\n{p}"
        for n, c, t, p in zip(names, counts, num_truth, percentages)
    ]

    labels = np.asarray(labels).reshape(2, 2)
    fig = plt.figure()
    sns.heatmap(matrix, annot=labels, fmt='', cmap='Blues')
    fig.tight_layout()

    try:
        buffer = BytesIO()
        fig.savefig(buffer, format='png')
        fsutil.write_binary(filepath, buffer.getvalue())
    except FileNotFoundError:
        logger.warning(
            "Confusion matrix not saved because {} does not exist.".format(
                filepath))
    plt.close(fig)


def visualize_graph(meta_table_configs: List[TableMetadataConfig], name: str,
                    path: Union[str, BytesIO,
                                None] = None, show_cols: bool = True):
    def has_graphviz() -> bool:
        try:
            import graphviz
        except ImportError:
            return False

        try:
            graphviz.Digraph().pipe()
        except graphviz.backend.ExecutableNotFound:
            return False

        return True

    def get_stype(stype: Stype):
        if stype and isinstance(stype, Stype):
            return stype.title_case()
        else:
            return '???'

    def left_align(list_of_text: List[str]):
        return '\\l'.join(list_of_text) + '\\l'

    if not has_graphviz():
        raise ModuleNotFoundError('graphviz is required for visualization.')
    else:
        import graphviz

    format = None
    if isinstance(path, str):
        format = path.split('.')[-1]
    elif isinstance(path, BytesIO):
        format = 'svg'
    graph = graphviz.Graph(name, format=format)

    if not isinstance(meta_table_configs, list):
        meta_table_configs = [meta_table_configs]

    meta_table_configs_dict = {
        f'{meta_table_config.data_source}.{meta_table_config.table_name}':
        meta_table_config
        for meta_table_config in meta_table_configs
    }

    # create list of data sources
    data_sources = set(table.data_source
                       for table in meta_table_configs_dict.values())

    for table_name_full, meta_table_config in meta_table_configs_dict.items():
        keys = []
        if meta_table_config.pkey is not None:
            keys += [f'{meta_table_config.pkey.name} (PK)']
        keys += [f'{fkey.name} (FK)' for fkey in meta_table_config.fkeys]
        if meta_table_config.time_col:
            keys += [f'{meta_table_config.time_col.name} (Time)']

        keys = left_align(keys)

        cols = []
        if show_cols and meta_table_config.cols:
            cols += [
                f'{col.name} : {get_stype(col.stype)}'
                for col in meta_table_config.cols
            ]
            cols = left_align(cols)

        # if there are multiple data sources used, then use fully qualified
        # name. Otherwise just use the table name
        heading = []
        if len(data_sources) > 1:
            table_name = table_name_full
            heading += [f'{meta_table_config.table_name}']
            heading += [f'Connector: {meta_table_config.data_source}']
            heading = left_align(heading)
        else:
            table_name = meta_table_config.table_name
            heading += [f'{table_name}']
            heading = left_align(heading)

        if cols:
            label = f'{{{heading}|{keys}|{cols}}}'
        else:
            label = f'{{{heading}|{keys}}}'

        graph.node(table_name, shape='record', label=label)

        # dst_table in ForeignKey must be in the format of
        # {data_source}.{name} or {name}
        for fkey in meta_table_config.fkeys:
            fkey_dst_table_name_full = fkey.dst_table
            if fkey_dst_table_name_full is None:
                continue

            dst_key = fkey.dst_key

            fkey_dst_table_name = fkey_dst_table_name_full
            if len(data_sources) == 1:
                fkey_dst_table_name = get_table_name(fkey_dst_table_name_full)
            fkey_dst_table = meta_table_configs_dict[fkey_dst_table_name_full]
            dst_key = fkey_dst_table.pkey.name

            if fkey.name != dst_key:  # Print both key names if different:
                label = f' {fkey.name}\n< >\n{dst_key} '
            else:
                label = f' {fkey.name} '

            if (fkey_dst_table.pkey is not None
                    and fkey_dst_table.pkey.name == dst_key):
                headlabel, taillabel = '1', '*'
            else:
                headlabel, taillabel = '*', '*'

            graph.edge(table_name, fkey_dst_table_name, label=label,
                       headlabel=headlabel, taillabel=taillabel, minlen='2',
                       fontsize='11pt', labeldistance='1.5')

    if isinstance(path, str):
        path = '.'.join(path.split('.')[:-1])
        graph.render(path, cleanup=True)
    elif isinstance(path, BytesIO):
        path.write(graph.pipe())
    else:
        graph.view()


def plot_forecasting(
        df: pd.DataFrame,
        entity_col: str,
        filepath: str,
        target_col: str = 'TARGET',
        pred_col: str = 'TARGET_PRED',
        time_col: str = 'TIMESTAMP',
        plot_random: bool = True,
        random_state: int = 42,
        num_plot_entities: int = 12,
        time_unit: pd.Timedelta = pd.Timedelta(1, 'd'),
):
    r"""Plot forecasting truth targets and predicted targets.

    Args:
        df (pd.DataFrame): The DataFrame from validation or test.
        entity_col (str): Entity column name.
        filepath (str): File path to save the plot.
        target_col (str): Target column name. (default: :obj:`TARGET`)
        pred_col (str): Predicted target column name.
            (default: :obj:`TARGET_PRED`)
        time_col (str): Time column name. (default: :obj:`TIMESTAMP`)
        plot_random (bool): Whether randomly select entities to plot.
            (default: :obj:`True`)
        random_state (int): The seed to sample the entities to plot.
            (default: :obj:`42`)
        num_plot_entities (int): Number of entities to plot.
            (default: :obj:`12`)
        time_unit (pd.Timedelta): Time unit for the forecasting.
            (default: :obj:`pd.Timedelta(1, 'd')`)
    """
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use("Agg")

    if plot_random:
        samples = df.sample(n=num_plot_entities, random_state=random_state)
    else:
        samples = df[:num_plot_entities]
    forecast_length = len(df.iloc[0][target_col])

    # Let ncols and nrows as closed as possible to
    # make the plot looks better:
    sqrt_num = int(math.sqrt(num_plot_entities))
    ncols = num_plot_entities
    for nrows in range(sqrt_num, 0, -1):
        if ncols % nrows == 0:
            ncols = ncols // nrows
            break

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
    fig.set_size_inches(24, 12)
    fig.subplots_adjust(wspace=0.5)
    fig.subplots_adjust(hspace=0.5)

    for (_, (_, row)), ax in zip(enumerate(samples.iterrows()), axes.flatten()):
        targets = row[target_col]
        preds = row[pred_col]
        timestamps = [(i + 1) * time_unit + row[time_col]
                      for i in range(forecast_length)]
        timestamps = [
            timestamp.strftime('%Y-%m-%d %X') for timestamp in timestamps
        ]
        ax.plot(timestamps, targets, '-o', label=target_col)
        ax.plot(timestamps, preds, '-o', label=pred_col)
        title_max_length = 25
        if len(row[entity_col]) > title_max_length:
            title = row[entity_col][:title_max_length] + '...'
        else:
            title = row[entity_col]
        ax.title.set_text(title)
        ax.legend()

    fig.tight_layout()

    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=300)
    fsutil.write_binary(filepath, buffer.getvalue())
    plt.close(fig)
