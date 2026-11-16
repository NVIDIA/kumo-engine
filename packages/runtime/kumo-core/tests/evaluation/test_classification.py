# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Literal

import pytest
import torch
import torchmetrics
from sklearn.metrics import (
    auc,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from torch import Tensor

from kumo_core.evaluation.classification import BinaryAUPRC, BinaryAUROC, BinaryAveragePrecision, BinaryF1Score, MultilabelAUPRC, MultilabelAUROC, MultilabelAveragePrecision


def test_auroc() -> None:
    m = BinaryAUROC()
    for _ in range(2):  # stress-testing reset()
        m.reset()
        pred = torch.rand(200)
        target = (torch.rand(200) >= 0.5).to(torch.long)
        for y_hat, y in zip(pred.split(10), target.split(10)):
            m.update(y_hat, y)
        auroc = m.compute()  # type: ignore
        assert auroc is not None
        expected_auroc = roc_auc_score(target.numpy(), pred.numpy())
        assert torch.allclose(auroc, torch.tensor(expected_auroc))


def test_auprc() -> None:
    m = BinaryAUPRC()
    for _ in range(2):  # stress-testing reset()
        m.reset()
        pred = torch.rand(200)
        target = (torch.rand(200) >= 0.5).to(torch.long)
        for y_hat, y in zip(pred.split(10), target.split(10)):
            m.update(y_hat, y)

        auprc = m.compute()  # type: ignore
        precision, recall, _ = precision_recall_curve(target.numpy(),
                                                      pred.numpy())
        expected_auprc = auc(recall, precision)
        assert torch.allclose(auprc, torch.tensor(expected_auprc))


def test_ap() -> None:
    m = BinaryAveragePrecision()
    for _ in range(2):  # stress-testing reset()
        m.reset()
        pred = torch.rand(200)
        target = (torch.rand(200) >= 0.5).to(torch.long)
        for y_hat, y in zip(pred.split(10), target.split(10)):
            m.update(y_hat, y)
        ap = m.compute()  # type: ignore
        assert ap is not None
        expceted_ap = average_precision_score(target.numpy(), pred.numpy())
        assert torch.allclose(ap, torch.tensor(expceted_ap))


def test_f1() -> None:
    m = BinaryF1Score()
    for _ in range(2):  # stress-testing reset()
        m.reset()
        pred = torch.rand(200)
        target = (torch.rand(200) >= 0.5).to(torch.long)
        for y_hat, y in zip(pred.split(10), target.split(10)):
            m.update(y_hat, y)
        max_f1 = m.compute()  # type: ignore
        # Max F1 score should be at least as large as thresholded F1 score:
        f1 = f1_score(target.numpy(), (pred >= 0.5).to(torch.long).numpy())
        assert float(max_f1) >= float(f1)


@pytest.mark.parametrize('average', [None, 'none', 'micro', 'macro'])
def test_multilabel_auprc(
        average: Literal['micro', 'macro', 'none'] | None) -> None:
    auprc_metric = MultilabelAUPRC(num_labels=8, average=average)
    for _ in range(2):  # stress-testing reset()
        auprc_metric.reset()
        pred = torch.rand(200, 8)
        target = (torch.rand(200, 8) >= 0.5).to(torch.long)
        for y_hat, y in zip(pred.split(10), target.split(10)):
            auprc_metric.update(y_hat, y)

        auprc = auprc_metric.compute()  # type: ignore
        if average in {None, 'none', 'macro'}:
            expected_auprc_list = []
            for col in range(pred.shape[1]):
                precision, recall, _ = precision_recall_curve(
                    target[:, col].numpy(), pred[:, col].numpy())
                expected_auprc = auc(recall, precision)
                expected_auprc_list.append(expected_auprc)
            expected_auprc = torch.tensor(expected_auprc_list,
                                          dtype=torch.float32)
            if average in {None, 'none'}:
                assert torch.allclose(auprc, expected_auprc)
            else:
                assert torch.allclose(auprc, expected_auprc.mean())
        else:  # average == 'micro'
            precision, recall, _ = precision_recall_curve(
                target.flatten().numpy(),
                pred.flatten().numpy())
            expected_auprc = torch.tensor(auc(recall, precision),
                                          dtype=torch.float32)
            assert torch.allclose(auprc, expected_auprc)


@pytest.mark.parametrize('average', [None, 'none', 'micro', 'macro'])
def test_multilabel_ap_parity_with_torchmetrics(
        average: Literal['micro', 'macro', 'none'] | None) -> None:
    ap = MultilabelAveragePrecision(num_labels=8, average=average)
    ap_torchmetrics = torchmetrics.classification.MultilabelAveragePrecision(
        num_labels=8, average=average)
    for _ in range(2):  # stress-testing reset()
        ap.reset()
        ap_torchmetrics.reset()
        pred = torch.rand(200, 8)
        target = (torch.rand(200, 8) >= 0.5).to(torch.long)
        for y_hat, y in zip(pred.split(10), target.split(10)):
            ap.update(y_hat, y)
            ap_torchmetrics.update(y_hat, y)
        assert torch.allclose(
            ap.compute(),  # type: ignore
            ap_torchmetrics.compute(),  # type: ignore
        )


@pytest.mark.parametrize('average', [None, 'none', 'micro', 'macro'])
def test_multilabel_auroc_parity_with_torchmetrics(
        average: Literal['micro', 'macro', 'none'] | None) -> None:
    auroc = MultilabelAUROC(num_labels=8, average=average)
    auroc_torchmetrics = torchmetrics.classification.MultilabelAUROC(
        num_labels=8, average=average)
    for _ in range(2):  # stress-testing reset()
        auroc.reset()
        auroc_torchmetrics.reset()
        pred = torch.rand(200, 8)
        target = (torch.rand(200, 8) >= 0.5).to(torch.long)
        for y_hat, y in zip(pred.split(10), target.split(10)):
            auroc.update(y_hat, y)
            auroc_torchmetrics.update(y_hat, y)
        assert torch.allclose(
            auroc.compute(),  # type: ignore
            auroc_torchmetrics.compute(),  # type: ignore
        )


@pytest.mark.parametrize(
    'metric_cls',
    [BinaryAUPRC, BinaryAUROC, BinaryAveragePrecision],
)
def test_all_pos_neg_binary(
    metric_cls: type[BinaryAUPRC | BinaryAUROC | BinaryAveragePrecision]
) -> None:
    m = metric_cls()
    for label in [0, 1]:
        pred = torch.rand(100)
        target = torch.full_like(pred, fill_value=label, dtype=torch.long)
        m.reset()
        m.update(pred, target)
        out = m.compute()  # type: ignore
        assert isinstance(out, Tensor)
        assert torch.isnan(out)


@pytest.mark.parametrize(
    'metric_cls',
    [MultilabelAUPRC, MultilabelAUROC, MultilabelAveragePrecision],
)
def test_all_pos_neg_multilabel(
    metric_cls: type[MultilabelAUPRC | MultilabelAUROC
                     | MultilabelAveragePrecision]
) -> None:
    m_dict = {}
    average_list: list[Literal['none', 'micro', 'macro']
                       | None] = ['none', 'micro', 'macro', None]
    for average in average_list:
        m_dict[average] = metric_cls(average=average, num_labels=8)

    for label in [0, 1]:
        pred = torch.rand(200, 8)
        target = (torch.rand(200, 8) >= 0.5).to(torch.long)
        # artificially create all zero/one scenario in the 0-th column
        target[:, 0] = label
        out_dict = {}
        for average in average_list:
            m = m_dict[average]
            m.reset()
            m.update(pred, target)
            out_dict[average] = m.compute()

        assert torch.isnan(out_dict['none'][:1])
        assert torch.allclose(out_dict['none'][1:].mean(), out_dict['macro'])
