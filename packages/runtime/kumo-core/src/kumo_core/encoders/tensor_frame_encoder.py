# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Literal

import torch
from kumoapi.encoder import Datetime, Encoder, Null
from kumoapi.typing import Stype
from torch import Tensor
from torch_frame import TensorFrame
from torch_frame.typing import TensorData
from torch_geometric.config_mixin import ConfigMixin
from torch_geometric.config_store import register
from torch_geometric.nn.module_dict import ModuleDict
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.typing import NodeType

import kumo_core.modeling.context as modeling_context
from kumo_core.encoders.tensor_mapper import TimestampTensorMapper
from kumo_core.modeling.base import Module
from kumo_core.encoders.relative_time import RelTimeEncoder
from kumo_core.encoders.fused.categorical import FusedCategoricalEncoder
from kumo_core.encoders.fused.base import FusedEncoder
from kumo_core.encoders.fused.multicategorical import FusedMultiCategoricalEncoder
from kumo_core.encoders.fused.numerical import FusedNumericalEncoder
from kumo_core.encoders.fused.sequence import FusedSequenceEncoder
from kumo_core.encoders.fused.text import FusedTextEncoder
from kumo_core.encoders.fused.timestamp import FusedTimestampEncoder
from kumo_core.types.features import NANOSECONDS_IN_SECONDS

# TODO(akihironitta): Use stype instead of stype.value once PyTorch
# addresses https://github.com/pytorch/pytorch/issues/111603
_STYPE_TO_ENCODER: dict[str, type[FusedEncoder]] = {
    Stype.categorical.value: FusedCategoricalEncoder,
    Stype.multicategorical.value: FusedMultiCategoricalEncoder,
    Stype.numerical.value: FusedNumericalEncoder,
    Stype.sequence.value: FusedSequenceEncoder,
    Stype.text.value: FusedTextEncoder,
    Stype.timestamp.value: FusedTimestampEncoder,
}
_DEFAULT_ENC_MODE: Literal['fuse', 'column'] = 'fuse'


class TensorFrameEncoder(torch.nn.Module):
    r"""Encodes the column features of a table into a shared embedding space.

    Args:
        name: The name of the table to encode.
        col_encoder_dict: The column-level encoders for this table.
        time_col: The time column of the table (in case it exists).
        out_channels: Size of each output sample.
        enc_mode: The method used to aggregate features across columns.
        act: The non-linear activation function to use.
        act_kwargs: Arguments passed to the respective activation function
            defined by ``act``.
        use_seq_id: If set to :obj:`True`, will also use positional encodings
            to encode the sequence ID of the group of facts belonging to an
            entity.
        exclude_cols: If specified, excludes the columns from encoding.
        device: The device to use.
    """
    def __init__(
        self,
        name: str,
        col_encoder_dict: dict[str, Encoder],
        time_col: str | None,
        out_channels: int,
        enc_mode: Literal['fuse', 'column'] = _DEFAULT_ENC_MODE,
        act: str | None = 'relu',
        act_kwargs: dict[str, Any] | None = None,
        use_seq_id: bool = False,
        exclude_cols: list[str] | None = None,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.name = name
        self.time_col = time_col
        self.out_channels = out_channels
        self.enc_mode: Literal['fuse', 'column'] = enc_mode
        self.act = activation_resolver(act, **(act_kwargs or {}))
        self.use_seq_id = use_seq_id
        self.exclude_cols = exclude_cols or []
        self.device = device

        self.stype_encoder_dict = self._create_stype_encoder_dict(
            col_encoder_dict)

        # Handle relative time encoding #######################################
        time_cols = [
            col_name for col_name, mapper in col_encoder_dict.items()
            if isinstance(mapper, Datetime)
        ]
        self.rel_time_encoder: RelTimeEncoder | None = None
        if len(time_cols) > 0 or time_col is not None:
            # If the table is temporal, collect all time columns to compute
            # relative temporal encodings for:
            if time_col is not None and time_col not in time_cols:
                time_cols = [time_col] + time_cols
            self.rel_time_encoder = RelTimeEncoder(
                num_cols=len(time_cols),
                out_channels=self.out_channels,
                enc_mode=self.enc_mode,
                use_seq_id=self.use_seq_id,
                device=self.device,
            )

        self.reset_parameters()

    def _create_stype_encoder_dict(
        self,
        col_encoder_dict: dict[str, Encoder],
    ) -> ModuleDict:
        # TODO(akihironitta): Use stype instead of stype.value once PyTorch
        # addresses https://github.com/pytorch/pytorch/issues/111603
        stype_encoder_dicts: dict[str, dict[str, Encoder]] = defaultdict(dict)
        for col_name, encoder in col_encoder_dict.items():
            if isinstance(encoder, Null):
                continue
            stype = next(iter(encoder.supported_stypes)).to_parent_stype()
            stype_encoder_dicts[stype][col_name] = encoder

        stype_encoder_dict: dict[str, FusedEncoder] = {}
        for stype, encoder_dict in stype_encoder_dicts.items():
            stype_encoder_cls = _STYPE_TO_ENCODER.get(stype, None)
            if stype_encoder_cls is None:
                continue
            stype_encoder_dict[stype] = stype_encoder_cls(
                col_encoder_dict=encoder_dict,
                exclude_cols=self.exclude_cols,
                out_channels=self.out_channels,
                enc_mode=self.enc_mode,
                device=self.device,
            )
        return ModuleDict(stype_encoder_dict)

    def reset_parameters(self) -> None:
        for fused_encoder in self.stype_encoder_dict.values():
            fused_encoder.reset_parameters()
        if self.rel_time_encoder is not None:
            self.rel_time_encoder.reset_parameters()

    def forward_fixed(self, tf: TensorFrame) -> list[Any]:
        r"""The forward pass that doesn't include weights is decoupled from the
        :meth:`forward` to let ShallowRHSLinkGNNLightningModule cache the
        intermediate tensors in ``RHSEmbeddingMode.{FEATURE,FUSION}``.

        Args:
            tf: Tensor representation of a :class:`pandas.DataFrame` to encode.
        """
        outs: list[Any] = []
        for stype, fused_encoder in self.stype_encoder_dict.items():
            feat: TensorData = tf.feat_dict[stype]
            col_names = tf.col_names_dict[stype]
            outs.append(fused_encoder.forward_fixed(feat, col_names))
        return outs

    def forward_learnable(
        self,
        feats: list[Any],
    ) -> list[Tensor] | Tensor | None:
        """Forward pass of forward_learnable of all fused encoders.

        Args:
            feats: A list of features encoded by
                :meth:`FusedEncoder.forward_fixed`.

        Returns:
            list[Tensor]: A list of tensors or an empty list if
                ``enc_mode="column"``.
            Tensor | None: A tensor or :obj:`None` if ``enc_mode="fuse"``.
        """
        if self.enc_mode == 'column':
            outs: list[Tensor] = []
            for feat, fused_encoder in zip(feats,
                                           self.stype_encoder_dict.values()):
                outs.append(fused_encoder.forward_learnable(feat))
                # outs_list: list[[num_rows, num_all_cols, out_channels]]
            return outs
        else:  # self.enc_mode == 'fuse':
            out: Tensor | None = None
            for feat, fused_encoder in zip(feats,
                                           self.stype_encoder_dict.values()):
                tmp = fused_encoder.forward_learnable(feat)
                out = tmp if out is None else out + tmp
            return out

    def forward_fixed_and_learnable(
        self,
        tf: TensorFrame,
        batch: Tensor | None = None,
        seed_time: Tensor | None = None,
        time: Tensor | None = None,
    ) -> list[Tensor] | Tensor | None:
        """Forward pass of all fused encoders.

        Args:
            tf: Tensor representation of a :class:`pd.DataFrame` to encode.
            batch (Tensor, optional): The batch tensor to index the seed
                time tensor.
            seed_time (Tensor, optional): The seed time tensor.
            time (Tensor, optional): The time tensor for the time column.

        Returns:
            list[Tensor]: A list of tensors or an empty list if
                ``enc_mode="column"``.
            Tensor | None: A tensor or :obj:`None` if ``enc_mode="fuse"``.
        """
        time_feat: Tensor | None = None
        if seed_time is not None:
            seed_time = seed_time // NANOSECONDS_IN_SECONDS

            time_feat = tf.feat_dict.get(Stype.timestamp)
            if time_feat is not None:
                index = TimestampTensorMapper.index['unix']
                time_feat = time_feat[:, :, index]

            col_names = tf.col_names_dict.get(Stype.timestamp) or []
            if (time is not None and self.time_col is not None
                    and self.time_col not in col_names):
                time = time.view(-1, 1) // NANOSECONDS_IN_SECONDS
                if time_feat is None:
                    time_feat = time
                else:
                    time_feat = torch.cat([time, time_feat], dim=-1)

        if self.enc_mode == 'column':
            outs: list[Tensor] = []
            for stype, fused_encoder in self.stype_encoder_dict.items():
                feat = tf.feat_dict[stype]
                col_names = tf.col_names_dict[stype]
                outs.append(fused_encoder(feat, col_names))
            if time_feat is not None and seed_time is not None:
                assert self.rel_time_encoder is not None
                outs.append(self.rel_time_encoder(time_feat, seed_time, batch))
            return outs
        else:  # self.enc_mode == 'fuse':
            out: Tensor | None = None
            for stype, fused_encoder in self.stype_encoder_dict.items():
                feat = tf.feat_dict[stype]
                tmp = fused_encoder(feat)
                out = tmp if out is None else out + tmp
            if time_feat is not None and seed_time is not None:
                assert self.rel_time_encoder is not None
                tmp = self.rel_time_encoder(time_feat, seed_time, batch)
                out = tmp if out is None else out + tmp
            return out

    def forward_post_fused_encoders(
        self,
        outs: list[Tensor] | Tensor | None,
        batch: Tensor,
    ) -> Tensor:
        if outs is None or (isinstance(outs, list) and len(outs) == 0):
            if self.enc_mode == 'column':
                outs = [
                    torch.ones(
                        batch.numel(),
                        1,
                        self.out_channels,
                        device=batch.device,
                    )
                ]
            else:  # self.enc_mode == 'fuse':
                outs = torch.ones(
                    batch.numel(),
                    self.out_channels,
                    device=batch.device,
                )

        if self.enc_mode == 'column':
            assert not isinstance(outs, Tensor)
            # num_all_cols += 1 if is_temporal
            # num_all_cols += 1 if use_seq_id
            # out: [num_rows, num_all_cols, out_channels]
            out = torch.cat(outs, dim=1) if len(outs) > 1 else outs[0]
            # TODO: Add table convolutions & decoders from PyTorch Frame
            out = out.sum(dim=1)
        else:  # self.enc_mode == 'fuse':
            assert isinstance(outs, Tensor)
            out = outs

        assert isinstance(out, Tensor)

        if self.act is not None:
            out = self.act(out)

        return out

    def forward(
        self,
        tf: TensorFrame,
        batch: Tensor,
        time: Tensor | None = None,
        seed_time: Tensor | None = None,
    ) -> Tensor:
        """Encodes columns of a table into a shared embedding space.

        Args:
            tf: Tensor representation of a :class:`pandas.DataFrame` to encode.
            batch: Batch vector mapping each row to its corresponding
                example.
            time: The time value for each time column.
            seed_time: The seed time tensor.

        Returns:
            Tensor of size ``[num_rows, out_channels]``.
        """
        out = self.forward_post_fused_encoders(
            self.forward_fixed_and_learnable(tf, batch, seed_time, time),
            batch,
        )

        if modeling_context.is_debugging() and out.isnan().sum() > 0:
            raise ValueError(f"'{self.__class__.__name__}' produced NaN "
                             f"values for the table '{self.name}'")

        return out


@register(with_target=True)
class HeteroTensorFrameEncoder(Module, ConfigMixin):
    # Tainer takes care of removing table_encoder_dict
    # from the configuration before prediction time.
    LAZY_ATTRS = {
        'table_encoder_dict',
        'time_col_dict',
        'out_channels',
        'exclude_cols_dict',
    }

    def __init__(
        self,
        table_encoder_dict: dict[str, dict[str, Encoder]] | None = None,
        time_col_dict: dict[str, str] | None = None,
        out_channels: int | None = None,
        enc_mode: Literal['fuse', 'column'] = _DEFAULT_ENC_MODE,
        use_seq_id: bool = False,
        exclude_cols_dict: dict[str, list[str]] | None = None,
        act: str | None = 'relu',
        act_kwargs: dict[str, Any] | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.table_encoder_dict: dict[str, dict[str, Encoder]] | None
        super().__init__(
            table_encoder_dict=table_encoder_dict,
            time_col_dict=time_col_dict,
            out_channels=out_channels,
            enc_mode=enc_mode,
            act=act,
            act_kwargs=act_kwargs,
            use_seq_id=use_seq_id,
            exclude_cols_dict=exclude_cols_dict,
            device=device,
        )

    def _set_exclude_cols_dict(
        self,
        exclude_cols_dict: dict[str, list[str]],
    ) -> None:
        self.exclude_cols_dict = exclude_cols_dict
        for node_type, encoder in self.enc_dict.items():
            exclude_cols = exclude_cols_dict.get(node_type) or []

            encoder.exclude_cols = exclude_cols
            for stype_encoder in encoder.stype_encoder_dict.values():
                stype_encoder.exclude_cols = exclude_cols
                device = stype_encoder.exclude_cols_mask.device
                mask = stype_encoder._create_exclude_cols_mask(device)
                stype_encoder.register_buffer('exclude_cols_mask', mask)

    @contextmanager
    def set_exclude_cols_dict(
        self,
        exclude_cols_dict: dict[str, list[str]],
    ) -> Iterator[None]:

        orig_exclude_cols_dict = self.exclude_cols_dict
        self._set_exclude_cols_dict(exclude_cols_dict)
        yield
        self._set_exclude_cols_dict(orig_exclude_cols_dict)

    def init_modules(self) -> None:
        self.enc_dict: torch.nn.ModuleDict[
            str,
            TensorFrameEncoder,
        ] = ModuleDict()
        assert self.table_encoder_dict is not None
        for node_type in self.exclude_cols_dict.keys():
            assert node_type in self.table_encoder_dict
        for node_type in self.table_encoder_dict.keys():
            self.enc_dict[node_type] = TensorFrameEncoder(
                name=node_type,
                col_encoder_dict=self.table_encoder_dict[node_type],
                time_col=self.time_col_dict.get(node_type),
                out_channels=self.out_channels,
                enc_mode=self.enc_mode,
                act=self.act,
                act_kwargs=self.act_kwargs,
                use_seq_id=self.use_seq_id,
                exclude_cols=self.exclude_cols_dict.get(node_type),
                device=self.device,
            )

    def reset_parameters(self) -> None:
        for tf_encoder in self.enc_dict.values():
            tf_encoder.reset_parameters()

    def forward(
        self,
        tf_dict: dict[NodeType, TensorFrame],
        batch_dict: dict[NodeType, Tensor],
        time_dict: dict[NodeType, Tensor] | None = None,
        seed_time: Tensor | None = None,
    ) -> dict[NodeType, Tensor]:
        """Encodes columns of all tables.

        Args:
            tf_dict: Dictionary of tensor representation of a DataFrame to
                encode per node type.
            batch_dict: Dictionary of batch vector mapping each row to its
                corresponding example per node type.
            time_dict: The time column tensor for each table.
            seed_time: The seed time tensor.

        Returns:
            Dictionary of tensor of size ``[num_rows, out_channels]``.
        """
        out_dict: dict[NodeType, Tensor] = {}
        for node_type in tf_dict.keys():
            tf_encoder = self.enc_dict[node_type]
            out_dict[node_type] = tf_encoder(
                tf=tf_dict[node_type],
                batch=batch_dict[node_type],
                time=(time_dict or {}).get(node_type),
                seed_time=seed_time,
            )
        return out_dict
