// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { HighlightTypes, HoverHintsType } from "types/codeEditor";

export const yamlDictionary: HoverHintsType = {
    "encoder_overrides": {
        title: "encoder_overrides",
        subtitle: "encoder_overrides (Optional)",
        hint: `The encoder_overrides field allows you to configure the way Kumo processes your input data and override the encoders that are inferred by Kumo.

        Note that nested fields require exactly 4 spaces. Each column must be specified with two case sensitive fields.
        encoder_overrides: &lttable_name&gt.&ltcolumn_name&gt: &ltencoder&gt`,
        link: "https://kumo.ai/docs/column_processing#encoder-overrides",
        type: HighlightTypes.KEYWORD
    },
    "activation": {
        title: "activation",
        subtitle: "activation: &ltList[string]&gt (Optional)",
        hint: `A list of activation functions to use during AutoML. Possible values are relu, leaky_relu, elu, and gelu.`,
        link: "https://kumo.ai/docs/activation",
        type: HighlightTypes.KEYWORD
    },
    "aggregation": {
        title: "aggregation",
        subtitle:
            "aggregation: &ltList[&ltList[sum | mean | max | min | std]]&gt (Optional)",
        hint: `A nested list of aggregation operators in graph neural network aggregation process.`,
        link: "https://kumo.ai/docs/aggregation",
        type: HighlightTypes.KEYWORD
    },
    "channels": {
        title: "channels",
        subtitle: "channels: List[int] (Optional)",
        hint: `A list of potential dimension of layers in graph neural network model.

        Channels must be an even number between >= 2 and <= 1024.`,
        link: "https://kumo.ai/docs/channels",
        type: HighlightTypes.KEYWORD
    },
    "handle_new_target_entities": {
        title: "handle_new_target_entities",
        subtitle:
            "handle_new_target_entities: (false (default) | true) (Optional)",
        hint: `Whether to make link prediction models be able to handle predictions on new target entities at batch prediction time.

        Keeping this option to false generally provides better model performance. It is recommended to keep this option to False unless you need to predict unseen target entities at batch prediction time.`,
        link: "https://kumo.ai/docs/handle_new_target_entities",
        type: HighlightTypes.KEYWORD
    },
    "module": {
        title: "module",
        subtitle: "module: (ranking (default) | embedding) (Optional)",
        hint: `By default, Kumo selects the model architecture and designs the hyperparameter search space that best fits your query. However, if you have specific requirements for your use case, you can customize the model architecture to suite your desired output/objective.

        The module to choose from: link_prediction_embedding, link_prediction_ranking.`,
        link: "https://kumo.ai/docs/module",
        type: HighlightTypes.KEYWORD
    },
    "normalization": {
        title: "normalization",
        subtitle: "normalization: &ltList[string | None]&gt (Optional)",
        hint: `The normalization layer to apply. Possible values are batch_norm, layer_norm or None to skip normalization.`,
        link: "https://kumo.ai/docs/normalization",
        type: HighlightTypes.KEYWORD
    },
    "num_post_message_passing_layers": {
        title: "num_post_message_passing_layers",
        subtitle:
            "num_post_message_passing_layers: &ltList[&ltinteger&gt]&gt (Optional)",
        hint: `A list of potential number of multilayer perceptron layers after message passing layer in graph neural network model.

        num_post_mp_layers must be > 0 and <= 8.`,
        link: "https://kumo.ai/docs/num_post_message_passing_layers",
        type: HighlightTypes.KEYWORD
    },
    "num_pre_message_passing_layers": {
        title: "num_pre_message_passing_layers",
        subtitle:
            "num_pre_message_passing_layers: &ltList[&ltinteger&gt]&gt (Optional)",
        hint: `A list of potential number of multilayer perceptron layers before message passing layer in graph neural network model.
        num_pre_message_passing_layers must be >= 0 and <= 4.`,
        link: "https://kumo.ai/docs/num_pre_message_passing_layers",
        type: HighlightTypes.KEYWORD
    },
    "ranking_embedding_loss_coeff": {
        title: "ranking_embedding_loss_coeff",
        subtitle: "lp_ranking_emb_loss_coeff: &ltfloat&gt (Optional)",
        hint: `The coefficient of embedding loss applied to train ranking-based link prediction model. By default, ranking_emb_loss_coeff is set to 0., which means that the model is solely trained to optimize for the final predicted ranking.

        For embedding-based ranking predictions, set lp_ranking_emb_loss_coeff to a positive value. This instructs Kumo to explicitly train embeddings to produce high-quality inner-product scores for ranking prediction.

        It is only recommended to change this value in case the user is interested in embeddings. In this case, the recommended value of ranking_emb_loss_coeff is 0.2.`,
        link: "https://kumo.ai/docs/ranking_embedding_loss_coeff",
        type: HighlightTypes.KEYWORD
    },
    "output_embedding_dim": {
        title: "output_embedding_dim",
        subtitle: "output_embedding_dim: (Optional)",
        hint: `The output embedding dimension for link prediction models.

        output_embedding_dim must be > 0 and <= 256.`,
        link: "https://kumo.ai/docs/output_embedding_dim",
        type: HighlightTypes.KEYWORD
    },
    "target_embedding_mode": {
        title: "target_embedding_mode",
        subtitle:
            "target_embedding_mode: (lookup (default) | feature | fusion) (Optional)",
        hint: `Specifies the mode of target node embeddings to use in the final link prediction calculation (i.e., dot product). The mode should be either lookup, feature, or fusion.

        By default, it is set to lookup mode, which will use lookup/shallow embeddings for target nodes.

        The feature mode will generate target node embeddings based purely on target node features.

        The fusion mode will fuse the the lookup embeddings and feature-based embeddings.`,
        link: "https://kumo.ai/docs/target_embedding_mode",
        type: HighlightTypes.KEYWORD
    },
    "use_seq_id": {
        title: "use_seq_id",
        subtitle: "use_seq_id: (false (default) | true) (Optional)",
        hint: `If set to True, will use positional encodings of the sequence order of facts as an additional model feature.

        By default, use_seq_id is set to False.`,
        link: "https://kumo.ai/docs/use_seq_id",
        type: HighlightTypes.KEYWORD
    },
    "max_target_neighbors_per_entity": {
        title: "max_target_neighbors_per_entity",
        subtitle: "max_target_neighbors_per_entity: &ltinteger&gt (Optional)",
        hint: `Determines the maximum amount of neighbors of the target table to sample for a single entity.

        By default, max_target_neighbors_per_entity will be determined using edge degree statistics between entity table and target table.`,
        link: "https://kumo.ai/docs/max_target_neighbors_per_entity",
        type: HighlightTypes.KEYWORD
    },
    // TODO: Revisit this hint definition!
    "num_neighbors": {
        title: "num_neighbors",
        subtitle: "num_neighbors: &ltList[&ltinteger&gt]&gt (Optional)",
        hint: `Kumo’s GNN-based predictions use node feature information to efficiently generate node embeddings for previously unseen data. Rather than train individual embeddings for each node, Kumo generates embeddings by sampling and aggregating features from a node's local neighborhood.

        The num_neighbors parameter determines the default number of neighbors to sample for each connection in each hop (i.e., how many neighbors are sampled for each node in each iteration).`,
        link: "https://kumo.ai/docs/num_neighbors",
        type: HighlightTypes.KEYWORD
    },
    "sample_from_entity_table": {
        title: "sample_from_entity_table",
        subtitle:
            "sample_from_entity_table: (true (default) | false) (Optional)",
        hint: `Whether to allow sampling of other entities in the entity table besides the seed entity. Useful to set to false in case entities represent candidates/hypothetical examples and to restrict information flow between different candidates.`,
        link: "https://kumo.ai/docs/sample_from_entity_table",
        type: HighlightTypes.KEYWORD
    },
    "base_lr": {
        title: "base_lr",
        subtitle: "base_lr: &ltList[&ltfloat&gt]&gt (Optional)",
        hint: `A list of potential learning rates for model optimization.
        base_lr must be > 0.`,
        link: "https://kumo.ai/docs/base_lr",
        type: HighlightTypes.KEYWORD
    },
    "batch_size": {
        title: "batch_size",
        subtitle: "batch_size: &ltList[&ltint&gt]&gt (Optional)",
        hint: `A list of potential batch sizes during data loading.

        batch_size must be <= 2048.`,
        link: "https://kumo.ai/docs/batch_size",
        type: HighlightTypes.KEYWORD
    },
    "early_stopping": {
        title: "early_stopping",
        subtitle:
            "early_stopping: &ltList[&ltlist&gt[&ltEarlyStopper&gt]]&gt (Optional)",
        hint: `A list of potential early stopping strategies for model optimization.`,
        link: "https://kumo.ai/docs/early_stopping",
        type: HighlightTypes.KEYWORD
    },
    "lr_scheduler": {
        title: "lr_scheduler",
        subtitle:
            "lr_scheduler: &ltList[&ltList&gt[&ltScheduler&gt]]&gt (Optional)",
        hint: `A list of potential learning rate schedulers for model optimization.`,
        link: "https://kumo.ai/docs/lr_scheduler",
        type: HighlightTypes.KEYWORD
    },
    "name": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.STRING
    },
    "interval": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.STRING
    },
    "kwargs": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.STRING
    },
    "majority_sampling_ratio": {
        title: "majority_sampling_ratio",
        subtitle: "majority_sampling_ratio: &ltfloat&gt (Optional)",
        hint: `This ratio specifies how examples are sampled from the majority class when the task is binary classification.
        This means that for every example in the minority class, we sample majority_sampling_ratio number of examples from the majority class. majority_sampling_ratio must be > 0 in case it is specified.

        For example, a ratio of 10 means that we will sample the examples of the majority class with ratio 1:10, or 10 majority class examples for every minority class example.

        If the ratio of majority:minority is greater than majority_sampling_ratio:1, we will sample uniformly, ignoring this value.

        For example, if the ratio of majority:minority = 10:1 and we choose majority_sampling_ratio=9 then we will ignore this setting.`,
        link: "https://kumo.ai/docs/majority_sampling_ratio",
        type: HighlightTypes.KEYWORD
    },
    "max_epochs": {
        title: "max_epochs",
        subtitle: "max_epochs: &ltinteger&gt (Optional)",
        hint: `The maximum number of epochs to train.`,
        link: "https://kumo.ai/docs/max_epochs",
        type: HighlightTypes.KEYWORD
    },
    "max_steps_per_epoch": {
        title: "max_steps_per_epoch",
        subtitle: "max_steps_per_epoch: &ltinteger&gt (Optional)",
        hint: `The maximum number of steps to train in a single epoch. It is useful to customize time spent on training versus validation for large training data.`,
        link: "https://kumo.ai/docs/max_steps_per_epoch",
        type: HighlightTypes.KEYWORD
    },
    "min_steps_per_epoch": {
        title: "min_steps_per_epoch",
        subtitle: "min_steps_per_epoch: &ltinteger&gt (Optional)",
        hint: "The minimum number of steps to train in a single epoch. Useful to tune for small-scale datasets.",
        link: "",
        type: HighlightTypes.KEYWORD
    },
    "max_test_steps": {
        title: "max_test_steps",
        subtitle: "max_test_steps: &ltinteger&gt (Optional)",
        hint: `The maximum number of test steps to perform. If set, max_test_steps will override the number of test steps set by run_mode. Setting max_test_steps is useful for customizing the time spent on testing the model.`,
        link: "https://kumo.ai/docs/max_test_steps",
        type: HighlightTypes.KEYWORD
    },
    "min_test_steps": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.KEYWORD
    },
    "max_val_steps": {
        title: "max_val_steps",
        subtitle: "max_val_steps: &ltinteger&gt (Optional)",
        hint: `The maximum number of validation steps to perform. If set, max_val_steps will override the number of steps set by run_mode. Setting max_val_steps is useful for customizing the time spent on validation the model.`,
        link: "https://kumo.ai/docs/max_val_steps",
        type: HighlightTypes.KEYWORD
    },
    "min_val_steps": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.KEYWORD
    },
    "weight_decay": {
        title: "weight_decay",
        subtitle: "weight_decay: &ltList[&ltinteger&gt]&gt (Optional)",
        hint: `A list of potential weight decay parameters for model optimization.

        weight_decay must be >= 0.`,
        link: "https://kumo.ai/docs/weight_decay",
        type: HighlightTypes.KEYWORD
    },
    "weight_mode": {
        title: "weight_mode",
        subtitle: "weight_mode: &ltstr&gt (Optional)",
        hint: ` weight_mode specifies how to use weight column, if present, in training:

        - sample: samples training examples with replacement according to the weight distribution.
        - weighted_loss: weighs training examples in the loss function according to the weight distribution.

        weight_mode is only activated when weight col is set or for binary classifcation`,
        link: "https://kumo.ai/docs/weight_mode",
        type: HighlightTypes.KEYWORD
    },
    "refit_full": {
        title: "refit_full",
        subtitle: "refit_full: (false (default) | true) (Optional)",
        hint: `Specifies whether to refit the best AutoML model (fitted on train data) on the full (train+val+test) data.`,
        link: "https://kumo.ai/docs/refit_full",
        type: HighlightTypes.KEYWORD
    },
    "split": {
        title: "split",
        subtitle:
            "split: (DateOffsetSplit | RandomSplit | TemporalSplit | TimeRangeSplit | ApproxDateOffsetSplit | TimeSplit) (Optional)",
        hint: `Kumo will generate a training table with entities and labels according to your pQuery to be used in GNN Training. A default split method will be used to split the generated training table to 3 disjoint train, validation and test set for training purpose.

        You can specify a custom split method and parameter to generate a training, validation, and test set by setting the split value in the Advanced Options box.`,
        link: "https://kumo.ai/docs/split",
        type: HighlightTypes.KEYWORD
    },
    "entity_candidate_aggregation": {
        title: "entity_candidate_aggregation",
        subtitle: "",
        hint: `This field can be used used to specify hypothetical future-looking information that the user can supply at batch prediction time, such as the type of discount sent to the user. It should always be used in conjunction with the entity_candidate field.

        The syntax of the aggregation follows the same instructions as the syntax for the target column.
        result_column_name should be the name of a column, provided inside of the entity_candidate table to supply this information at batch prediction time. The schema of the entity_candidate candidate table should contain the column with the entity primary key and &ltresult_column_name&gt, see entity_candidate_aggregation.`,
        link: "https://kumo.ai/docs/entity_candidate_aggregation-1",
        type: HighlightTypes.KEYWORD
    },
    "refit_trainval": {
        title: "refit_trainval",
        subtitle: "refit_trainval: (false (default) | true) (Optional)",
        hint: `Specifies whether to refit the best AutoML model (fitted on train data) on the training+validation data.`,
        link: "https://kumo.ai/docs/refit_trainval",
        type: HighlightTypes.KEYWORD
    },
    "metrics": {
        title: "metrics",
        subtitle: "metrics: List[str] (Optional)",
        hint: `This field can be specified to get additional reported metrics for evaluating your models. Available metrics vary per task, with incompatible metrics automatically filtered out (e.g., specifying metrics: [auroc] will result in auroc being filtered out, as it's not compatible to link prediction).`,
        link: "https://kumo.ai/docs/extra_metrics",
        type: HighlightTypes.KEYWORD
    },
    "tune_metric": {
        title: "tune_metric",
        subtitle:
            "tune_metric: (auroc, acc, mae, f1, loss, ap_macro) (Optional)",
        hint: `This is the metric that the model will use to optimize AutoML experiments. It also needs to be present in the list of metrics.`,
        link: "https://kumo.ai/docs/tune_metric",
        type: HighlightTypes.KEYWORD
    },
    "auroc_per_label": {
        title: "auroc_per_label",
        subtitle: `Area Under the Receiver Operating Characteristic curve per Label`,
        hint: `The AUROC stands for the area under the receiver operating characteristic curve, which is a curve plotting sensitivity against specificity under different classification thresholds.`,
        link: "https://kumo.ai/docs/evaluation#multilabel-classification",
        type: HighlightTypes.VARIABLE_NAME
    },
    "auroc_macro": {
        title: "auroc_macro",
        subtitle:
            "Unweighted mean of all the per-class Area Under the Receiver Operating Characteristic curve (AUROC) scores.",
        hint: `The AUROC stands for the area under the receiver operating characteristic curve, which is a curve plotting sensitivity against specificity under different classification thresholds.`,
        link: "https://kumo.ai/docs/evaluation#multilabel-classification",
        type: HighlightTypes.VARIABLE_NAME
    },
    "auroc_micro": {
        title: "auroc_micro",
        subtitle:
            "Global average of Area Under the Receiver Operating Characteristic curve (AUROC) scores.",
        hint: `The AUROC stands for the area under the receiver operating characteristic curve, which is a curve plotting sensitivity against specificity under different classification thresholds.`,
        link: "https://kumo.ai/docs/evaluation#multilabel-classification",
        type: HighlightTypes.VARIABLE_NAME
    },
    "auroc": {
        title: "auroc",
        subtitle: `Area Under the Receiver Operating Characteristic curve`,
        hint: `The AUROC stands for the area under the receiver operating characteristic curve, which is a curve plotting sensitivity against specificity under different classification thresholds.`,
        link: "https://kumo.ai/docs/classification#auroc",
        type: HighlightTypes.VARIABLE_NAME
    },
    "acc": {
        title: "Accuracy",
        subtitle: "",
        hint: `The accuracy tells you the overall percentage of correct predictions. The accuracy score is given by the total number of correct predictions over the number of samples.`,
        link: "https://kumo.ai/docs/classification#accuracy",
        type: HighlightTypes.VARIABLE_NAME
    },
    "auprc_macro": {
        title: "auprc_macro",
        subtitle: `Unweighted mean of all the per-class Area Under the Precision-Recall Curve (AUPRC) scores.`,
        hint: `The AUPRC stands for the area under the precision-recall curve, which is a curve plotting precision against recall under different classification thresholds.`,
        link: "https://kumo.ai/docs/evaluation#multilabel-classification",
        type: HighlightTypes.VARIABLE_NAME
    },
    "auprc_micro": {
        title: "auprc_micro",
        subtitle: `Global average of Area Under the Precision-Recall Curve (AUPRC) scores.`,
        hint: `The AUPRC stands for the area under the precision-recall curve, which is a curve plotting precision against recall under different classification thresholds.`,
        link: "https://kumo.ai/docs/evaluation#multilabel-classification",
        type: HighlightTypes.VARIABLE_NAME
    },
    "auprc": {
        title: "auprc",
        subtitle: `Area Under the Precision-Recall Curve`,
        hint: `The AUPRC stands for the area under the precision-recall curve, which is a curve plotting precision against recall under different classification thresholds.`,
        link: "https://kumo.ai/docs/classification#auprc",
        type: HighlightTypes.VARIABLE_NAME
    },
    "ap_macro": {
        title: "ap_macro",
        subtitle: "",
        hint: `Unweighted mean of all the per-class Average Precision (AP) scores.`,
        link: "https://kumo.ai/docs/evaluation#multilabel-classification",
        type: HighlightTypes.VARIABLE_NAME
    },
    "ap_micro": {
        title: "ap_micro",
        subtitle: "",
        hint: `Global average of Average Precision (AP) scores.`,
        link: "https://kumo.ai/docs/evaluation#multilabel-classification",
        type: HighlightTypes.VARIABLE_NAME
    },
    "ap_per_label": {
        title: "ap_per_label",
        subtitle: "",
        hint: `Average Precision per Label`,
        link: "https://kumo.ai/docs/evaluation#multilabel-classification",
        type: HighlightTypes.VARIABLE_NAME
    },
    "ap": {
        title: "ap",
        subtitle: "Average Precision",
        hint: "",
        link: "https://kumo.ai/docs/evaluation#multilabel-classification",
        type: HighlightTypes.VARIABLE_NAME
    },
    "recall@100": {
        title: "recall@100",
        subtitle: "",
        hint: `The fraction of true positive instances among all true positive and flase negative instances combined. Recall@100 or Recall at 100 corresponds to the share of relevant results among the top 100 retrieved instances.`,
        link: "https://kumo.ai/docs/evaluation-metrics#link-prediction",
        type: HighlightTypes.VARIABLE_NAME
    },
    "recall@10": {
        title: "recall@10",
        subtitle: "",
        hint: `The fraction of true positive instances among all true positive and flase negative instances combined. Recall@10 or Recall at 10 corresponds to the share of relevant results among the top 10 retrieved instances.`,
        link: "https://kumo.ai/docs/evaluation-metrics#link-prediction",
        type: HighlightTypes.VARIABLE_NAME
    },
    "recall@1": {
        title: "recall@1",
        subtitle: "",
        hint: `The fraction of true positive instances among all true positive and flase negative instances combined. Recall@1 or Recall at 1 corresponds to the share of relevant results among the top 1 retrieved instance.`,
        link: "https://kumo.ai/docs/evaluation-metrics#link-prediction",
        type: HighlightTypes.VARIABLE_NAME
    },
    "recall@25": {
        title: "recall@25",
        subtitle: "",
        hint: `The fraction of true positive instances among all true positive and flase negative instances combined. Recall@25 or Recall at 25 corresponds to the share of relevant results among the top 25 retrieved instances.`,
        link: "https://kumo.ai/docs/evaluation-metrics#link-prediction",
        type: HighlightTypes.VARIABLE_NAME
    },
    "recall": {
        title: "recall",
        subtitle: "",
        hint: `The fraction of true positive instances among all true positive and flase negative instances combined.`,
        link: "https://kumo.ai/docs/evaluation-metrics#link-prediction",
        type: HighlightTypes.VARIABLE_NAME
    },
    "precision@25": {
        title: "precision@25",
        subtitle: "",
        hint: `The fraction of true positive instances among all true positive and flase positive instances combined. P@25 or Precision at 25 corresponds to the number of relevant results among the top 25 retrieved instances.`,
        link: "https://kumo.ai/docs/evaluation-metrics#link-prediction",
        type: HighlightTypes.VARIABLE_NAME
    },
    "precision@100": {
        title: "precision@100",
        subtitle: "",
        hint: `The fraction of true positive instances among all true positive and flase positive instances combined. P@100 or Precision at 100 corresponds to the number of relevant results among the top 100 retrieved instances.`,
        link: "https://kumo.ai/docs/evaluation-metrics#link-prediction",
        type: HighlightTypes.VARIABLE_NAME
    },
    "precision@10": {
        title: "precision@10",
        subtitle: "",
        hint: `The fraction of true positive instances among all true positive and flase positive instances combined. P@10 or Precision at 10 corresponds to the number of relevant results among the top 10 retrieved instances.`,
        link: "https://kumo.ai/docs/evaluation-metrics#link-prediction",
        type: HighlightTypes.VARIABLE_NAME
    },
    "precision@1": {
        title: "precision@1",
        subtitle: "",
        hint: `The fraction of true positive instances among all true positive and flase positive instances combined. P@1 or Precision at 1 corresponds to the number of relevant results among the top 1 retrieved instance.`,
        link: "https://kumo.ai/docs/evaluation-metrics#link-prediction",
        type: HighlightTypes.VARIABLE_NAME
    },
    "precision": {
        title: "precision",
        subtitle: "",
        hint: `The fraction of true positive instances among all true positive and flase positive instances combined.`,
        link: "https://kumo.ai/docs/evaluation-metrics#link-prediction",
        type: HighlightTypes.VARIABLE_NAME
    },
    "ndcg": {
        title: "Normalized Discounted Cumulative Gain (NDCG)",
        subtitle: "",
        hint: `Normalized Discounted Cumulative Gain (NDCG) is calculated by:
        <img src="https://files.readme.io/ffa14c4-image.png" alt="NDCG formula">
        where reli is the graded relevance of the result at position i.`,
        link: "https://kumo.ai/docs/link-prediction",
        type: HighlightTypes.VARIABLE_NAME
    },
    "loss": {
        title: "loss",
        subtitle: "Training Loss ",
        hint: "The loss type to use during model optimization depending on the task type.",
        link: "https://kumo.ai/docs/reference/loss",
        type: HighlightTypes.KEYWORD
    },
    "mrr": {
        title: "mrr",
        subtitle: "Mean Reciprocal Rank (MRR)",
        hint: `Mean Reciprocal Rank (MRR) is calculated by:
        <img src="https://files.readme.io/b649ade-image.png" alt="MRR formula">

        where rank_i refers to the rank position of the first relevant item for the i-th query.`,
        link: "https://kumo.ai/docs/link-prediction",
        type: HighlightTypes.VARIABLE_NAME
    },
    "mae": {
        title: "mae",
        subtitle: "Mean Average Error",
        hint: `Mean Absolute Error (MAE) is calculated by:
        <img src="https://files.readme.io/c5a2666-image.png" alt="MAE formula">

        where yi is the ith label and ŷi is the ith predicted value.`,
        link: "https://kumo.ai/docs/regression",
        type: HighlightTypes.VARIABLE_NAME
    },
    "mape": {
        title: "mape",
        subtitle: `Mean Absolute Percentage Error`,
        hint: "",
        link: "",
        type: HighlightTypes.VARIABLE_NAME
    },
    "smape": {
        title: "smape",
        subtitle: `Symmetric Mean Absolute Percentage Error`,
        hint: `Symmetric Mean Absolute Percentage Error (SMAPE) is calculated by:
        <img src="https://files.readme.io/892f5e8-image.png" alt="SMAPE formula">

        SMAPE normalizes for the scale of the labels and also for the scale of the predictions. Similar to RMSE, this is especially useful if a label is 0 or near 0.`,
        link: "https://kumo.ai/docs/regression",
        type: HighlightTypes.VARIABLE_NAME
    },
    "mse": {
        title: "mse",
        subtitle: "Mean Squared Error",
        hint: `Mean Squared Error (MSE) is calculated by:
        <img src="https://files.readme.io/f8a42ca-image.png" alt="MSE formula">

        MSE focuses more on "how wrong" the prediction is. A prediction that is very far from the label will contribute more to increasing the MSE than the MAE.`,
        link: "https://kumo.ai/docs/regression",
        type: HighlightTypes.VARIABLE_NAME
    },
    "rmse": {
        title: "rmse",
        subtitle: `Root Mean Squared Error`,
        hint: `Root Mean Square Error (RMSE) is calculated by:
        <img src="https://files.readme.io/caac1f7-image.png" alt="RMSE formula">

        This score is in the same scale as the target and can be used even if a label is 0 or near 0.`,
        link: "https://kumo.ai/docs/regression",
        type: HighlightTypes.VARIABLE_NAME
    },
    "map@100": {
        title: "map@100",
        subtitle: "Mean Average Precision @ 100",
        hint: `MAP is the average of all Average Precision (AP) metrics across all classes in a model. MAP@100 is assessed by considering only its top 100 queries.`,
        link: "https://kumo.ai/docs/link-prediction",
        type: HighlightTypes.VARIABLE_NAME
    },
    "map@10": {
        title: "map@10",
        subtitle: "Mean Average Precision @ 10",
        hint: `MAP is the average of all Average Precision (AP) metrics across all classes in a model. MAP@10 is assessed by considering only its top 10 queries.`,
        link: "https://kumo.ai/docs/link-prediction",
        type: HighlightTypes.VARIABLE_NAME
    },
    "map@25": {
        title: "map@25",
        subtitle: "Mean Average Precision @ 25",
        hint: `MAP is the average of all Average Precision (AP) metrics across all classes in a model. MAP@25 is assessed by considering only its top 25 queries.`,
        link: "https://kumo.ai/docs/link-prediction",
        type: HighlightTypes.VARIABLE_NAME
    },
    "map@1": {
        title: "map@1",
        subtitle: "Mean Average Precision @ 1",
        hint: `MAP is the average of all Average Precision (AP) metrics across all classes in a model. MAP@1 is assessed by considering only its top 1 query.`,
        link: "https://kumo.ai/docs/link-prediction",
        type: HighlightTypes.VARIABLE_NAME
    },
    "map": {
        title: "map",
        subtitle: "Mean Average Precision",
        hint: "MAP is the average of all Average Precision (AP) metrics across all classes in a model.",
        link: "https://kumo.ai/docs/link-prediction",
        type: HighlightTypes.VARIABLE_NAME
    },
    "f1@100": {
        title: "f1@100",
        subtitle: "F-score @ 100",
        hint: `F-score -- Harmonic mean of the precision and recall among the top 100 retrieved instances.`,
        link: "https://kumo.ai/docs/link-prediction",
        type: HighlightTypes.VARIABLE_NAME
    },
    "f1@25": {
        title: "f1@25",
        subtitle: "F-score @ 25",
        hint: `F-score -- Harmonic mean of the precision and recall among the top 25 retrieved instances.`,
        link: "https://kumo.ai/docs/link-prediction",
        type: HighlightTypes.VARIABLE_NAME
    },
    "f1@10": {
        title: "f1@10",
        subtitle: "F-score @ 10",
        hint: `F-score -- Harmonic mean of the precision and recall among the top 10 retrieved instances.`,
        link: "https://kumo.ai/docs/link-prediction",
        type: HighlightTypes.VARIABLE_NAME
    },
    "f1@1": {
        title: "f1@1",
        subtitle: "F-score @ 1",
        hint: `F-score -- Harmonic mean of the precision and recall among the top 1 retrieved instance.`,
        link: "https://kumo.ai/docs/link-prediction",
        type: HighlightTypes.VARIABLE_NAME
    },
    "f1": {
        title: "f1",
        subtitle: "F-score",
        hint: `F-score -- Harmonic mean of the precision and recall.`,
        link: "https://kumo.ai/docs/link-prediction",
        type: HighlightTypes.VARIABLE_NAME
    },
    "num_experiments": {
        title: "num_experiments",
        subtitle: "num_experiments: &ltinteger&gt (Optional)",
        hint: `The number of experiments to run during AutoML.

        num_experiments must be > 0.`,
        link: "https://kumo.ai/docs/num_experiments",
        type: HighlightTypes.KEYWORD
    },
    "forecast_length": {
        title: "forecast_length",
        subtitle: "forecast_length: &ltinteger&gt (Optional)",
        hint: `Turns a node regression problem into a forecasting problem. The given target is forecasted for forecast_length steps into the future. This only applies to temporal queries (queries that include a temporal aggregation such as SUM(TRANSACTIONS.AMOUNT, 0, 2, days)). The unit of the forecasting step is the same as the unit in the aggregation.`,
        link: "https://kumo.ai/docs/forecast_length",
        type: HighlightTypes.KEYWORD
    },
    "lag_timesteps": {
        title: "lag_timesteps",
        subtitle: "lag_timesteps: &ltinteger&gt (Optional)",
        hint: `For forecasting problems, leverage the auto-regressive labels as inputs. This parameter controls the number of previous values should be considered as auto-regressive labels.

        The unit of the forecasting step is the same as the unit in the aggregation.`,
        link: "https://kumo.ai/docs/lag_timesteps",
        type: HighlightTypes.KEYWORD
    },
    "year_over_year": {
        title: "year_over_year",
        subtitle: "year_over_year: (false (default) | true) (Optional)",
        hint: `For forecasting problems, integrate Year-Over-Year features as inputs to give more attention to the dat from the previous year when making a prediction. Helps to capture seasonal trends and holiday behavior.`,
        link: "https://kumo.ai/docs/year_over_year",
        type: HighlightTypes.KEYWORD
    },
    "timeframe_step": {
        title: "timeframe_step",
        subtitle: "timeframe_step: &ltinteger&gt (Optional)",
        hint: `Defines the step size of generating time intervals for training table generation. This only applies to temporal queries (queries that include a temporal aggregation such as SUM(TRANSACTIONS.AMOUNT, 0, 2, days)). The time unit of this step is the same as the time unit in the aggregation.

        timeframe_step must be < full time range - target aggregation.`,
        link: "https://kumo.ai/docs/timeframe_step",
        type: HighlightTypes.KEYWORD
    },
    "train_end_offset": {
        title: "train_end_offset",
        subtitle: "train_end_offset: &ltinteger&gt (Optional)",
        hint: `Defines the numerical offset from the most recent entry to not use to generate training data labels. Regardless of this value, all data is used as an input to the model, but this value can help limit what labels are generated.

        train_end_offset must be > 0.

        If used at the same time as train_start_offset, train_end_offset should be strictly smaller.`,
        link: "https://kumo.ai/docs/train_end_offset",
        type: HighlightTypes.KEYWORD
    },
    "train_start_offset": {
        title: "train_start_offset",
        subtitle: "train_start_offset: &ltinteger&gt (Optional)",
        hint: `Defines the numerical offset from the most recent entry to use to generate training data labels. Regardless of this value, all data is used as an input to the model, but this value can help limit what labels are generated.

        train_start_offset must be > 0.`,
        link: "https://kumo.ai/docs/train_start_offset",
        type: HighlightTypes.KEYWORD
    },
    "start_time": {
        title: "start_time",
        subtitle: "start_time: &ltstring&gt (Optional)",
        hint: `Defines the actual timestamp or the numerical offset from the most recent entry to use to generate training data labels. Regardless of this value, all data is used as an input to the model, but this value can help limit what labels are generated.
        start_time must be <= 0.`,
        link: "https://kumo.ai/docs/reference/start_time",
        type: HighlightTypes.KEYWORD
    },
    "end_time": {
        title: "end_time",
        subtitle: "end_time: &ltinteger&gt (Optional)",
        hint: `Defines the actual timestamp or the numerical offset from the most recent entry to not use to generate training data labels. Regardless of this value, all data is used as an input to the model, but this value can help limit what labels are generated.
        end_time must be <= 0.
        If used at the same time as start_time, end_time should be strictly larger.`,
        link: "https://kumo.ai/docs/reference/end_time",
        type: HighlightTypes.KEYWORD
    },
    "prediction_time_encodings": {
        title: "prediction_time_encodings",
        subtitle: "prediction_time_encodings",
        hint: `Adds the prediction anchor timestamp to the model input.`,
        link: "",
        type: HighlightTypes.KEYWORD
    },
    "handle_new_entities": {
        title: "handle_new_entities",
        subtitle: "year_over_year: (false | true (default)) (Optional)",
        hint: `Whether to make forecasting models transductive by learning entity-specific heads. This can improve performance by avoiding mean-collapse in case entities stay static over time, but will decrease performance on new entities arising during batch prediction time.`,
        link: "https://kumo.ai/docs/handle_new_entities",
        type: HighlightTypes.KEYWORD
    },
    "training_table_generation": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.TYPE_NAME
    },
    "training_job": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.TYPE_NAME
    },
    "column_processing": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.TYPE_NAME
    },
    "neighbor_sampling": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.TYPE_NAME
    },
    "optimization": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.TYPE_NAME
    },
    "model_architecture": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.TYPE_NAME
    },
    "0": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.NUMBER
    },
    "1": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.NUMBER
    },
    "2": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.NUMBER
    },
    "3": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.NUMBER
    },
    "4": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.NUMBER
    },
    "5": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.NUMBER
    },
    "6": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.NUMBER
    },
    "7": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.NUMBER
    },
    "8": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.NUMBER
    },
    "9": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.NUMBER
    },
    "e-": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.NUMBER
    },
    "-": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.NUMBER
    },
    ".": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.NUMBER
    }
};

export const yamlDict: HoverHintsType = {
    "min_steps_per_epoch": {
        title: "min_steps_per_epoch",
        subtitle: "min_steps_per_epoch: &ltinteger&gt (Optional)",
        hint: "The minimum number of steps to train in a single epoch. Useful to tune for small-scale datasets.",
        link: "",
        type: HighlightTypes.KEYWORD
    },
    "min_test_steps": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.KEYWORD
    },
    "min_val_steps": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.KEYWORD
    },
    "auroc_per_label": {
        title: "auroc_per_label",
        subtitle: `Area Under the Receiver Operating Characteristic curve per Label`,
        hint: `The AUROC stands for the area under the receiver operating characteristic curve, which is a curve plotting sensitivity against specificity under different classification thresholds.`,
        link: "https://kumo.ai/docs/evaluation#multilabel-classification",
        type: HighlightTypes.VARIABLE_NAME
    },
    "auroc_macro": {
        title: "auroc_macro",
        subtitle:
            "Unweighted mean of all the per-class AUROC (Area Under the Receiver Operating Characteristic curve) scores.",
        hint: `The AUROC stands for the area under the receiver operating characteristic curve, which is a curve plotting sensitivity against specificity under different classification thresholds.`,
        link: "https://kumo.ai/docs/evaluation#multilabel-classification",
        type: HighlightTypes.VARIABLE_NAME
    },
    "auroc_micro": {
        title: "auroc_micro",
        subtitle:
            "Global average of Area Under the Receiver Operating Characteristic curve (AUROC) scores.",
        hint: `The AUROC stands for the area under the receiver operating characteristic curve, which is a curve plotting sensitivity against specificity under different classification thresholds.`,
        link: "https://kumo.ai/docs/evaluation#multilabel-classification",
        type: HighlightTypes.VARIABLE_NAME
    },
    "auroc": {
        title: "auroc",
        subtitle: `Area Under the Receiver Operating Characteristic curve`,
        hint: `The AUROC stands for the area under the receiver operating characteristic curve, which is a curve plotting sensitivity against specificity under different classification thresholds.`,
        link: "https://kumo.ai/docs/classification#auroc",
        type: HighlightTypes.VARIABLE_NAME
    },
    "acc": {
        title: "Accuracy",
        subtitle: "",
        hint: `The accuracy tells you the overall percentage of correct predictions. The accuracy score is given by the total number of correct predictions over the number of samples.`,
        link: "https://kumo.ai/docs/classification#accuracy",
        type: HighlightTypes.VARIABLE_NAME
    },
    "auprc_macro": {
        title: "auprc_macro",
        subtitle: `Unweighted mean of all the per-class Area Under the Precision-Recall Curve (AUPRC) scores.`,
        hint: `The AUPRC stands for the area under the precision-recall curve, which is a curve plotting precision against recall under different classification thresholds.`,
        link: "https://kumo.ai/docs/evaluation#multilabel-classification",
        type: HighlightTypes.VARIABLE_NAME
    },
    "auprc_micro": {
        title: "auprc_micro",
        subtitle: `Global average of Area Under the Precision-Recall Curve (AUPRC) scores.`,
        hint: `The AUPRC stands for the area under the precision-recall curve, which is a curve plotting precision against recall under different classification thresholds.`,
        link: "https://kumo.ai/docs/evaluation#multilabel-classification",
        type: HighlightTypes.VARIABLE_NAME
    },
    "auprc": {
        title: "auprc",
        subtitle: `Area Under the Precision-Recall Curve`,
        hint: `The AUPRC stands for the area under the precision-recall curve, which is a curve plotting precision against recall under different classification thresholds.`,
        link: "https://kumo.ai/docs/classification#auprc",
        type: HighlightTypes.VARIABLE_NAME
    },
    "ap_macro": {
        title: "ap_macro",
        subtitle: "",
        hint: `Unweighted mean of all the per-class Average Precision (AP) scores.`,
        link: "https://kumo.ai/docs/evaluation#multilabel-classification",
        type: HighlightTypes.VARIABLE_NAME
    },
    "ap_micro": {
        title: "ap_micro",
        subtitle: "",
        hint: `Global average of Average Precision (AP) scores.`,
        link: "https://kumo.ai/docs/evaluation#multilabel-classification",
        type: HighlightTypes.VARIABLE_NAME
    },
    "ap_per_label": {
        title: "ap_per_label",
        subtitle: "",
        hint: `Average Precision per Label`,
        link: "https://kumo.ai/docs/evaluation#multilabel-classification",
        type: HighlightTypes.VARIABLE_NAME
    },
    "ap": {
        title: "ap",
        subtitle: "Average Precision",
        hint: "",
        link: "https://kumo.ai/docs/evaluation#multilabel-classification",
        type: HighlightTypes.VARIABLE_NAME
    },
    "recall@K": {
        title: "recall@K",
        subtitle: "",
        hint: `The fraction of true positive instances among all true positive and false negative instances combined. <code>Recall@K</code> or Recall at K corresponds to the share of relevant results among the top K retrieved instances.`,
        link: "https://kumo.ai/docs/evaluation-metrics#link-prediction",
        type: HighlightTypes.VARIABLE_NAME
    },
    "recall": {
        title: "recall",
        subtitle: "",
        hint: `The fraction of true positive instances among all true positive and flase negative instances combined.`,
        link: "https://kumo.ai/docs/evaluation-metrics#link-prediction",
        type: HighlightTypes.VARIABLE_NAME
    },
    "precision@K": {
        title: "precision@K",
        subtitle: "",
        hint: `The fraction of true positive instances among all true positive and false positive instances combined. <code>precision@K</code> or Precision at K corresponds to the number of relevant results among the top K retrieved instances.`,
        link: "https://kumo.ai/docs/evaluation-metrics#link-prediction",
        type: HighlightTypes.VARIABLE_NAME
    },
    "precision": {
        title: "precision",
        subtitle: "",
        hint: `The fraction of true positive instances among all true positive and flase positive instances combined.`,
        link: "https://kumo.ai/docs/evaluation-metrics#link-prediction",
        type: HighlightTypes.VARIABLE_NAME
    },
    "ndcg": {
        title: "Normalized Discounted Cumulative Gain (NDCG)",
        subtitle: "",
        hint: `Normalized Discounted Cumulative Gain (NDCG) is calculated by:
        <img src="https://files.readme.io/ffa14c4-image.png" alt="NDCG formula">
        where reli is the graded relevance of the result at position i.`,
        link: "https://kumo.ai/docs/link-prediction",
        type: HighlightTypes.VARIABLE_NAME
    },
    "loss": {
        title: "loss",
        subtitle: "Training Loss ",
        hint: "During model training KUMO will provide LOSS curves for every experiment. Similar to tensorboard, these loss curves help you check for underfitting or overfitting.",
        link: "https://kumo.ai/docs/reference/loss/",
        type: HighlightTypes.KEYWORD
    },
    "mrr": {
        title: "mrr",
        subtitle: "Mean Reciprocal Rank (MRR)",
        hint: `Mean Reciprocal Rank (MRR) is calculated by:
        <img src="https://files.readme.io/b649ade-image.png" alt="MRR formula">

        where rank_i refers to the rank position of the first relevant item for the i-th query.`,
        link: "https://kumo.ai/docs/link-prediction",
        type: HighlightTypes.VARIABLE_NAME
    },
    "mae": {
        title: "mae",
        subtitle: "Mean Average Error",
        hint: `Mean Absolute Error (MAE) is calculated by:
        <img src="https://files.readme.io/c5a2666-image.png" alt="MAE formula">

        where yi is the ith label and ŷi is the ith predicted value.`,
        link: "https://kumo.ai/docs/regression",
        type: HighlightTypes.VARIABLE_NAME
    },
    "mape": {
        title: "mape",
        subtitle: `Mean Absolute Percentage Error`,
        hint: "",
        link: "",
        type: HighlightTypes.VARIABLE_NAME
    },
    "smape": {
        title: "smape",
        subtitle: `Symmetric Mean Absolute Percentage Error`,
        hint: `Symmetric Mean Absolute Percentage Error (SMAPE) is calculated by:
        <img src="https://files.readme.io/892f5e8-image.png" alt="SMAPE formula">

        SMAPE normalizes for the scale of the labels and also for the scale of the predictions. Similar to RMSE, this is especially useful if a label is 0 or near 0.`,
        link: "https://kumo.ai/docs/regression",
        type: HighlightTypes.VARIABLE_NAME
    },
    "mse": {
        title: "mse",
        subtitle: "Mean Squared Error",
        hint: `Mean Squared Error (MSE) is calculated by:
        <img src="https://files.readme.io/f8a42ca-image.png" alt="MSE formula">

        MSE focuses more on "how wrong" the prediction is. A prediction that is very far from the label will contribute more to increasing the MSE than the MAE.`,
        link: "https://kumo.ai/docs/regression",
        type: HighlightTypes.VARIABLE_NAME
    },
    "rmse": {
        title: "rmse",
        subtitle: `Root Mean Squared Error`,
        hint: `Root Mean Square Error (RMSE) is calculated by:
        <img src="https://files.readme.io/caac1f7-image.png" alt="RMSE formula">

        This score is in the same scale as the target and can be used even if a label is 0 or near 0.`,
        link: "https://kumo.ai/docs/regression",
        type: HighlightTypes.VARIABLE_NAME
    },
    "map@K": {
        title: "map@K",
        subtitle: "Mean Average Precision @ K",
        hint: `MAP is the average of all Average Precision (AP) metrics across all classes in a model. <code>MAP@K</code> is assessed by considering only its top K queries.`,
        link: "https://kumo.ai/docs/link-prediction",
        type: HighlightTypes.VARIABLE_NAME
    },
    "map": {
        title: "map",
        subtitle: "Mean Average Precision",
        hint: "MAP is the average of all Average Precision (AP) metrics across all classes in a model.",
        link: "https://kumo.ai/docs/link-prediction",
        type: HighlightTypes.VARIABLE_NAME
    },
    "f1@K": {
        title: "f1@K",
        subtitle: "F-score @ K",
        hint: `F-score -- Harmonic mean of the precision and recall among the top K retrieved instances.`,
        link: "https://kumo.ai/docs/link-prediction",
        type: HighlightTypes.VARIABLE_NAME
    },
    "f1": {
        title: "f1",
        subtitle: "F-score",
        hint: `F-score -- Harmonic mean of the precision and recall.`,
        link: "https://kumo.ai/docs/link-prediction",
        type: HighlightTypes.VARIABLE_NAME
    },
    "prediction_time_encodings": {
        title: "prediction_time_encodings",
        subtitle: "prediction_time_encodings",
        hint: `Adds the prediction anchor timestamp to the model input.`,
        link: "",
        type: HighlightTypes.KEYWORD
    },
    "training_table_generation": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.TYPE_NAME
    },
    "training_job": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.TYPE_NAME
    },
    "column_processing": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.TYPE_NAME
    },
    "neighbor_sampling": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.TYPE_NAME
    },
    "optimization": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.TYPE_NAME
    },
    "model_architecture": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.TYPE_NAME
    },
    "0": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.NUMBER
    },
    "1": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.NUMBER
    },
    "2": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.NUMBER
    },
    "3": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.NUMBER
    },
    "4": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.NUMBER
    },
    "5": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.NUMBER
    },
    "6": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.NUMBER
    },
    "7": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.NUMBER
    },
    "8": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.NUMBER
    },
    "9": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.NUMBER
    },
    "e-": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.NUMBER
    },
    "-": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.NUMBER
    },
    ".": {
        title: "",
        subtitle: "",
        hint: "",
        link: "",
        type: HighlightTypes.NUMBER
    }
};
