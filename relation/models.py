import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers import BertModel, BertPreTrainedModel
from transformers import AlbertModel, AlbertPreTrainedModel

import torch.nn.functional as F

def batched_index_select(target, indices,flattened_indices=None):
    if flattened_indices is None:
        # Shape: (batch_size * d_1 * ... * d_n)
        flattened_indices = flatten_and_batch_shift_indices(indices, target.size(1))
    # Shape: (batch_size * sequence_length, embedding_size)
    flattened_target = target.view(-1, target.size(-1))
    # Shape: (batch_size * d_1 * ... * d_n, embedding_size)
    flattened_selected = flattened_target.index_select(0,flattened_indices)
    selected_shape = list(indices.size()) + [target.size(-1)]
    # Shape: (batch_size, d_1, ..., d_n, embedding_size)
    selected_targets = flattened_selected.view(*selected_shape)
    return selected_targets
def flatten_and_batch_shift_indices(indices: torch.Tensor, sequence_length: int):
    offsets = get_range_vector(indices.size(0), get_device_of(indices)) * sequence_length
    for _ in range(len(indices.size()) - 1):
        offsets = offsets.unsqueeze(1)

    # Shape: (batch_size, d_1, ..., d_n)
    offset_indices = indices + offsets

    # Shape: (batch_size * d_1 * ... * d_n)
    offset_indices = offset_indices.view(-1)
    return offset_indices
def get_device_of(tensor: torch.Tensor):
    """
    Returns the device of the tensor.
    """
    if not tensor.is_cuda:
        return -1
    else:
        return tensor.get_device()
def get_range_vector(size: int, device: int) -> torch.Tensor:
    """
    Returns a range vector with the desired size, starting at 0. The CUDA implementation
    is meant to avoid copy data from CPU to GPU.
    """
    if device > -1:
        return torch.cuda.LongTensor(size, device=device).fill_(1).cumsum(0) - 1
    else:
        return torch.arange(0, size, dtype=torch.long)

class FeedForward(nn.Module):
    def __init__(self, input_dim, num_layers, hidden_dims,activations,dropout):
        super(FeedForward, self).__init__()
        if not isinstance(hidden_dims, list):
            hidden_dims = [hidden_dims] * num_layers  # type: ignore
        if not isinstance(activations, list):
            activations = [activations] * num_layers  # type: ignore
        if not isinstance(dropout, list):
            dropout = [dropout] * num_layers  # type: ignore
        self._activations = torch.nn.ModuleList(activations)
        input_dims = [input_dim] + hidden_dims[:-1]
        linear_layers = []
        for layer_input_dim, layer_output_dim in zip(input_dims, hidden_dims):
            linear_layers.append(torch.nn.Linear(layer_input_dim, layer_output_dim))
        self._linear_layers = torch.nn.ModuleList(linear_layers)
        dropout_layers = [torch.nn.Dropout(p=value) for value in dropout]
        self._dropout = torch.nn.ModuleList(dropout_layers)
        self._output_dim = hidden_dims[-1]
        self.input_dim = input_dim

    def forward(self, inputs):# connect the layers to net
        output = inputs
        for layer, activation, dropout in zip(
            self._linear_layers, self._activations, self._dropout
        ):
            output = dropout(activation(layer(output)))
        return output

BertLayerNorm = torch.nn.LayerNorm
class BertForRelation(BertPreTrainedModel):
    def __init__(self, config, num_rel_labels):
        super(BertForRelation, self).__init__(config)
        self.num_labels = num_rel_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm = BertLayerNorm(config.hidden_size * 2)
        self.classifier = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, sub_idx=None, obj_idx=None, input_position=None):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, output_hidden_states=False, output_attentions=False, position_ids=input_position)
        sequence_output = outputs[0]
        sub_output = torch.cat([a[i].unsqueeze(0) for a, i in zip(sequence_output, sub_idx)])
        obj_output = torch.cat([a[i].unsqueeze(0) for a, i in zip(sequence_output, obj_idx)])
        rep = torch.cat((sub_output, obj_output), dim=1)
        rep = self.layer_norm(rep)
        rep = self.dropout(rep)
        logits = self.classifier(rep)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits

class AlbertForRelation(AlbertPreTrainedModel):
    def __init__(self, config, num_rel_labels):
        super(AlbertForRelation, self).__init__(config)
        self.num_labels = num_rel_labels
        self.albert = AlbertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm = BertLayerNorm(config.hidden_size * 2)
        self.classifier = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, sub_idx=None, obj_idx=None):
        outputs = self.albert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, output_hidden_states=False, output_attentions=False)
        sequence_output = outputs[0]
        sub_output = torch.cat([a[i].unsqueeze(0) for a, i in zip(sequence_output, sub_idx)])
        obj_output = torch.cat([a[i].unsqueeze(0) for a, i in zip(sequence_output, obj_idx)])
        rep = torch.cat((sub_output, obj_output), dim=1)
        rep = self.layer_norm(rep)
        rep = self.dropout(rep)
        logits = self.classifier(rep)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits

class BertForRelationApprox(BertPreTrainedModel):
    def __init__(self, config, num_rel_labels):
        super(BertForRelationApprox, self).__init__(config)
        self.num_labels = num_rel_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm = BertLayerNorm(config.hidden_size * 2)
        self.classifier = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, sub_obj_ids=None, sub_obj_masks=None, input_position=None):
        """
        attention_mask: [batch_size, from_seq_length, to_seq_length]
        """
        batch_size = input_ids.size(0)
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, output_hidden_states=False, output_attentions=False, position_ids=input_position)
        sequence_output = outputs[0]

        sub_ids = sub_obj_ids[:, :, 0].view(batch_size, -1)
        sub_embeddings = batched_index_select(sequence_output, sub_ids)
        obj_ids = sub_obj_ids[:, :, 1].view(batch_size, -1)
        obj_embeddings = batched_index_select(sequence_output, obj_ids)
        rep = torch.cat((sub_embeddings, obj_embeddings), dim=-1)
        rep = self.layer_norm(rep)
        rep = self.dropout(rep)
        logits = self.classifier(rep)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            active_loss = (sub_obj_masks.view(-1) == 1)
            active_logits = logits.view(-1, logits.shape[-1])
            active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
            loss = loss_fct(active_logits, active_labels)
            return loss
        else:
            return logits

class AlbertForRelationApprox(BertPreTrainedModel):
    """
    ALBERT approximation model is not supported by the current implementation, 
    as Huggingface's Transformers ALBERT doesn't support an attention mask with a shape of [batch_size, from_seq_length, to_seq_length]."
    """
    def __init__(self, config, num_rel_labels):
        super(AlbertForRelationApprox, self).__init__(config)
        self.num_labels = num_rel_labels
        self.albert = AlbertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm = BertLayerNorm(config.hidden_size * 2)
        self.classifier = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, sub_obj_ids=None, sub_obj_masks=None, input_position=None):
        batch_size = input_ids.size(0)
        outputs = self.albert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, output_hidden_states=False, output_attentions=False, position_ids=input_position)
        sequence_output = outputs[0]

        sub_ids = sub_obj_ids[:, :, 0].view(batch_size, -1)
        sub_embeddings = batched_index_select(sequence_output, sub_ids)
        obj_ids = sub_obj_ids[:, :, 1].view(batch_size, -1)
        obj_embeddings = batched_index_select(sequence_output, obj_ids)
        rep = torch.cat((sub_embeddings, obj_embeddings), dim=-1)
        rep = self.layer_norm(rep)
        rep = self.dropout(rep)
        logits = self.classifier(rep)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            active_loss = (sub_obj_masks.view(-1) == 1)
            active_logits = logits.view(-1, logits.shape[-1])
            active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
            loss = loss_fct(active_logits, active_labels)
            return loss
        else:
            return logits
