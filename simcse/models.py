import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import transformers
from transformers import RobertaTokenizer
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaLMHead, RobertaConfig
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertLMPredictionHead, BertConfig
from transformers.activations import gelu
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions

class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[1]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError


def cl_init(cls, config):
    """
    Contrastive learning class init function.
    """
    cls.pooler_type = cls.model_args.pooler_type
    cls.pooler = Pooler(cls.model_args.pooler_type)
    if cls.model_args.pooler_type == "cls":
        cls.mlp = MLPLayer(config)
    cls.sim = Similarity(temp=cls.model_args.temp)
    cls.init_weights()

####lz changed here the inputs
def cl_forward(cls,
    encoder, is_mask=False,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    output_attentions=None,
    return_dict=None,
    mlm_input_ids=None, 
    # labels=None,mlm_labels=None,output_hidden_states=None,
):
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    ori_input_ids = input_ids
    batch_size = input_ids.size(0)
    num_sent = input_ids.size(1)

    # input_ids = batch_size x triplet(sent, pos, neg)

    mlm_outputs = None
    
    # (batch_size * 3, sentence_length)
    input_ids = input_ids.view((-1, input_ids.size(-1)))
    attention_mask = attention_mask.view((-1, attention_mask.size(-1)))
    if token_type_ids is not None:
        token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1)))

    # Get raw embeddings
    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )

    # print(cls.model_args.pooler_type)
    # exit()

    # Pooling
    pooler_output = cls.pooler(attention_mask, outputs)
    pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1))) # (bs, num_sent, hidden)

    # If using "cls", we add an extra MLP layer
    # (same as BERT's original implementation) over the representation.
    if cls.pooler_type == "cls" and is_mask == False:
        pooler_output = cls.mlp(pooler_output)

    #batch_size * 3 * hidden
    return pooler_output, outputs


####added both masked pretrain and new model
def cl_masked_forward(cls,
    encoder, encoder_masked,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    mlm_input_ids=None,
    mlm_labels=None,
):
    
    mlm_outputs = None
    outputs_rep, outputs_encoder = cl_forward(cls, encoder, False, input_ids, attention_mask, token_type_ids,position_ids, head_mask, inputs_embeds, output_attentions, return_dict, mlm_input_ids)
    
    outputs_rep_masked, _ = cl_forward(cls, encoder_masked, True, input_ids, attention_mask,
                                                       token_type_ids, position_ids,
                                                       head_mask, inputs_embeds, output_attentions, return_dict,
                                                       mlm_input_ids)
    
    attention_mask = attention_mask.view((-1, attention_mask.size(-1)))
    if token_type_ids is not None:
        token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1)))
    if mlm_input_ids is not None:
        mlm_input_ids = mlm_input_ids.view((-1, mlm_input_ids.size(-1)))
        mlm_outputs = encoder(
            mlm_input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            return_dict=True,
        )

    # Separate representation
    # print(outputs_rep.size(), outputs_rep_masked.size()) (bs, num_sent, size_hidden)
    
    ####changed here, need to compare both pos (second col) and hard neg (third col)
    z1, z2, z_negative = outputs_rep[:,0], outputs_rep[:,1], outputs_rep[:,2]
    z1_mask, z2_mask, z_negative_mask = outputs_rep_masked[:,0], outputs_rep_masked[:,1], outputs_rep_masked[:,2]

    # Gather all embeddings if using distributed training
    if dist.is_initialized() and cls.training:
        # print("yes")
        # Dummy vectors for allgather
        z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
        z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
        z_negative_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
        z1_list_mask = [torch.zeros_like(z1_mask) for _ in range(dist.get_world_size())]
        z2_list_mask = [torch.zeros_like(z2_mask) for _ in range(dist.get_world_size())]
        z_negative_list_mask = [torch.zeros_like(z_negative_mask) for _ in range(dist.get_world_size())]
        # Allgather
        dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
        dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())
        dist.all_gather(tensor_list=z_negative_list, tensor=z_negative.contiguous())
        dist.all_gather(tensor_list=z1_list_mask, tensor=z1_mask.contiguous())
        dist.all_gather(tensor_list=z2_list_mask, tensor=z2_mask.contiguous())
        dist.all_gather(tensor_list=z_negative_list_mask, tensor=z_negative_mask.contiguous())

        # Since allgather results do not have gradients, we replace the
        # current process's corresponding embeddings with original tensors
        z1_list[dist.get_rank()] = z1
        z2_list[dist.get_rank()] = z2
        z_negative_list[dist.get_rank()] = z_negative

        z1_list_mask[dist.get_rank()] = z1_mask
        z2_list_mask[dist.get_rank()] = z2_mask
        z_negative_list_mask[dist.get_rank()] = z_negative_mask

        # Get full batch embeddings: (bs x N, hidden)
        z1 = torch.cat(z1_list, 0)
        z2 = torch.cat(z2_list, 0)
        z_negative = torch.cat(z_negative_list, 0)
        z1_mask = torch.cat(z1_list_mask, 0)
        z2_mask = torch.cat(z2_list_mask, 0)
        z_negative_mask = torch.cat(z_negative_list_mask, 0)

    # z1, z2 = outputs_rep[:,0], outputs_rep[:,1]
    # z1_mask, z2_mask = outputs_rep_masked[:,0], outputs_rep_masked[:,1]
    # print(z1.unsqueeze(1).size(), z2.unsqueeze(0).size()) torch.Size([128, 1, 768]), torch.Size([1, 128, 768])
    cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    # cos_sim_12 = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    # cos_sim_13 = cls.sim(z1.unsqueeze(1), z_negative.unsqueeze(0))
    # print(cos_sim.size()) # bs*bs
    cos_sim_mask = cls.sim(z1_mask.unsqueeze(1), z2_mask.unsqueeze(0))
    # cos_sim_mask_12 = cls.sim(z1_mask.unsqueeze(1), z2_mask.unsqueeze(0))
    # cos_sim_mask_13 = cls.sim(z1_mask.unsqueeze(1), z_negative_mask.unsqueeze(0))

    labels = torch.arange(cos_sim.size(0)).long().to(cls.device) # [0, 1, 2, ..., 127]
    loss_fct = nn.CrossEntropyLoss()

    cos_sim_negative = cls.sim(z1.unsqueeze(1), z_negative.unsqueeze(0))
    cos_sim = torch.cat([cos_sim, cos_sim_negative], 1)
    ####changed here, here cos_sim_negative and cos_sim_negative_mask are both from data instead of defined
    cos_sim_negative_mask = cls.sim(z1_mask.unsqueeze(1), z_negative_mask.unsqueeze(0))
    cos_sim_mask = torch.cat([cos_sim_mask, cos_sim_negative_mask], 1)

    
    labels_mask = torch.cat([torch.eye(cos_sim.size(0), device=cos_sim.device)[labels], torch.zeros_like(cos_sim_negative)], -1)
    # print(torch.max(cos_sim_mask))
    # exit()
    weights = torch.where(cos_sim_mask > cls.model_args.phi * 20, 0, 1)
    mask_weights = torch.eye(cos_sim.size(0), device = cos_sim.device) - torch.diag_embed(torch.diag(weights))
    weights = weights + torch.cat([mask_weights, torch.zeros_like(cos_sim_negative)], -1)

    soft_cos_sim = torch.softmax(cos_sim * weights, -1)
    loss = - (labels_mask * torch.log(soft_cos_sim) + (1 - labels_mask) * torch.log(1 - soft_cos_sim))
    loss = torch.mean(loss)
    
    # Calculate loss for MLM
    if mlm_outputs is not None and mlm_labels is not None:
        mlm_labels = mlm_labels.view(-1, mlm_labels.size(-1))
        prediction_scores = cls.lm_head(mlm_outputs.last_hidden_state)
        masked_lm_loss = loss_fct(prediction_scores.view(-1, cls.config.vocab_size), mlm_labels.view(-1))
        loss = loss + cls.model_args.mlm_weight * masked_lm_loss

    if not return_dict:
        output = (cos_sim,) + outputs_encoder[2:]
        return ((loss,) + output) if loss is not None else output
    return SequenceClassifierOutput(
        loss=loss,
        logits=cos_sim,
        hidden_states=outputs_encoder.hidden_states,
        attentions=outputs_encoder.attentions,
    )


def sentemb_forward(
    cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):

    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )

    pooler_output = cls.pooler(attention_mask, outputs)
    if cls.pooler_type == "cls" and not cls.model_args.mlp_only_train:
        pooler_output = cls.mlp(pooler_output)

    if not return_dict:
        return (outputs[0], pooler_output) + outputs[2:]

    return BaseModelOutputWithPoolingAndCrossAttentions(
        pooler_output=pooler_output,
        last_hidden_state=outputs.last_hidden_state,
        hidden_states=outputs.hidden_states,
    )


class BertForCL(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.bert = BertModel(config, add_pooling_layer=False)

        pretrained_config = BertConfig.from_pretrained(self.model_args.pretrained_model_name_or_path)
        self.pretrained_model = BertModel(pretrained_config)

        if self.model_args.do_mlm:
            self.lm_head = BertLMPredictionHead(config)

        cl_init(self, config)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
    ):
        if sent_emb:
            return sentemb_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            ####changed here, added encoder_masked into the new forward
            return cl_masked_forward(self, self.bert, self.pretrained_model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
            )



class RobertaForCL(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.roberta = RobertaModel(config, add_pooling_layer=False)

        pretrained_config = RobertaConfig.from_pretrained(self.model_args.pretrained_model_name_or_path)
        self.pretrained_model = RobertaModel(pretrained_config)

        if self.model_args.do_mlm:
            self.lm_head = RobertaLMHead(config)

        cl_init(self, config)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
    ):
        if sent_emb:
            return sentemb_forward(self, self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_masked_forward(self, self.roberta, self.pretrained_model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
            )
