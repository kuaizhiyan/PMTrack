# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import List,Dict,Optional,Union

import torch
import torch.nn as nn
from torch import Tensor
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmengine.model import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm
from mmdet.utils import OptConfigType, OptMultiConfig
from mmdet.registry import MODELS
from typing import List, Tuple, Union
from mmdet.models.detectors import ConditionalDETR
from mmdet.models.layers import (ConditionalDetrTransformerDecoder,
                      DetrTransformerEncoder, SinePositionalEncoding,)
from mmdet.models.layers import ResLayer,SimplifiedBasicBlock
from mmdet.models.reid import GlobalAveragePooling
from mmcv.cnn import ConvModule
from mmdet.models.necks import ChannelMapper
from mmdet.models.layers.transformer.utils import  coordinate_to_encoding
from mmdet.models.layers.transformer.conditional_detr_layers import MyConditionalDetrTransformerDecoder
from mmdet.models.layers.transformer.detr_layers import DetrTransformerEncoder
# co /home/kzy/project/mmdetection/mmdet/models/layers/transformer/conditional_detr_layers.py
# from mmdet.utils import register_all_modules
# register_all_modules()

class MSPCTransformerDecoder(MyConditionalDetrTransformerDecoder):
    """The decoder part of Part Decoder."""

    def forward(self,
                query: Tensor,
                key: Tensor = None,
                query_pos: Tensor = None,
                key_pos: Tensor = None,
                key_padding_mask: Tensor = None):
        """Forward function of decoder.

        Args:
            query (Tensor): The input query with shape
                (bs, num_queries, dim).
            key (Tensor): The input key with shape (bs, num_keys, dim) If
                `None`, the `query` will be used. Defaults to `None`.
            query_pos (Tensor): The positional encoding for `query`, with the
                same shape as `query`. If not `None`, it will be added to
                `query` before forward function. Defaults to `None`.
            key_pos (Tensor): The positional encoding for `key`, with the
                same shape as `key`. If not `None`, it will be added to
                `key` before forward function. If `None`, and `query_pos`
                has the same shape as `key`, then `query_pos` will be used
                as `key_pos`. Defaults to `None`.
            key_padding_mask (Tensor): ByteTensor with shape (bs, num_keys).
                Defaults to `None`.
        Returns:
            List[Tensor]: forwarded results with shape (num_decoder_layers,
            bs, num_queries, dim) if `return_intermediate` is True, otherwise
            with shape (1, bs, num_queries, dim). References with shape
            (bs, num_queries, 2).
        """
        global_query_pos = (query_pos[:,0,:]).unsqueeze(1)     # fetch out the global query. global_query_pos[bs,1,dims]
        part_query_pos = query_pos[:,1:,:]                     # fetch out part queries. part_query_pos[bs,num_queries-1,dims]
        reference_unsigmoid = self.ref_point_head(      # 2d coord embedding.  query_pos[bs,num_queries,dims]->reference_unsigmoid[bs,num_queries,2]
            part_query_pos)  
        reference = reference_unsigmoid.sigmoid()       # sigmoid. reference[bs, num_queries, 2]
        reference_xy = reference[..., :2] # reference_xy [bs, num_queries, 2]
        intermediate = []
        pos_attn_score_layers = []
        for layer_id, layer in enumerate(self.layers):
            if layer_id == 0:
                pos_transformation = 1
            else:
                pos_transformation = self.query_scale(query)
            # get sine embedding for the query reference
            ref_sine_embed = coordinate_to_encoding(coord_tensor=reference_xy)  # ref_sine_embed:[bs,num_queries,dims] （p_s）# concat global embedding
            ref_sine_embed = torch.concat([global_query_pos,ref_sine_embed],dim=1)
            # apply transformation
            ref_sine_embed = ref_sine_embed * pos_transformation  # ref_sine_embed:[bs,num_queries,dims]·1|[bs,num_queries,dims]=[bs,num_queries,dims]
            query,ca_pos_attention_score = layer(
                query,
                key=key,
                query_pos=query_pos,
                key_pos=key_pos,
                key_padding_mask=key_padding_mask,
                ref_sine_embed=ref_sine_embed,
                is_first=(layer_id == 0))
            if self.return_intermediate:
                intermediate.append(self.post_norm(query))
                pos_attn_score_layers.append(ca_pos_attention_score)

        if self.return_intermediate:
            return torch.stack(intermediate), reference,torch.stack(pos_attn_score_layers)

        query = self.post_norm(query)
        return query.unsqueeze(0), reference,torch.stack(pos_attn_score_layers)

class MSPCTransformer(BaseModule):
    """ Learn part features by part querier from feature maps.
    
    Args:
        num_queries (int): the number of part query.
        
    Return:
        dict: the output hidden states of global queries and part queries.
    """
    
    def __init__(self,
                 embed_dims:int = 256,
                 decoder: OptConfigType = None,
                 positional_encoding: OptConfigType = None,
                 num_queries: int = 16,
                 with_agg:bool = False,
                 with_encoder:bool = False,
                 encoder: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 ) -> None:
        super(MSPCTransformer,self).__init__()
        # process args
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.decoder = decoder      # cfg
        self.with_encoder = with_encoder
        self.encoder = encoder
        self.positional_encoding = positional_encoding  # cfg
        self.num_queries = num_queries
        self.embed_dims = embed_dims
        self.with_agg = with_agg

        self._init_layers()
    
    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding)
        if self.with_encoder:
            self.encoder = DetrTransformerEncoder(**self.encoder)
        else:
            self.encoder = nn.Identity()
            
        self.decoder = MSPCTransformerDecoder(**self.decoder)
            
        self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)

        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            f'embed_dims should be exactly 2 times of num_feats. ' \
            f'Found {self.embed_dims} and {num_feats}.'   
    
    def init_weights(self) -> None:
        """Initialize weights for Transformer and other components."""
        super().init_weights()
        for p in self.decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def pre_transformer(
            self,
            img_feats: Tuple[Tensor],
        ) -> Tuple[Dict, Dict]:
        """Prepare the inputs of the Transformer.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            img_feats (Tuple[Tensor]): Tuple of features output from the neck,
                has shape (bs, c, h, w).
        Returns:
            tuple[dict, dict]: The first dict contains the inputs of encoder
            and the second dict contains the inputs of decoder.

            - encoder_inputs_dict (dict): The keyword args dictionary of
              `self.forward_encoder()`, which includes 'feat', 'feat_mask',
              and 'feat_pos'.
            - decoder_inputs_dict (dict): The keyword args dictionary of
              `self.forward_decoder()`, which includes 'memory_mask',
              and 'memory_pos'.
        """
        if not self.with_agg:
            feat = img_feats[-1]  # NOTE fetch out the last layer
            batch_size, feat_dim, _, _ = feat.shape
            
            # construct binary masks which for the transformer.    
            masks = None
            # [batch_size, embed_dim, h, w]
            pos_embed = self.positional_encoding(masks, input=feat) # feat[bs,dim(2048),h,w] ->pos_emb [bs,embed_dim(256),h,w]
        
            # use `view` instead of `flatten` for dynamically exporting to ONNX
            # [bs, c, h, w] -> [bs, h*w, c]
            feat = feat.view(batch_size, feat_dim, -1).permute(0, 2, 1)
            pos_embed = pos_embed.view(batch_size, feat_dim, -1).permute(0, 2, 1)
            # [bs, h, w] -> [bs, h*w]
            if masks is not None:
                masks = masks.view(batch_size, -1)

            # prepare transformer_inputs_dict
            encoder_inputs_dict = dict(
                feat=feat, feat_mask=masks, feat_pos=pos_embed)
            decoder_inputs_dict = dict(memory_mask=masks, memory_pos=pos_embed)
            return encoder_inputs_dict, decoder_inputs_dict 
        else:
            all_feat = None # [256,672,256]
            all_pos_embed = None # [256,672,256]
            for i,feat in enumerate(img_feats):
                feat = img_feats[i]  # NOTE img_feats contains only one feature.   
                batch_size, feat_dim, _, _ = feat.shape
                
                # construct binary masks which for the transformer.    
                masks = None
                # [batch_size, embed_dim, h, w]
                pos_embed = self.positional_encoding(masks, input=feat) # feat[2,256,32,16]->[2,256,32,16];feat[2,256,16,8]->[2,256,16,8]
            
                # use `view` instead of `flatten` for dynamically exporting to ONNX
                # [bs, c, h, w] -> [bs, h*w, c]
                feat = feat.view(batch_size, feat_dim, -1).permute(0, 2, 1)
                pos_embed = pos_embed.view(batch_size, feat_dim, -1).permute(0, 2, 1)
                # [bs, h, w] -> [bs, h*w]
                # if masks is not None:
                #     masks = masks.view(batch_size, -1)
                if all_feat is None:
                    all_feat = feat
                else:
                    all_feat = torch.cat([all_feat,feat],dim=1)
                if all_pos_embed is None :
                    all_pos_embed = pos_embed
                else:
                    all_pos_embed = torch.cat([all_pos_embed,pos_embed],dim=1)

            # prepare transformer_inputs_dict
            encoder_inputs_dict = dict(
                feat=all_feat, feat_mask=masks, feat_pos=all_pos_embed)
            decoder_inputs_dict = dict(memory_mask=masks, memory_pos=all_pos_embed)
            return encoder_inputs_dict, decoder_inputs_dict
        
    
    def forward_encoder(self, feat: Tensor, feat_mask: Tensor,
                        feat_pos: Tensor) -> Dict:
        """Forward with Transformer encoder.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            feat (Tensor): Sequential features, has shape (bs, num_feat_points,
                dim).
            feat_mask (Tensor): ByteTensor, the padding mask of the features,
                has shape (bs, num_feat_points).
            feat_pos (Tensor): The positional embeddings of the features, has
                shape (bs, num_feat_points, dim).

        Returns:
            dict: The dictionary of encoder outputs, which includes the
            `memory` of the encoder output.
        """
        memory = self.encoder(
            query=feat, query_pos=feat_pos,
            key_padding_mask=feat_mask)  # for self_attn
        encoder_outputs_dict = dict(memory=memory)
        return encoder_outputs_dict
    
    def pre_decoder(self, memory: Tensor) -> Tuple[Dict, Dict]:
        """Prepare intermediate variables before entering Transformer decoder,
        such as `query`, `query_pos`.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).

        Returns:
            tuple[dict, dict]: The first dict contains the inputs of decoder
            and the second dict contains the inputs of the bbox_head function.

            - decoder_inputs_dict (dict): The keyword args dictionary of
              `self.forward_decoder()`, which includes 'query', 'query_pos',
              'memory'.
            - head_inputs_dict (dict): The keyword args dictionary of the
              bbox_head functions, which is usually empty, or includes
              `enc_outputs_class` and `enc_outputs_class` when the detector
              support 'two stage' or 'query selection' strategies.
        """

        batch_size = memory.size(0)  # [bs, num_feat_points, dim]
        query_pos = self.query_embedding.weight
        # [num_queries, dim] -> [bs, num_queries, dim]
        query_pos = query_pos.unsqueeze(0).repeat(batch_size, 1, 1)
        query = torch.zeros_like(query_pos)

        decoder_inputs_dict = dict(
            query_pos=query_pos, query=query, memory=memory)
        head_inputs_dict = dict()
        return decoder_inputs_dict, head_inputs_dict
    
    def forward_decoder(self, query: Tensor, query_pos: Tensor, memory: Tensor,
                        memory_mask: Tensor, memory_pos: Tensor) -> Dict:
        """Forward with Transformer decoder.

        Args:
            query (Tensor): The queries of decoder inputs, has shape
                (bs, num_queries, dim).
            query_pos (Tensor): The positional queries of decoder inputs,
                has shape (bs, num_queries, dim).
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).
            memory_mask (Tensor): ByteTensor, the padding mask of the memory,
                has shape (bs, num_feat_points).
            memory_pos (Tensor): The positional embeddings of memory, has
                shape (bs, num_feat_points, dim).

        Returns:
            dict: The dictionary of decoder outputs, which includes the
            `hidden_states` and `references` of the decoder output.

            - hidden_states (Tensor): Has shape
                (num_decoder_layers, bs, num_queries, dim)
            - references (Tensor): Has shape
                (bs, num_queries, 2)
        """

        hidden_states, references,pos_attn_score_layers = self.decoder(
            query=query,
            key=memory,
            query_pos=query_pos,
            key_pos=memory_pos,
            key_padding_mask=memory_mask)
        head_inputs_dict = dict(
            hidden_states=hidden_states, references=references)
        return head_inputs_dict,pos_attn_score_layers
    
    def forward(self,img_feats):    
        
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats)
        
        if self.with_encoder:
            encoder_outputs_dict = self.forward_encoder(**encoder_inputs_dict)
        else:
            encoder_outputs_dict = {'memory':encoder_inputs_dict['feat']}
            
        tmp_dec_in, head_inputs_dict = self.pre_decoder(**encoder_outputs_dict)
        decoder_inputs_dict.update(tmp_dec_in)  
        
        decoder_outputs_dict,pos_attn_score_layers = self.forward_decoder(**decoder_inputs_dict) 
        head_inputs_dict.update(decoder_outputs_dict)
        
        return head_inputs_dict ,pos_attn_score_layers
        
    
@MODELS.register_module()
class PDNet(BaseModule):
    """
    The PartDecoder serves as the neck of the ReID network.
    The data processing flow is as :
        +----------------+
        |Pyramid Sampler |
        +----------------+
               |
               V
        +-----------+
        |PartDecoder|
        +-----------+
               |
               V
        Get the global query for further operation...
    
    Args:
        embed_dims (int): hidden feature dimension.
        with_agg (bool): whether operate pyramid sample.
        decoder (OptConfigType) : config of decoder.
        positional_encoding (OptConfigType) :config of positional embedding.
        num_queries (int): the number of part queries, not include global query.
        channel_mapper (OptConfigType): config of pyramid sampler.
    
    Return:
        tuple: tuple of output hidden state (global query) of each layer.
    """
    
    def __init__(self, 
                 embed_dims:int=256,
                 with_agg:bool=False,
                 with_encoder:bool=False,
                 encoder: OptConfigType = None,
                 decoder: OptConfigType = None,
                 positional_encoding: OptConfigType = None,
                 num_queries: int = 16,
                 channel_mapper:OptConfigType=None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptConfigType = None
                 )->tuple:
        super(PDNet,self).__init__(init_cfg)
        self.num_queries = num_queries
        # self.encoder = DetrTransformerEncoder(**encoder)
        self.mspctrans = MSPCTransformer(    
                                       embed_dims=embed_dims,
                                       with_encoder=with_encoder,
                                       encoder=encoder,
                                       decoder=decoder,
                                       with_agg=with_agg,
                                       positional_encoding=positional_encoding,
                                       num_queries=num_queries,
                                       train_cfg=train_cfg,
                                       test_cfg=test_cfg)
        self.pyramid_sampler = ChannelMapper(**channel_mapper)
   
    def init_weights(self) -> None:
        return super().init_weights()

    
    def forward(self,x):    
        # Pyramid feature sampler 
        out = self.pyramid_sampler(x)   # ([256,512,32,16],[256,1024,16,8],[256,2048,8,4])->([256,512,32,16],[256,512,16,8],[256,512,8,4]) 
        out,pos_attn_score_layers = self.mspctrans(out) # out{'hidden_states'[num_layers,batch,num_queries+1,dim],'references'[1,batch,2]}
        hidden_state = (out['hidden_states'][-1][:,0],) # [256,256] -> 0=([256,256])
        # return (hidden_state,pos_attn_score_layers)  
        return hidden_state
        
    