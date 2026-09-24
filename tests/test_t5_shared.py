from types import SimpleNamespace
import torch
from transformers import T5Gemma2DecoderConfig
from transformers.models.t5gemma2.modeling_t5gemma2 import T5Gemma2Decoder
from personal_jev.t5_shared import T5SharedScorer


def tiny():
    torch.manual_seed(7)
    cfg=T5Gemma2DecoderConfig(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=2,
        num_attention_heads=2,num_key_value_heads=1,head_dim=8,query_pre_attn_scalar=8,
        layer_types=['sliding_attention','full_attention'],sliding_window=4,dropout_rate=0,attention_dropout=0)
    class Encoder(torch.nn.Module):
        def __init__(self):super().__init__();self.emb=torch.nn.Embedding(32,16)
        def forward(self,input_ids,attention_mask):return SimpleNamespace(last_hidden_state=self.emb(input_ids))
    class Base(torch.nn.Module):
        def __init__(self):
            super().__init__();self.model=torch.nn.Module();self.model.encoder=Encoder()
            self.model.decoder=T5Gemma2Decoder(cfg);self.head=torch.nn.Linear(16,32,bias=False)
            self.config=SimpleNamespace(get_text_config=lambda:cfg)
        def get_output_embeddings(self):return self.head
    sc=T5SharedScorer.__new__(T5SharedScorer);sc.model=Base().eval();sc.pad=0
    sc.device='cpu';sc.answer_ids=[4,5];sc.branch_batch=2;sc.name='t5gemma2';sc.last_counts={}
    return sc


def test_bounded_shared_matches_native_and_independent_documents():
    sc=tiny()
    es=[dict(root=[3,4,5,6,7,8],branches=[[2,3,4],[2,4,5,6]],n=2),
        dict(root=[3,4,5,6,7,8],branches=[[2,7]],n=1),dict(root=[9,8,7],branches=[[2,9,3]],n=1)]
    with torch.no_grad():
        want=sc.t5_forward(es,share=False)
        together=sc.shared_entries(es)
        counts=dict(sc.last_counts)
        separate=torch.cat([sc.shared_entries([e]) for e in es])
    torch.testing.assert_close(together,want,atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(together,separate,atol=1e-6,rtol=1e-5)
    assert counts['encoder_calls']==2 and counts['cross_projection_pairs']==4
    assert counts['decoder_batches']==3 and counts['max_branch_rows']==2


def test_many_branches_do_not_multiply_persistent_cross_kv_storage():
    sc=tiny(); e=dict(root=list(range(3,22)),branches=[[2,4,5]],n=1)
    sc.shared_entries([e]); original=sc.last_counts['cross_kv_storage_bytes']
    repeated=e|dict(branches=e['branches']*49,n=49)
    scores=sc.shared_entries([repeated])
    torch.testing.assert_close(scores,scores[:1].expand_as(scores))
    assert sc.last_counts['encoder_calls']==1
    assert sc.last_counts['cross_kv_storage_bytes']==original
    assert sc.last_counts['max_branch_rows']==2


def test_training_gradients_reach_encoder_and_decoder():
    sc=tiny();sc.model.train()
    es=[dict(root=[3,4,5],branches=[[2,6,7],[2,8]],n=2)]
    loss=sc.t5_forward(es).square().mean();loss.backward()
    assert sc.model.model.encoder.emb.weight.grad.abs().sum()>0
    assert sc.model.model.decoder.layers[0].self_attn.q_proj.weight.grad.abs().sum()>0


def test_shared_decoder_prefix_matches_full_with_sliding_cache_and_permutations():
    sc=tiny()
    # Common decoder prefix exceeds the tiny sliding window (4), so the cache
    # must retain the native cumulative position, not just the stored KV length.
    prefix=[2,3,4,5,6,7,8]
    es=[dict(root=[9,10,11,12],branches=[prefix+[9,10],prefix+[11],prefix+[12,13,14]],n=3),
        dict(root=[13,14,15],branches=[[2,4,6],[2,5]],n=2)]
    want=sc.shared_entries(es)
    sc.decoder_prefix=True
    got=sc.shared_entries(es); counts=dict(sc.last_counts)
    independent=torch.cat([sc.shared_entries([e]) for e in es])
    reverse=torch.cat(list(reversed(torch.split(sc.shared_entries(es[::-1]),[2,3]))))
    torch.testing.assert_close(got,want,atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(got,independent,atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(got,reverse,atol=1e-6,rtol=1e-5)
    assert counts['encoder_calls']==2 and counts['cross_projection_pairs']==4
    assert counts['shared_decoder_tokens']==8
    assert counts['decoder_batches']==5 and counts['max_branch_rows']==2


def test_native_full_model_lora_covers_and_backpropagates_through_both_text_stacks():
    from transformers import T5Gemma2Config, T5Gemma2ForConditionalGeneration
    from peft import LoraConfig,get_peft_model
    from personal_jev.t5_shared import t5_text_lora_targets
    cfg=dict(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=2,
        num_attention_heads=2,num_key_value_heads=1,head_dim=8,query_pre_attn_scalar=8,
        layer_types=['sliding_attention','full_attention'],sliding_window=4,
        max_position_embeddings=64,pad_token_id=0,bos_token_id=2,eos_token_id=1)
    config=T5Gemma2Config(encoder={'text_config':cfg,'vision_config':dict(hidden_size=16,intermediate_size=32,
        num_hidden_layers=1,num_attention_heads=2,image_size=16,patch_size=4),
        'mm_tokens_per_image':4,'image_token_index':30,'boi_token_index':28,'eoi_token_index':29},
        decoder=cfg,image_token_index=30)
    torch.manual_seed(19)
    model=T5Gemma2ForConditionalGeneration(config)
    targets,coverage=t5_text_lora_targets(model)
    assert coverage=={'encoder':{'layers':2,'modules':8},'decoder':{'layers':2,'modules':8}}
    assert len(targets)==16 and all('vision' not in n for n in targets)
    sc=T5SharedScorer.__new__(T5SharedScorer)
    sc.model=get_peft_model(model,LoraConfig(r=2,lora_alpha=4,target_modules=targets,bias='none'))
    sc.device='cpu';sc.pad=0;sc.answer_ids=[4,5];sc.name='t5gemma2'
    sc.model.train()
    scores=sc.forward_entries([dict(root=[3,4,5,6],branches=[[2,7,8],[2,9,10]],n=2)])
    torch.nn.functional.binary_cross_entropy_with_logits(scores,torch.tensor([1.,0.])).backward()
    for path in ['.encoder.text_model.layers.','.decoder.layers.']:
        params=[p for n,p in sc.model.named_parameters() if path in n and p.requires_grad]
        assert len(params)==16
        assert all(p.grad is not None for p in params)
        assert sum(p.grad.abs().sum() for p in params)>0
