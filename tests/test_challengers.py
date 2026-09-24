"""Small deterministic checks for cache branching and GLiClass isolation."""
import torch
from transformers import DynamicCache,Qwen3Config
from personal_jev.challengers import fork_cache,padded

def test_fork_preserves_root_and_independent_rows():
 cache=DynamicCache(config=Qwen3Config(num_hidden_layers=1))
 k=torch.randn(1,2,4,8);v=torch.randn_like(k);cache.update(k,v,0)
 branched=fork_cache(cache,3)
 assert branched.layers[0].keys.shape==(3,2,4,8)
 branched.layers[0].keys[0].zero_()
 assert torch.equal(cache.layers[0].keys,k)
 assert torch.equal(branched.layers[0].keys[1],k[0])

def test_fork_linear_state_independence():
 from transformers import Qwen3_5TextConfig
 cache=DynamicCache(config=Qwen3_5TextConfig(num_hidden_layers=2,layer_types=['linear_attention','full_attention']))
 cache.update_conv_state(torch.randn(1,12,4),0)
 cache.update_recurrent_state(torch.randn(1,2,4,4),0)
 cache.update(torch.randn(1,2,3,4),torch.randn(1,2,3,4),1)
 branch=fork_cache(cache,2)
 assert branch.layers[0].conv_states[0].shape[0]==2
 old=cache.layers[0].recurrent_states[0].clone()
 branch.layers[0].recurrent_states[0][0].zero_()
 assert torch.equal(cache.layers[0].recurrent_states[0],old)
 assert torch.equal(branch.layers[0].recurrent_states[0][1],old[0])

def test_padding_retains_all_tokens():
 ids,mask=padded([[3,4,5],[6]],0,'cpu')
 assert ids.tolist()==[[3,4,5],[6,0,0]]
 assert mask.sum(1).tolist()==[3,1]

def test_deberta_different_tree_positions_batch_matches_independent():
 from transformers import DebertaV2Config,DebertaV2Model
 from personal_jev.challengers import deberta_relative_positions
 torch.manual_seed(5)
 cfg=DebertaV2Config(vocab_size=32,hidden_size=16,num_hidden_layers=2,num_attention_heads=2,intermediate_size=32,
   relative_attention=True,position_buckets=8,max_position_embeddings=32,pos_att_type=['p2c','c2p'],hidden_dropout_prob=0,attention_probs_dropout_prob=0)
 model=DebertaV2Model(cfg).eval();ids=torch.randint(0,32,(2,7))
 # Two different root lengths and branch resets, sharing one padded batch.
 pos=torch.tensor([[0,1,2,3,4,3,4],[0,1,2,3,2,3,4]])
 seg=torch.tensor([[0,0,0,1,1,2,2],[0,0,1,1,2,2,2]])
 allowed=(seg[:,None,:]==0)|(seg[:,:,None]==seg[:,None,:])
 with torch.no_grad():
  emb=model.embeddings(input_ids=ids,position_ids=pos,mask=torch.ones_like(ids))
  together=model.encoder(emb,attention_mask=allowed,relative_pos=deberta_relative_positions(pos,cfg)).last_hidden_state
  separate=torch.cat([model.encoder(emb[i:i+1],attention_mask=allowed[i:i+1],relative_pos=deberta_relative_positions(pos[i:i+1],cfg)).last_hidden_state for i in range(2)])
 torch.testing.assert_close(together,separate,atol=1e-6,rtol=1e-5)

def test_t5_precomputed_cross_kv_matches_native_decoder():
 from types import SimpleNamespace
 from transformers import T5Gemma2DecoderConfig
 from transformers.models.t5gemma2.modeling_t5gemma2 import T5Gemma2Decoder
 from personal_jev.challengers import ChallengerScorer
 torch.manual_seed(7)
 cfg=T5Gemma2DecoderConfig(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=8,query_pre_attn_scalar=8,layer_types=['sliding_attention','full_attention'],sliding_window=4,dropout_rate=0,attention_dropout=0)
 class Encoder(torch.nn.Module):
  def __init__(self):super().__init__();self.emb=torch.nn.Embedding(32,16)
  def forward(self,input_ids,attention_mask):return SimpleNamespace(last_hidden_state=self.emb(input_ids))
 class Base(torch.nn.Module):
  def __init__(self):
   super().__init__();self.model=torch.nn.Module();self.model.encoder=Encoder();self.model.decoder=T5Gemma2Decoder(cfg);self.head=torch.nn.Linear(16,32,bias=False);self.config=SimpleNamespace(get_text_config=lambda:cfg)
  def get_output_embeddings(self):return self.head
 sc=ChallengerScorer.__new__(ChallengerScorer);sc.model=Base().eval();sc.pad=0;sc.device='cpu';sc.answer_ids=[4,5];sc.branch_batch=2
 entries=[dict(root=[3,4,5,6,7,8],branches=[[2,3,4],[2,4,5,6]],n=2),dict(root=[3,4,5,6,7,8],branches=[[2,7]],n=1),dict(root=[9,8,7],branches=[[2,9,3]],n=1)]
 with torch.no_grad():
  a=sc.t5_forward(entries,cache_cross=True);b=sc.t5_forward(entries,cache_cross=False)
  c=sc.t5_forward(entries[:2],cache_cross=True);d=sc.t5_forward(entries[:2],cache_cross=False)
  tree=sc.t5_cached_prefix(entries)
 torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
 torch.testing.assert_close(c,d,atol=1e-6,rtol=1e-5)

 torch.testing.assert_close(tree,b,atol=1e-6,rtol=1e-5)

def test_native_rotary_precision_survives_bf16_device_transfer():
 from personal_jev.challengers import place_model
 model=torch.nn.Linear(3,2,dtype=torch.bfloat16)
 frequency=torch.tensor([0.123456789,0.000123456789],dtype=torch.float32)
 model.register_buffer('inv_freq',frequency.clone())
 place_model(model,'cpu','bfloat16')
 assert model.weight.dtype==torch.bfloat16
 assert model.inv_freq.dtype==torch.float32
 assert torch.equal(model.inv_freq,frequency)
 place_model(model,'cpu','float32')
 assert model.weight.dtype==torch.float32
 assert torch.equal(model.inv_freq,frequency)
