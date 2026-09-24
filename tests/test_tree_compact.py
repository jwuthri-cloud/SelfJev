import json
import random

import pytest
import torch
from peft import LoraConfig, get_peft_model

from personal_jev.classify import classify
from personal_jev.model import MODEL_ID, MODEL_REVISION
from personal_jev.schemas import parse_request
from personal_jev.train import grouped_loss, shuffle_candidates
from personal_jev.train_tree import distillation_loss
from personal_jev.tree import COMPACT_FORMAT, Encoder, TreeScorer, format_config, leaf_paths
from tests.test_tree import REQ, close, scorer, scores_by_id, standalone, tiny_lm


def test_compact_standalone_and_isolation():
    sc=scorer(format_name=COMPACT_FORMAT)
    trees,_=sc.trees([parse_request(REQ)])
    want=[standalone(sc,[trees[0]['ids'][i] for i in p]) for p in leaf_paths(trees[0])]
    with torch.no_grad():
        for scores in (sc.model.packed(trees),sc.model.cached(trees)):
            assert torch.allclose(scores,torch.tensor(want),atol=1e-4,rtol=0)
    original=scores_by_id(classify(sc,REQ))
    modified=json.loads(json.dumps(REQ)); modified['questions'].reverse()
    modified['questions'].append({'id':'noise','type':'binary','instruction':'Is this an unrelated question?'})
    close(original,scores_by_id(classify(sc,modified)))
    old=Encoder(sc.tokenizer,MODEL_ID)
    assert len(sc.enc.question([])) < len(old.question([]))
    assert len(sc.enc.leaf([])) < len(old.leaf([]))


def test_compact_adapter_format_is_automatic_and_checked(tmp_path):
    lm=get_peft_model(tiny_lm(),LoraConfig(r=4,lora_alpha=8,target_modules=['q_proj','v_proj']))
    lm.save_pretrained(tmp_path)
    (tmp_path/'tree_format.json').write_text(json.dumps(format_config(MODEL_ID,COMPACT_FORMAT)))
    sc=scorer(lm=tiny_lm(),adapter=tmp_path)
    assert sc.meta['prompt']==COMPACT_FORMAT
    with pytest.raises(ValueError,match='differs'):
        scorer(lm=tiny_lm(),adapter=tmp_path,format_name='tree-v1')


@pytest.mark.parametrize('kind',['multiclass','multilabel'])
def test_teacher_and_gold_targets_follow_candidate_permutation(kind):
    item={'ids':[[0],[1],[2]],'type':kind,'target':1 if kind=='multiclass' else [False,True,False],
          'teacher_scores':[-2.,3.,.5],'candidate_ids':['a','b','c']}
    scores=torch.tensor([-.4,.9,1.1],requires_grad=True)
    original=grouped_loss(scores,[item])+distillation_loss(scores,[item])
    shuffled=shuffle_candidates(item,random.Random(13))
    reordered=scores[[v[0] for v in shuffled['ids']]]
    after=grouped_loss(reordered,[shuffled])+distillation_loss(reordered,[shuffled])
    assert torch.allclose(original,after)
    after.backward(); assert torch.isfinite(scores.grad).all()
