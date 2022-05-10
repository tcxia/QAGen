import torch
from transformers import BertTokenizer

import collections
import os
import json
from tqdm import tqdm


from evals.eval import eval_qg
from squad import write_predictions, evaluate


def to_string(index, tokenizer):
    tok_tokens = tokenizer.convert_ids_to_tokens(index)
    tok_text = " ".join(tok_tokens)

    tok_text = tok_text.replace("[PAD]", "")
    tok_text = tok_text.replace("[SEP]", "")
    tok_text = tok_text.replace("[CLS]", "")
    tok_text = tok_text.replace(" ##", "")
    tok_text = tok_text.replace("##", "")

    tok_text = tok_text.strip()
    tok_text = " ".join(tok_text.split())
    return tok_text


def batch_to_device(batch, device):
    batch = (b.to(device) for b in batch)
    c_ids, q_ids, a_ids, start_pos, end_pos = batch
    c_len = torch.sum(torch.sign(c_ids), 1)
    max_c_len = torch.max(c_len)
    c_ids = c_ids[:, :max_c_len]
    a_ids = a_ids[:, :max_c_len]

    q_len = torch.sum(torch.sign(q_ids), 1)
    max_q_len = torch.max(q_len)
    q_ids = q_ids[:, :max_q_len]
    return c_ids, q_ids, a_ids, start_pos, end_pos


class Result(object):
    def __init__(self, context, real_question, posterior_question,
                 prior_question, real_answer, posterior_answer, prior_answer,
                 posterior_z_prob, prior_z_prob) -> None:
        self.context = context

        self.real_question = real_question
        self.posterior_question = posterior_question
        self.prior_question = prior_question

        self.real_answer = real_answer
        self.posterior_answer = posterior_answer
        self.prior_answer = prior_answer

        self.posterior_z_prob = posterior_z_prob
        self.prior_z_prob = prior_z_prob


def eval_vae(epoch, args, trainer, eval_data):
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    rawResult = collections.namedtuple(
        "rawResult", ["unique_id", "start_logits", "end_logits"])

    eval_loader, eval_examples, eval_features = eval_data

    all_results = []
    qa_results = []
    qg_results = []
    ret_dict = {}
    example_index = -1

    for batch in tqdm(eval_loader, desc="Eval Iter", leave=False, position=4):
        c_ids, q_ids, a_ids, start, end = batch_to_device(batch, args.device)
        batch_size = c_ids.size(0)
        batch_c_ids = c_ids.cpu().tolist()
        batch_q_ids = q_ids.cpu().tolist()
        batch_start = start.cpu().tolist()
        batch_end = end.cpu().tolist()

        batch_posterior_q_ids, batch_posterior_start, batch_posterior_end, posterior_z_prob = trainer.generate_posterior(
            c_ids, q_ids, a_ids)
        batch_start_logits, batch_end_logits = trainer.generate_answer_logits(
            c_ids, q_ids, a_ids)

        batch_posterior_q_ids = batch_posterior_q_ids.cpu().tolist()
        batch_posterior_start = batch_posterior_start.cpu().tolist()
        batch_posterior_end = batch_posterior_end.cpu().tolist()
        posterior_z_prob = posterior_z_prob.cpu()


        batch_prior_q_ids, batch_prior_start, batch_prior_end, prior_z_prob = trainer.generate_prior(c_ids)

        batch_prior_q_ids = batch_prior_q_ids.cpu().tolist()
        batch_prior_start = batch_prior_start.cpu().tolist()
        batch_prior_end = batch_prior_end.cpu().tolist()
        prior_z_prob = prior_z_prob.cpu()

        for i in range(batch_size):
            example_index += 1
            start_logits = batch_start_logits[i].detach().cpu().tolist()
            end_logits = batch_end_logits[i].detach().cpu().tolist()
            eval_feature = eval_features[example_index]
            unique_id = int(eval_feature.unique_id)

            context = to_string(batch_c_ids[i], tokenizer)

            real_question = to_string(batch_q_ids[i], tokenizer)
            posterior_question = to_string(batch_posterior_q_ids[i], tokenizer)
            prior_question = to_string(batch_prior_q_ids[i], tokenizer)

            real_answer = to_string(batch_c_ids[i][batch_start[i]:(batch_end[i] + 1)], tokenizer)
            posterior_answer = to_string(batch_c_ids[i][batch_posterior_start[i]:(batch_posterior_end[i] + 1)], tokenizer)
            prior_answer = to_string(batch_c_ids[i][batch_prior_start[i]:(batch_prior_end[i] + 1)], tokenizer)

            all_results.append(Result(context=context,
                                      real_question=real_question,
                                      posterior_question=posterior_question,
                                      prior_question=prior_question,
                                      real_answer=real_answer,
                                      posterior_answer=posterior_answer,
                                      prior_answer=prior_answer,
                                      posterior_z_prob=posterior_z_prob[i],
                                      prior_z_prob=prior_z_prob[i]))

            qg_results[unique_id] = posterior_question
            ret_dict[unique_id] = real_question
            qa_results.append(rawResult(unique_id=unique_id,
                                        start_logits=start_logits,
                                        end_logits=end_logits))

    output_prediction_file = os.path.join(args.model_dir, "pred.json")
    write_predictions(eval_examples, eval_features, qa_results, n_best_size=20,
                      max_answer_length=30, do_lower_case=True,
                      output_prediction_file=output_prediction_file,
                      verbose_logging=False,
                      version_2_with_negative=False,
                      null_score_diff_threshold=0,
                      noq_position=True)

    with open(args.dev_dir) as f:
        dataset_json = json.load(f)
        dataset = dataset_json["data"]
    with open(os.path.join(args.model_dir, "pred.json")) as prediction_file:
        predictions = json.load(prediction_file)
    ret = evaluate(dataset, predictions)
    bleu = eval_qg(ret_dict, qg_results)

    return ret, bleu, all_results
