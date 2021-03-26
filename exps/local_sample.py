from summarize_from_feedback import sample
from summarize_from_feedback.utils import experiment_helpers as utils

HParams = sample.HParams(
	model_spec = utils.sup4_ppo_rm4(),
	task = utils.tldr_task,
    query_dataset_split= "valid",
    num_queries = 4
	)

sample.main(H = HParams)