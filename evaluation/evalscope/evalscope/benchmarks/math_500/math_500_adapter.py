from collections import defaultdict

from evalscope.benchmarks import Benchmark, DataAdapter
from evalscope.constants import AnswerKeys
from evalscope.metrics.math_parser import extract_answer, math_equal, strip_answer_string
from evalscope.models import ChatGenerationModelAdapter
from evalscope.utils.logger import get_logger

# flake8: noqa

logger = get_logger()


@Benchmark.register(
    name='math_500',
    dataset_id='AI-ModelScope/MATH-500',
    model_adapter=ChatGenerationModelAdapter,
    subset_list=['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5'],
    metric_list=['AveragePass@1'],
    few_shot_num=0,
    train_split=None,
    eval_split='test',
    prompt_template='{query}\nPlease reason step by step, and put your final answer within \\boxed{{}}.',
)
class Math500Adapter(DataAdapter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load(self, **kwargs):
        # default load all levels
        kwargs['subset_list'] = ['default']
        return super().load(**kwargs)

    def gen_prompts(self, data_dict: dict) -> dict:
        res_dict: dict = defaultdict(list, {key: [] for key in self.subset_list})

        #  use level as subset
        for sub_name, sub_data_dict in data_dict.items():
            for sample_d in sub_data_dict[self.eval_split]:
                level = f"Level {sample_d['level']}"
                if level not in self.subset_list:
                    continue
                prompt_d = self.gen_prompt(input_d=sample_d, few_shot_list=None)
                prompt_d[AnswerKeys.RAW_INPUT] = sample_d
                res_dict[level].append(prompt_d)

        return res_dict

    def gen_prompt(self, input_d: dict, few_shot_list: list, **kwargs) -> dict:
        """
        Generate the prompt for the model input.
        """
        problem = input_d['problem']
        full_prompt = self.prompt_template.format(query=problem)

        return {'data': [full_prompt], 'system_prompt': self.system_prompt}

    def get_gold_answer(self, input_d: dict) -> str:
        # Extract the gold answer from the input dict.
        return strip_answer_string(input_d['answer'])

    def parse_pred_result(self, result: str, raw_input_d: dict = None, eval_type: str = 'checkpoint') -> str:
        """
        Parse the model output to get the answer. Could be the best choice index.
        """
        # Note: Use same extraction method for both of checkpoint/service/custom
        result = strip_answer_string(extract_answer(result))
        return result

    def match(self, gold: str, pred: str) -> float:
        return math_equal(pred, gold)
