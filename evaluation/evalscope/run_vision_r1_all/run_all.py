import sys
from pathlib import Path
import argparse


# add our customized vlmeval into import path
current_file = Path(__file__).resolve()
evalscope_root = current_file.parent.parent
sys.path.insert(0, str(evalscope_root))
# -------------------------------------------

def build_task_config(args):
    model_config = {
        'api_base': 'http://127.0.0.1:8000/v1/chat/completions',
        'key': 'EMPTY',
        'name': 'CustomAPIModel',
        'type': args.model_name,
        'max_tokens': args.max_tokens
    }

    optional_args = {
        'temperature': args.temperature,
        'top_p': args.top_p,
        'timeout': args.timeout
    }
    
    for k, v in optional_args.items():
        if v is not None:
            model_config[k] = v

    return {
        'work_dir': args.work_dir,
        'eval_backend': 'VLMEvalKit',
        'eval_config': {
            'nproc': 128,
            'data': ['MathVista_MINI', 'MathVerse_MINI', 'MM-Math'],
            'mode': 'all',
            'model': [model_config],
            'verbose': True,
        },
    }
# you need to deploy vllm first

parser = argparse.ArgumentParser()
parser.add_argument('--work_dir', required=True)
parser.add_argument('--model_name', required=True)
parser.add_argument('--max_tokens', type=int, required=True)
parser.add_argument('--temperature', type=float, default=0.6)
parser.add_argument('--top_p', type=float, default=0.95)
parser.add_argument('--timeout', type=int, default=120000)

args = parser.parse_args()

task_cfg_dict = build_task_config(args)
print(f'task config dict : {task_cfg_dict}')

from evalscope.run import run_task
from evalscope.summarizer import Summarizer

def run_eval():
    task_cfg = task_cfg_dict
    run_task(task_cfg=task_cfg)

    print('>> Start to get the report with summarizer ...')
    report_list = Summarizer.get_report_from_cfg(task_cfg)
    print(f'\n>> The report list: {report_list}')

run_eval()