import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import pandas as pd
import numpy as np
import json
import base64
import os
import json
from gen_eval import *
import my_api_keys
import argparse
import pickle as pkl
from tools.codeact_agent import CodeActAgent
import agentscope
import nest_asyncio
from agentscope.message import Msg

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = (
        my_api_keys.OPENAI_API_KEY
    )

Metric_Name_MAP = {
    "easy_stock":{"future price": "mape", "future volatility":"mape","future trend": "accuracy"},
    "stock": "total_return",
    "electricity_prediction": "mape",
    "electricity_prediction_single": "mape",
    "causal_relation": "accuracy",
    "causal_knowledge": "accuracy",
    "climate_anomaly": "f1",
    "climate_anomaly_large": "f1",
    "energy_anomaly": "f1",
    "ecg_anomaly": "f1",
    "electricity_prediction_large": "mape",
    "stock_investment": "result",
    "stock_ir_estimation":"absolute_diff",
    "stock_rv_estimation":"absolute_diff",
    "stock_var_estimation": "absolute_diff" # of violation_rate
}


def read_prompt_template(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    prompts = re.findall(r'(\w+)\s*=\s*"""(.*?)"""', content, re.DOTALL)
    
    # Convert to dictionary
    prompt_dict = {name: text.strip() for name, text in prompts}
    
    return prompt_dict

def get_model(model_id, available_variables):

    YOUR_MODEL_CONFIGURATION_NAME = model_id

    if model_id.startswith("deepseek"):
        YOUR_MODEL_CONFIGURATION = {
            "config_name": model_id,
            "model_type": "openai_chat",
            "model_name": model_id,
            "api_key":  my_api_keys.DEEPSEEK_API_KEY,
            "client_args":{
                "base_url": "https://api.deepseek.com",
            },
            "temperature": 0,
            "top_p":1
        }
    elif model_id.startswith("gemini"):
            YOUR_MODEL_CONFIGURATION = {
            "config_name": model_id,
            "model_type": "gemini_chat",
            "model_name": model_id,
            "api_key": my_api_keys.GEMINI_API_KEY,
        }
    elif model_id.startswith("qwen"):
        YOUR_MODEL_CONFIGURATION = {
            "config_name": model_id,
            "model_type": "openai_chat",
            "model_name": model_id,
            "client_args":{
                "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            },
            "temperature": 0,
            "top_p":1,
            "api_key": my_api_keys.QWEN_API_KEY
        }
    elif model_id.startswith("codestral"):
        YOUR_MODEL_CONFIGURATION = {
            "config_name": model_id,
            "model_type": "openai_chat",
            "model_name": model_id,
            "client_args":{
                "base_url": "https://codestral.mistral.ai/v1",
            },
            "temperature": 0,
            "top_p":1,
            "api_key": my_api_keys.CODESTRAL_API_KEY
        }
    elif model_id.startswith("mistral"):
        YOUR_MODEL_CONFIGURATION = {
            "config_name": model_id,
            "model_type": "openai_chat",
            "model_name": model_id,
            "client_args":{
                "base_url": "https://api.mistral.ai/v1",
            },
            "temperature": 0,
            "top_p":1,
            "api_key": my_api_keys.MISTRAL_API_KEY
        }
    elif model_id.startswith("claude"):
        YOUR_MODEL_CONFIGURATION =  {
            "config_name": model_id,
            "model_type": "anthropic_chat",
            "model_name": model_id,
            "api_key": my_api_keys.CLAUDE_API_KEY,
        }
    elif model_id.startswith("llama"):
        YOUR_MODEL_CONFIGURATION = {
            "config_name": model_id,
            "model_type": "openai_chat",
            "model_name": model_id,
            "client_args":{
                "base_url": "https://api.llmapi.com",
            },
            "temperature": 0,
            "top_p":1,
            "api_key": my_api_keys.LLAMA_API_KEY
        }

    else:
        YOUR_MODEL_CONFIGURATION = {
            "config_name": model_id,
            "model_type": "openai_chat",
            "model_name": model_id,
            "api_key":  os.environ["OPENAI_API_KEY"],
            "temperature": 0,
            "top_p":1
        }
    print("YOUR_MODEL_CONFIGURATION: ", YOUR_MODEL_CONFIGURATION)
    agentscope.init(model_configs=YOUR_MODEL_CONFIGURATION,disable_saving=True)
    nest_asyncio.apply()
    agent = CodeActAgent(
        name="assistant",
        model_config_name=YOUR_MODEL_CONFIGURATION_NAME,
        example_code=",".join(available_variables),
    )
    return agent

class Querier:
    def __init__(self, question_type: str, question_name: str, sample_num: int = None):
        # Read the prompt template
        self.question_type_fullname = question_type
        self.question_type = question_type.split("-")[0]
        # self.question_name = question_name
        # print("------------------------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>", self.question_name)
        # prompt_path = os.path.join(os.path.dirname(__file__), 'CoTPrompt.py')
        # prompt_dict = read_prompt_template(prompt_path)
        # self.template = prompt_dict[f"{question_type}_CoT_prompt"]
        # self.template = prompt_dict[self.question_name]
        use_class = Question_Type_MAP.get(self.question_type)
        generator_class = use_class[0]
        evaluator_class = use_class[1]
        self.generator_class = generator_class
        self.evaluator_class = evaluator_class
        self.sample_num = sample_num
        print("question_type: ", self.question_type)
        sub_case = question_type.split("-")[-1]
        print("sub_case: ", sub_case)
        if self.question_type =="easy_stock":
            self.sub_metric_name = Metric_Name_MAP[self.question_type][sub_case]
        else:
            self.sub_metric_name = Metric_Name_MAP[self.question_type]
        print(Metric_Name_MAP, self.sub_metric_name)
        # print("template: ", self.template)

    def check_problematic(self,question):
        if (self.question_type == "electricity_prediction") and np.min(question["ground_truth_data"])<1e-4:
            return True
        if (self.question_type == "stock") and np.any(np.isnan(question["executor_variables"]["VAL"]).values):
            return True
        if np.any(np.isnan(question["ground_truth_data"])):
            return True
        return False


    def query(self,questions):
        # set up evaluation
        success = 0
        sub_metric = []
        failure = 0
        causal_strict = 0
        problematic_question = 0
        if self.sample_num is not None:
            questions = questions[:self.sample_num]
        else:
            self.sample_num = len(questions)
        for i in range(len(questions)):
            print("success: ", success, "failure: ", failure, "sub_metric: ", sub_metric)
            question = questions[i]
            if self.check_problematic(question):
                problematic_question += 1
                continue
            context = question["context"]
            constraint = question["constraint"]
            user_prompt = question["prompt"]

            #model is an agent

            agent = get_model(args.model_id, list(question["executor_variables"].keys()))
            agent.code_executor.inject_available_variables(question["executor_variables"])

            print("user_prompt: ", user_prompt)
            mss = Msg(
                name="user", 
                content=user_prompt,
                role="user"
            )
            response_str = agent(mss)
            print("response_str: ", response_str)
            final_value = agent.code_executor.access_variable("predictions")
            import asyncio
            import gc
            asyncio.run(agent.code_executor._kill_client())  # 关闭 kernel 与 socket
            del agent
            gc.collect()
            print("final value \n",type(final_value),final_value.shape if hasattr(final_value, 'shape') else None,final_value)
            evaluator = self.evaluator_class(response = final_value, ground_truth_data = question["ground_truth_data"], context = context, constraint = constraint)
            try:
                score = evaluator.evaluate()
            except Exception as e:
                print("error: ", e)
                score = {"status": 0, "error": str(e)}
            print("evaluation result: ", score)
            if (self.question_type == "causal_relation"):
                if "strict_success" in score and (score["strict_success"]==1):
                    causal_strict+=1
            if score["status"]==1:
                success+=1
                entry = score.get(self.sub_metric_name,None)
                if entry is not None:
                    sub_metric.append(score[self.sub_metric_name])
            else:
                print(f"no {self.sub_metric_name} in the result", score.keys())
                failure+=1
        print("sample_num: ", self.sample_num)
        print("overall success rate: ", success/self.sample_num)
        print("execution and format failure rate: ", failure/self.sample_num)
        if self.sample_num-failure>0:
            print("no failure success rate: ", success/(self.sample_num-failure))
            if self.question_type == "stock_investment":
                aggregated = defaultdict(list)
                for d in sub_metric:
                    for k, v in d.items():
                        aggregated[k].append(v)
                print("aggregated: ", aggregated)
                mean_dict = {k: np.mean(v) for k, v in aggregated.items()}
                std_dict = {k: np.std(v) for k, v in aggregated.items()}
                print("mean_dict: ", mean_dict)
                print("std_dict: ", std_dict)
            else:
                print("no failure sub_metric: ", np.mean(sub_metric))
                print("no failure sub_metric std: ", np.std(sub_metric))
        else:
            print("all failures, no sub_metric")
        if self.question_type == "causal_relation":
            print("causal strict rate: ", causal_strict/self.sample_num)


            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Querier')
    # parser.add_argument('--question_type', type=str, default="stock", help='question type')
    # parser.add_argument('--question_name', type=str, default="Future_Price_CoT_Pure", help='question type')
    parser.add_argument('--sample_num', type=int, default=None, help='sample number')
    parser.add_argument("--global_seed", type=int, default=46)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--question_path", type=str, default="data.pkl")
    parser.add_argument("--model_id", type=str, default="gpt-4o",choices=["gpt-4o", "o1-preview", "gemini-2.0-flash-lite", "gemini-2.5-pro-exp-03-25", "claude-3-7-sonnet-20250219", "claude-3-5-sonnet-20241022","qwen-max-2025-01-25", "llama3.1-70b", "mistral-small-latest", "codestral-latest", "deepseek-chat","deepseek-reasoner"], help="model id")
    parser.add_argument("--begin_question_type", type=str, default=None, help="begin question type")
    parser.add_argument("--specific_question_type", type=str, default=None, help="specific question type")
    parser.add_argument('--exclude_question_type',nargs='+',type=str,help='List of strings', default=None)
    args = parser.parse_args()
    np.random.seed(args.global_seed)
    print("exclude_question_type: ", args.exclude_question_type)

    with open(args.question_path, "rb") as f:
        data = pkl.load(f)

    # Group dictionaries by 'question_type'
    grouped = defaultdict(list)
    for item in data:
        qtype = item.get("question_type")
        if qtype:
            grouped[qtype].append(item)

    
    begin_querying = False
    for qtype, samples in grouped.items():
        if args.begin_question_type is not None:
            assert args.specific_question_type is None, "you can only specify one of begin_question_type and specific_question_type"
            if qtype.startswith(args.begin_question_type):
                begin_querying = True
            if begin_querying:
                if args.exclude_question_type is not None and qtype.split("-")[0] in args.exclude_question_type:
                    print(f"skipping {qtype} as it is in the exclude list")
                    continue
                querier = Querier(qtype, "", args.sample_num)
                try:
                    querier.query(grouped[qtype])
                except Exception as e:
                    print("error: ", e)
                    print("continuing to the next question type due to error")
                    continue
        elif args.specific_question_type is not None:
            assert args.begin_question_type is None, "you can only specify one of begin_question_type and specific_question_type"
            if args.exclude_question_type is not None and qtype.split("-")[0] in args.exclude_question_type:
                    print(f"skipping {qtype} as it is in the exclude list")
                    continue
            if qtype.startswith(args.specific_question_type):
                querier = Querier(qtype, "", args.sample_num)
                try:
                    querier.query(grouped[qtype])
                except Exception as e:
                    print("error: ", e)
                    print("continuing to the next question type due to error")
                    continue
        else:
            if args.exclude_question_type is not None and qtype.split("-")[0] in args.exclude_question_type:
                print(f"skipping {qtype} as it is in the exclude list")
                continue
            querier = Querier(qtype, "", args.sample_num)
            try:
                querier.query(grouped[qtype])
            except Exception as e:
                print("error: ", e)
                print("continuing to the next question type due to error")
                continue