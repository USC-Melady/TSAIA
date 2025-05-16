import pandas as pd
import numpy as np
import json
import os
import re
import random
from sklearn.linear_model import LinearRegression
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support,accuracy_score
# import wfdb
from joblib import Parallel, delayed
import time
import warnings
warnings.filterwarnings("ignore")

VALI_NUM = 10

def detect_trend(time_series):
    """
    Detects the trend of a time series (increasing, decreasing, or steady).
    :param time_series: A pandas Series or list of time series values
    :return: A string indicating whether the trend is 'increasing', 'decreasing', or 'steady'
    """
    # Ensure the time series is a Pandas Series
    time_series = pd.Series(time_series)

    # Handle NaN values (if present) by dropping them
    time_series = time_series.dropna()

    # Create time index (e.g., 0, 1, 2, ..., n)
    time_index = np.arange(len(time_series)).reshape(-1, 1)

    # Fit a linear regression model to the time series
    model = LinearRegression()
    model.fit(time_index, time_series)

    # Get the slope (coefficient) of the fitted line
    slope = model.coef_[0]

    # Define thresholds for steady (you can tweak this value if needed)
    threshold = 1e-2

    # Determine the trend based on the slope
    if slope > threshold:
        return "increasing"
    elif slope < -threshold:
        return "decreasing"
    else:
        return "unknown"
    
def is_leap_year(year):
    """Check if a year is a leap year."""
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

def get_temporal_span(file_name):
    """
    Compute the temporal span (start and end timestamps) of a file.
    :param file_name: String in format 'YYYY_fileNumber.npz'
    :return: Tuple of (start_timestamp, end_timestamp)
    """
    # Parse file name
    year, file_number = map(int, file_name.replace('.npz', '').split('_'))
    
    # Constants
    hours_per_file = 546
    
    # Determine starting point based on whether previous year is a leap year
    if is_leap_year(year):
        start_of_data = pd.Timestamp(f"{year}-01-03 00:00:00")  # Leap year -> start Jan 3
    else:
        start_of_data = pd.Timestamp(f"{year}-01-02 00:00:00")  # Non-leap year -> start Jan 2
    
    # Compute hour offsets for this file
    start_hour = file_number * hours_per_file
    end_hour = start_hour + hours_per_file - 1

    # Calculate the timestamps
    start_timestamp = start_of_data + pd.Timedelta(hours=start_hour)
    end_timestamp = start_of_data + pd.Timedelta(hours=end_hour)

    return start_timestamp, end_timestamp

def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred

class Question_Generator:
    def __init__(self, input_data_paths: list, context: dict, constraint: dict, format: str, self_generate: bool = False, train_flag: bool = False):
        '''
        format: str, the format of the output, can be "prompt", "df", "json"
                promt: data is in the prompt, data is in the form of string
                df: data is in the form of pandas dataframe and returned separately
                json: data is in the form of json and returned separately
        '''
        self.format = format
        self.input_data_paths = input_data_paths
        self.context = context
        self.constraint = constraint
        assert self.format in ["prompt","df","json"]
        self.possible_case = None
        self.self_generate = self_generate
        self.train_flag = train_flag

    def generate(self):
        return NotImplementedError

class Evaluator:
    def __init__(self, response: dict, ground_truth_data: np.ndarray, context: dict, constraint: dict):
        self.response = response
        self.ground_truth_data = ground_truth_data
        self.context = context
        self.constraint = constraint

    def evaluate(self):
        return NotImplementedError



class Stock_Question_Generator_Single(Question_Generator):
    def __init__(self, input_data_paths: list, context: dict, constraint: dict, format: str, self_generate = False, train_flag = False):
        '''
        This function generates a question for the stock investment task where the goal is to generate an investment strategy for a given budget and corresponding outcome constriant.
        input_data_paths: list of input data paths for stock data
        context: dictionary containing the context information
                "hist_len": length of historical data; int
                "future_len": length of future data; int
                "names": list of stock names
                "var": variable to consider for stock data, default is "Close"
        constraint: dictionary containing the constraints for the question, you have to pass in a budget one other constraint
                "budget": budget for investment; float
        '''
        super().__init__(input_data_paths,context, constraint, format, self_generate, train_flag)
        #["profit_percent", "risk_tolerance"]#, "resolution_based_min_profit", "resolution_based_max_loss", "resolution_based_budget", "budget_allocation"]
        if len(self.input_data_paths) == 0:
            self.freq = "day"
            dir_name = "TS-Reasoning/day_yahoo"
            self.input_data_paths = [os.path.join(dir_name, path) for path in os.listdir(dir_name)]
            self.input_data_paths = sorted(self.input_data_paths)
            if self.train_flag:
                self.input_data_paths = self.input_data_paths[:len(self.input_data_paths)//2]
            else:
                self.input_data_paths = self.input_data_paths[len(self.input_data_paths)//2:]

    def generate(self):
        if self.self_generate:
            selected_paths = self.create_context_constraint()
        else:
            selected_paths = self.input_data_paths
        val, future_data = self.get_input_datas(selected_paths)
        hist_data = val.values
        budget = self.constraint.get("budget", None)
        assert budget is not None
        names = self.context["names"]
        prompt = f"I have historical stock price data of {names[0]} for {hist_data.shape[0]} days and I'm interested in investing in with a budget of {budget} dollars. "
        instruction =f"""Please give me an investment strategy for the next {len(future_data)} trading {self.freq}s. For each trading {self.freq}, generate a buy or sell signal based on the informaiton you have. Answer with a 1d numpy array for the next {len(future_data)} trading {self.freq}s where 1 indicates buy and -1 indicates sell and 0 indicates hold. The historical stock price data is stored in variable VAL and the future stock price data is stored in variable future_VAL."""
        output_requirement = "Answer: \n prediction = np.array([...],dtype=np.int64)\n prediction.shape = (future length,)"
        data_str = ""
        self.context["resolution"] = self.freq
        for i in range(len(names)):
            name = names[i]
            data_str+= f"The historical stock value data of {name} for the past {len(hist_data)} {self.freq}s is: ["
            data_str += ", ".join([str(round(x,2)) for x in hist_data[:,i]]) + "]. "
        if self.format == "prompt":
            prompt += data_str
            prompt += instruction
            return {"prompt": prompt,  "data_str": data_str, "ground_truth_data": future_data,  
                    "context": self.context, "constraint": self.constraint}
        elif self.format == "df":
            data_str = prompt + data_str + instruction
            prompt += instruction
            return {"prompt": prompt, "data_str": data_str, 
                    "ground_truth_data": future_data,  "context": self.context, "constraint": self.constraint,
                    "output_requirement": output_requirement,
                    "executor_variables":{"VAL": val, "future_VAL":future_data, "future_timestamp":future_data.index,"budget": budget}}
        elif self.format == "json":
            prompt += instruction
            data = self.input_datas.to_json()
            return {"prompt": prompt,  "data_str": data_str, "data": data, 
                    "ground_truth_data": future_data,  "context": self.context, "constraint": self.constraint}

    def get_input_datas(self,ps):
        assert len(ps) == 1, "Only one stock data is allowed"
        self.input_datas = pd.read_csv(ps[0],index_col=0,parse_dates=True)
        hist_len = self.context["hist_len"]
        future_len = self.context["future_len"]
        self.input_datas.index = pd.to_datetime(self.input_datas.index,utc=True).tz_convert(None).date
        after_date = pd.to_datetime("2006-03-01")
        slice_index = np.where(self.input_datas.index>=after_date)[0][0]
        self.input_datas = self.input_datas.iloc[slice_index:].copy()
        start = np.random.randint(0, len(self.input_datas) - hist_len - future_len)
        hist_data = self.input_datas[start:start+hist_len]
        future_data = self.input_datas[start+hist_len:start+hist_len+future_len]
        max_price = max(hist_data[["Open","High","Low","Close"]].max().max(),future_data[["Open","High","Low","Close"]].max().max())
        budget = np.random.randint(int(max_price*10), max(int(max_price*15),10000))
        self.constraint["budget"] = budget
        return hist_data, future_data

    def create_context_constraint(self):
        hist_len = np.random.randint(20, 40)
        future_len = np.random.randint(20, 60)
        paths = np.random.choice(self.input_data_paths, 1)
        names = [path.split("/")[-1].split(".")[0] for path in paths]
        self.context = {"hist_len": hist_len, "future_len": future_len, "names": names}
        return paths

class Stock_Evaluator_Single(Evaluator):
    def __init__(self, response: np.ndarray, ground_truth_data: pd.DataFrame, context: dict, constraint: dict):
        '''
        This function evaluates the response for the stock investment strategy.
        response: numpy array containing the response with 1 for buy and -1 for sell
        ground_truth_data: numpy array containing the ground truth data
        context: dictionary containing the context information
                "names": list of stock names
        constraint: dictionary containing the constraints for the question, you have to pass in a budget one other constraint
                "budget": budget for investment; float 
        '''
        super().__init__(response, ground_truth_data, context, constraint)
    
    def evaluate(self):
        try:
            assert len(self.ground_truth_data) == len(self.response), "Length of response and ground truth data should be the same"
            self.ground_truth_data["Position"] = self.response
            from backtest import evaluate_investment
            result = evaluate_investment(self.ground_truth_data, self.constraint["budget"])
            if result["Cumulative Return"]>=0:
                return {"status": 1, "result": result}
            else:
                return {"status": 0, "message": "Loss in investment", "result": result}
        except Exception as e:
            return {"status": 0, "message": str(e), "error": 1}
            
class Easy_Stock_Question_Generator(Question_Generator):
    def __init__(self, input_data_paths: list, context: dict, constraint: dict, format: str, self_generate = False, train_flag = False):
        '''
        This function generates a question for the stock investment task where the goal is to generate an investment strategy for a given budget and corresponding outcome constriant.
        input_data_paths: list of input data paths for stock data
        context: dictionary containing the context information
                "hist_len": length of historical data; int
                "future_len": length of future data; int
                "names": list of stock names
                "var": variable to consider for stock data, default is "Close"
        constraint: dictionary containing the constraints for the question, you have to pass in a budget one other constraint
                "future price": boolean, whether to predict future price
                "future volatility": boolean, whether to predict future volatility
                "future trend": boolean, whether to predict future trend
        '''
        super().__init__(input_data_paths,context, constraint, format, self_generate, train_flag)
        self.possible_case = ["future price", "future volatility", "future trend"]
        possible_resolutions = ["day", "hour"]
        chosen_resolution = np.random.choice(possible_resolutions)
        self.freq = chosen_resolution
        if self.freq == "day":
            dir_name = "TS-Reasoning/day_yahoo"
        else:
            dir_name = "TS-Reasoning/hour_yahoo"
        self.input_data_paths = [os.path.join(dir_name, path) for path in os.listdir(dir_name)]
        self.input_data_paths = sorted(self.input_data_paths)
        if self.train_flag:
            self.input_data_paths = self.input_data_paths[:len(self.input_data_paths)//2]
        else:
            self.input_data_paths = self.input_data_paths[len(self.input_data_paths)//2:]

    def generate(self):
        if self.self_generate:
            selected_paths = self.create_context_constraint()
        else:
            selected_paths = self.input_data_paths
        assert len(self.constraint.keys()) == 1
        assert sum(self.constraint.values()) == 1
        constriant_key = list(self.constraint.keys())[0]
        val, future_data, names = self.get_input_datas(selected_paths)
        hist_data = val.values
        
        constriant_prompt_map = {"future price": f"I want to predict the stock price for the future {len(future_data)} {self.freq}s. Your goal is to make the most accurate prediction. ",
                                 "future volatility": f"I want to predict the volatility of the stock price for the future {len(future_data)} {self.freq}s. Your goal is to make the most accurate prediction .",
                                 "future trend": f"I want to predict the trend of the stock price for the future {len(future_data)} {self.freq}s. Your goal is to make the most accurate prediction. "}

        prompt = f"I have the past {len(hist_data)} {self.freq}s historical stock value data for {hist_data.shape[1]} stocks that I'm interested in investing in. "
        constriant_description = constriant_prompt_map[constriant_key]
        prompt += constriant_description
        self.context["resolution"] = self.freq
        instruction_prompt_map = {"future price": "Please give me your prediction, return it as a 2d numpy array.",
                                    "future volatility": "Please give me your prediction, return a 1d numpy array with the predicted volatility of each stock.",
                                    "future trend": "Please give me your prediction, return a 1d numpy array with the predicted trend of each stock containing string values among (increasing, decreasing, unknown)."}
        output_requirement = {"future price": "Answer: \n prediction = np.array([[...], [...], ...],dtype=np.float64)\n prediction.shape = (future length, number of stocks)",
                                "future volatility": "Answer: \n prediction = np.array([...],dtype=np.float64)\n prediction.shape = (number of stocks)", 
                                "future trend": "Answer: \n prediction = np.array([...],dtype=<U)\n prediction.shape = (number of stocks)"}
        variable_information = "The historical stock value data is stored in variable VAL. "
        instruction = instruction_prompt_map[constriant_key]
        data_str = ""
        for i in range(len(names)):
            name = names[i]
            data_str+= f"The historical stock value data of {name} for the past {len(hist_data)} days is: ["
            data_str += ", ".join([str(round(x,2)) for x in hist_data[:,i]]) + "]. "
        if self.format == "prompt":
            prompt += data_str
            prompt += instruction
            return {"prompt": prompt, "ground_truth_data": future_data, 
                     "data_str": data_str, "context": self.context,"constraint": self.constraint,
                    "executor_variables":{"VAL": val,"N": len(future_data)},
                    "output_requirement": output_requirement[constriant_key]}
        elif self.format == "df":
            data_str = prompt + data_str + instruction
            prompt += instruction
            prompt += variable_information
            return {"prompt": prompt, 
                    "ground_truth_data": future_data,  "data_str": data_str, "context": self.context,"constraint": self.constraint,
                    "executor_variables":{"VAL": val,"N": len(future_data)},
                    "output_requirement": output_requirement[constriant_key]}
        elif self.format == "json":
            prompt += instruction
            data = self.input_datas.to_json()
            return {"prompt": prompt, 
                    "ground_truth_data": future_data, "data_str": data_str, "context": self.context, "constraint": self.constraint,
                    "executor_variables":{"VAL": val,"N": len(future_data)},
                    "output_requirement": output_requirement[constriant_key]}

    def get_input_datas(self, ps):
        var = self.context.get("var","Close")
        self.input_datas = []
        for i in range(len(ps)):
            self.input_datas.append(pd.read_csv(ps[i],index_col=0,parse_dates=True)[var])
        hist_len = self.context["hist_len"]
        future_len = self.context["future_len"]
        names = self.context["names"]
        if self.freq == "day":
            total_len = 252
        else:
            total_len = 7*5 #7 trading hours for 5 days
        for i in range(len(self.input_datas)):
            self.input_datas[i] = self.input_datas[i][-total_len:]
        assert hist_len > 0 and future_len > 0 and hist_len+future_len < total_len
        assert len(self.input_datas) == len(names)
        if len(self.input_datas) > 1:
            #check if all input datas have the same length
            for i in range(1, len(self.input_datas)):
                assert len(self.input_datas[i]) == len(self.input_datas[0])
        self.input_datas = pd.concat(self.input_datas, axis=1)
        self.input_datas.columns = names
        start = np.random.randint(0, len(self.input_datas) - hist_len - future_len)
        hist_data = self.input_datas[start:start+hist_len]#.values
        future_data = self.input_datas[start+hist_len:start+hist_len+future_len].values
        self.context["resolution"] = self.freq
        return hist_data, future_data, names

    def create_context_constraint(self):
        if self.freq == "day":
            hist_len = np.random.randint(20, 100)
            future_len = np.random.randint(10, 30)
        else:
            hist_len = np.random.randint(10, 20)
            future_len = np.random.randint(5, 35-hist_len)
        num_stocks = np.random.randint(1, 3)
        paths = np.random.choice(self.input_data_paths, num_stocks, replace=False)
        names = [path.split("/")[-1].split(".")[0] for path in paths]
        possible_constraints = ["future price", "future volatility", "future trend"]
        if self.constraint:
            assert len(self.constraint.keys()) == 1
            assert list(self.constraint.keys())[0] in possible_constraints
            chosen_constraints = list(self.constraint.keys())[0]
        else:
            chosen_constraints = np.random.choice(possible_constraints)
        self.context = {"hist_len": hist_len, "future_len": future_len, "names": names}
        self.constraint = {chosen_constraints: True}
        return paths

class Easy_Stock_Evaluator(Evaluator):
    def __init__(self, response: np.ndarray, ground_truth_data: np.ndarray, context: dict, constraint: dict):
        '''
        This function evaluates the response for the stock investment strategy.
        response: numpy array containing the response
        ground_truth_data: numpy array containing the ground truth data
        context: dictionary containing the context information
                "names": list of stock names
        constraint: dictionary containing the constraints for the question,
                "future price": boolean, whether to predict future price
                "future volatility": boolean, whether to predict future volatility
                "future trend": boolean, whether to predict future trend
        '''
        super().__init__(response, ground_truth_data, context, constraint)

    def evaluate(self):
        try:
            prediction = self.response
            future_data = self.ground_truth_data
            threshold = self.constraint.get("threshold", 1)
            names = self.context["names"]
            if self.constraint.get("future price", None):
                assert np.any(np.isnan(self.response)) == False, "Response contains NaN values"
                assert len(prediction) == len(future_data), "Prediction and ground truth data have different lengths"
                #calculate mape for each stock
                prediction += 1e-6
                future_data += 1e-6
                mape = np.mean(np.abs((prediction - future_data) / future_data),axis=0)
                mape = np.mean(mape)
                if mape < threshold:
                    return {"status": 1, "mape": mape, "answer": future_data}
                else:
                    return {"status": 0, "mape": mape, "answer": future_data}
            elif self.constraint.get("future volatility", None):
                assert np.any(np.isnan(self.response)) == False, "Response contains NaN values"
                #calculate mape
                volatility = np.std(future_data,axis=0)
                assert len(prediction) == len(volatility), "Prediction and ground truth data have different lengths"
                mape = np.abs((prediction - volatility) / volatility)
                mape = np.mean(mape)
                if mape < threshold:
                    return {"status": 1, "mape": mape, "answer": volatility}
                else:
                    return {"status": 0, "mape": mape, "answer": volatility}
            elif self.constraint.get("future trend", None):
                #calculate trend of ground truth data
                trend = []
                for i in range(len(names)):
                    trend.append(detect_trend(future_data[:,i]))
                assert len(prediction) == len(trend), "Prediction and ground truth data have different lengths"
                num_correct = np.sum([trend[n].upper() == prediction[n].upper() for n in range(len(names))])
                acc = num_correct/len(names)
                if acc > 0:
                    return {"status": 1, "answer": trend, "accuracy": acc}
                else:
                    return {"status": 0, "answer": trend, "accuracy": acc}
            else:
                raise ValueError("Unknown constraint")
        except Exception as e:
            return {"status": 0, "message": str(e), "error": 1}


class Electricity_Prediction_Question_Generator(Question_Generator):
    def __init__(self, input_data_paths: list, context: dict, constraint: dict, format: str, self_generate = False, train_flag = False):
        '''
        This function generates a question for the electricity consumption forecasting task.
        input_data_paths: list of input data paths
        context: dictionary containing the context information, it should contain keys "hist_len", "future_len", "influence_vars", "target_var"
            hist_len: length of historical data; int
            future_len: length of future data; int
            influence_vars: list of variables that influence the target variable; list of strings
            target_var: target variable; string
        constraint: dictionary containing the constraints for the question
            mape: Mean absolute percentage error; float
            max_load: Maximum allowable system load; float
            min_load: Minimum system load; float
            load_ramp_rate: Maximum allowable load change rate; float
            load_variability_limit: Maximum allowable variability in load over a given period; float
        '''
        super().__init__(input_data_paths, context, constraint, format, self_generate, train_flag)
        if len(self.input_data_paths) == 0:
            dir_name = "TS-Reasoning/energy_data/Minute-level Load and Renewable"
            self.input_data_paths = self.get_paths(dir_name)
        self.possible_case = ["max_load", "min_load", "load_ramp_rate", "load_variability_limit"]
        self.possible_target_var = ["load_power","wind_power","solar_power"]

    def get_paths(self, dir_name):
        paths = [path for path in os.listdir(dir_name) if path.endswith(".csv")]

        # **Sort using natural ordering (by numeric value)**
        def extract_zone_number(filename):
            match = re.search(r'zone_(\d+)_', filename)
            return int(match.group(1)) if match else float('inf')  # Assign large number if no match

        paths = sorted(paths, key=lambda x: (x.split("_")[0], extract_zone_number(x)))

        # Group by energy grid (e.g., PJM, SPP, etc.)
        grid_groups = {}
        for path in paths:
            grid_name = path.split("_")[0]  # Extract grid name (e.g., PJM, SPP)
            if grid_name not in grid_groups:
                grid_groups[grid_name] = []
            grid_groups[grid_name].append(os.path.join(dir_name, path))

        # **Split into first and second half properly**
        selected_paths = []
        for grid, grid_paths in grid_groups.items():
            mid = len(grid_paths) // 2
            if self.train_flag:
                selected_paths.extend(grid_paths[:mid])  # First half for training
            else:
                selected_paths.extend(grid_paths[mid:])  # Second half for testing
        return selected_paths

    def generate(self):
        
        if self.self_generate:
            self.create_context()
        hist_data, future_data = self.get_input_datas()
        assert np.sum(future_data[:,-1]) != 0 # Ensure that future_target_var is not all zeros.
        if self.self_generate:
            self.create_constraint(future_data[:,-1])
        
        
        hist_len = self.context["hist_len"]
        future_len = self.context["future_len"]
        influence_vars = self.context["influence_vars"]
        assert len(influence_vars) > 0
        target_var = self.context["target_var"]
        all_vars = influence_vars + target_var
        constriant_key = list(self.constraint.keys())[0]
        constriant_value = self.constraint[constriant_key]
        assert len(self.constraint.keys()) == 1
        assert len(target_var) == 1
        target_var = target_var[0]
        constriant_prompt_map = {"max_load": f"I need to ensure that the maximum allowable system load does not exceed {constriant_value} MW. ",
                            "min_load": f"I require that the system load is maintained above a minimum of {constriant_value} MW. ",
                            "load_ramp_rate": f"I must monitor the load ramp rate to ensure it does not exceed {constriant_value} MW for each time step. ",
                            "load_variability_limit": f"I need to manage the load variability so that it does not exceed {constriant_value} MW over the complete time period (i.e the maximum change in load over the entire period). "
                            }

                
        influence_vars_str = ", ".join(influence_vars)
        prompt = f"I have historical {influence_vars_str} data and the corresponding {target_var} data for the past {hist_len} minutes. "
        constriant_description = constriant_prompt_map[constriant_key]        
        prompt += constriant_description        

        output_requirement = defaultdict(lambda: f"""Answer: \n prediction = np.array([...],dtype=np.float64) \n prediction.shape = (future length)""")

        instruction = f"Think about how {influence_vars_str} influence {target_var}. Please give me a forecast for the next {future_len} minutes for {target_var}. Your goal is to make the most accurate forecast as possible, refine prediction result based on the constraint previously described, and return the result as a 1D numpy array. The historical data for both covariates and {target_var} are saved in variable VAL with last column being the target variable and future data of {influence_vars_str} are saved in variable MULVAL with last column being the target variable."
        data_str = ""
        for i in range(len(all_vars)):
            data_str+=f"The historical {all_vars[i]} data for the past {hist_len} minutes is: ["
            data_str += ", ".join([str(round(x,2)) for x in hist_data[:,i]]) + "]. "
        if self.format =="prompt":
            prompt += data_str
            prompt += instruction
            return {"prompt": prompt, "output_requirement": output_requirement[0],  "ground_truth_data": future_data[:,-1], 
                     "context": self.context, "constraint": self.constraint, "data_str": data_str, 
                    "executor_variables":{"VAL": self.input_datas[target_var].values,"N": len(future_data),"MULVAL":future_data[:, :-1], "constraint_str": constriant_description}}
        elif self.format == "df":
            target_var_np = self.input_datas[target_var].values.reshape(-1, 1)
            influence_vars_np = self.input_datas[influence_vars].values
            val = np.hstack((influence_vars_np, target_var_np))
            df = pd.DataFrame(val, columns=all_vars,index=self.input_datas.index)
            mulval = pd.DataFrame(future_data[:, :-1], columns=influence_vars, index=self.all_data.index[-future_len:])
            data_str = prompt + data_str + instruction
            prompt += instruction
            return_dict =  {"prompt": prompt, "output_requirement": output_requirement[0], "ground_truth_data": future_data[:,-1], 
                    "context": self.context, "constraint": self.constraint, "data_str": data_str,
                      "executor_variables":{"VAL": df,"N": len(future_data),"MULVAL":mulval, "constraint_str": constriant_description}}
            return return_dict
        elif self.format == "json":
            my_dict_serializable = {key: list(value) for key, value in zip(all_vars, hist_data.T)}
            data = json.dumps(my_dict_serializable)
            return {"prompt": prompt, "output_requirement": output_requirement[0], "data": data, 
                    "ground_truth_data": future_data[:,-1],  "context": self.context, 
                    "constraint": self.constraint, "data_str": data_str, 
                    "executor_variables":{"VAL": self.input_datas[target_var].values,"N": len(future_data),"MULVAL":future_data[:, :-1], "constraint_str": constriant_description}}

    def get_input_datas(self):
        influence_vars = self.context["influence_vars"]
        assert len(influence_vars) > 0
        target_var = self.context["target_var"]
        all_vars = influence_vars + target_var
        hist_len = self.context["hist_len"]
        future_len = self.context["future_len"]
        
        loc = np.random.randint(0, len(self.input_data_paths))
        print("reading file ", self.input_data_paths[loc])
        all_data = pd.read_csv(self.input_data_paths[loc],index_col=0).fillna(0) 
        temp = all_data[all_vars].values[:, -1]
        flag = np.all(np.isclose(temp, 0))
        while flag==True:
            print(f"in while loop for file {self.input_data_paths[loc]}! read a new file")
            print("new file is ", self.input_data_paths[loc])
            loc = np.random.randint(0, len(self.input_data_paths))
            all_data = pd.read_csv(self.input_data_paths[loc],index_col=0).fillna(0)
            temp = all_data[all_vars].values[:, -1]
            flag = np.all(np.isclose(temp, 0))
        data = all_data[all_vars].values
        def year_check(h,f):
            h_time = pd.to_datetime(h.index)
            f_time = pd.to_datetime(f.index)
            if (h_time.year.nunique() != 1) or (f_time.year.nunique() != 1):
                return False
            if h_time.year.unique()[0] != f_time.year.unique()[0]:
                return False
            return True
        future_target_var = np.zeros((future_len, 1))
        #don't want all zero cases
        while (np.all(np.isclose(future_target_var, 0))) or (not year_check(all_data[start:start+hist_len], all_data[start+hist_len:start+hist_len+future_len])):
            print("rechoosing start timestamp")
            start = np.random.randint(0, len(data)-hist_len-future_len)
            hist_data = data[start:start+hist_len]
            future_data = data[start+hist_len:start+hist_len+future_len]
            future_target_var = future_data[:,-1]
            self.input_datas = all_data[all_vars][start:start+hist_len]
            self.all_data = all_data[all_vars][start:start+hist_len+future_len]
        self.chosen_file = self.input_data_paths[loc]
        return hist_data, future_data

    def create_context(self):
        hist_len = np.random.randint(50,200)
        future_len = np.random.randint(10, 90)
        target_var = np.random.choice(self.possible_target_var, 1, replace=False).tolist()
        if "load_power" in target_var:
            influence_list = ["Dew Point", "Solar Zenith Angle", "Wind Speed", "Relative Humidity", "Temperature"]
            influence_vars = np.random.choice(influence_list, 3, replace=False).tolist()
        elif "wind_power" in target_var:
            influence_list = [ "Wind Speed", "Relative Humidity", "Temperature"]
            influence_vars = np.random.choice(influence_list, 2, replace=False).tolist()
        elif "solar_power" in target_var:
            influence_list = ["DHI","DNI","GHI","Dew Point","Solar Zenith Angle","Relative Humidity","Temperature"]
            influence_vars = np.random.choice(influence_list, 5, replace=False).tolist()

        # context
        self.context = {"hist_len": hist_len, "future_len": future_len}
        self.context["influence_vars"] = influence_vars
        self.context["target_var"] = target_var

        # print("context: ", self.context)

    def create_constraint(self, future_tar_data):
        if self.constraint:
            assert len(self.constraint.keys()) == 1
            assert list(self.constraint.keys())[0] in self.possible_case
            chosen_constraints = list(self.constraint.keys())[0]
        else: 
            chosen_constraints = np.random.choice(self.possible_case)
        
        # Ensure future target data satisfies constraints
        if chosen_constraints=="max_load":
            max_load = max(future_tar_data)
            constraint_value =  np.random.uniform(max_load, max_load*1.1)
        elif chosen_constraints=="min_load":
            min_load = min(future_tar_data)
            constraint_value =  np.random.uniform(min_load*0.9, min_load)
        elif chosen_constraints=="load_ramp_rate":
            load_changes = [future_tar_data[i] - future_tar_data[i - 1] for i in range(1, len(future_tar_data))]
            pred_ramp_rate = max([abs(change) for change in load_changes])
            constraint_value =  np.random.uniform(pred_ramp_rate, pred_ramp_rate*1.1)
        elif chosen_constraints=="load_variability_limit":
            forecast_variability = max(future_tar_data) - min(future_tar_data)
            pred_variability_limit = forecast_variability
            constraint_value =  np.random.uniform(pred_variability_limit, pred_variability_limit*1.1)
  
        self.constraint = {chosen_constraints: constraint_value}
    

class Electricity_Prediction_Evaluator(Evaluator):
    def __init__(self, response: dict, ground_truth_data: np.ndarray, context: dict, constraint: dict):
        '''
        This function evaluates the response for the target variable prediction task.
        response: dictionary containing the response with keys target_var and "explanation"
        ground_truth_data: numpy array containing the ground truth data
        context: dictionary containing the context information
            "target_var": target variable
        constraint: dictionary containing the constraints for the question
            mape: Mean absolute percentage error; float
            max_load: Maximum allowable system load; float
            min_load: Minimum system load; float
            load_ramp_rate: Maximum allowable load change rate; float
            load_variability_limit: Maximum allowable variability in load over a given period; float
        '''
        super().__init__(response, ground_truth_data, context, constraint)
    
    def evaluate(self):
        try:
            assert np.any(np.isnan(self.response)) == False, "Response contains NaN values"
            assert len(self.context["target_var"]) == 1 , "Only one target variable is allowed"
            target_var = self.context["target_var"][0]
            forecast = self.response
            ground_truth_data = self.ground_truth_data

            # explanation = self.response["explanation"] # TODO: Future work - how influence_vars influence target_var
            
            # If the MAPE value is not set, it will be assigned a default value of 0.1.
            threshold = self.constraint.get("threshold", 1)
            self.constraint["threshold"] = threshold
            
            assert len(self.constraint.keys()) == 2, "Only one constraint is allowed in addition to the MAPE threshold value"
            assert len(forecast) == len(ground_truth_data), "Length of forecast and ground truth data do not match"       
            
            max_load = self.constraint.get("max_load", None)
            min_load = self.constraint.get("min_load", None)
            load_ramp_rate = self.constraint.get("load_ramp_rate", None)
            load_variability_limit = self.constraint.get("load_variability_limit", None)
            
            # compute predicted constraint according to forecast
            pred_max_load = max(forecast)
            pred_min_load = min(forecast)
            
            load_changes = [forecast[i] - forecast[i - 1] for i in range(1, len(forecast))]
            pred_ramp_rate = max(abs(change) for change in load_changes)
            
            forecast_variability = max(forecast) - min(forecast)
            pred_variability_limit = forecast_variability
            # print(pred_max_load, pred_min_load, pred_ramp_rate, pred_variability_limit)
            
            forecast += 1e-6
            ground_truth_data += 1e-6
            
            mape_calculated = np.mean(np.abs((forecast - ground_truth_data) / ground_truth_data),axis=0)
            mape_calculated = np.mean(mape_calculated)
            
            if max_load and (pred_max_load-max_load)>1e-4:
                return {"status": 0, "message": f"Predicted load exceeds the maximum load limit of {max_load} MW. ", "mape": mape_calculated}
            elif min_load and (min_load-pred_min_load)>1e-4:
                return {"status": 0, "message": f"Predicted load falls below the minimum load limit of {min_load} MW.", "mape": mape_calculated}
            elif load_ramp_rate is not None and (pred_ramp_rate - load_ramp_rate)>1e-4:
                return {"status": 0, "message": f"Predicted load ramp rate exceeds the maximum allowable ramp rate of {load_ramp_rate} MW/min. ", "mape": mape_calculated}
            elif load_variability_limit is not None and (pred_variability_limit - load_variability_limit)>1e-4:
                return {"status": 0, "message": f"Predicted load variability exceeds the maximum allowable limit of {load_variability_limit} MW. ", "mape": mape_calculated}
            else:
                if mape_calculated > threshold:
                    message  = f"The MAPE value is {mape_calculated}, which exceeds the threshold {threshold}."
                else:
                    message = "All constraints are satisfied and the MAPE value is within the threshold."
                return {"status": int(mape_calculated<=threshold), "message": message, "mape": mape_calculated}
        except Exception as e:
            return {"status": 0, "message": str(e), "error": 1}
            
class Causal_Relation_Question_Generator(Question_Generator):
    def __init__(self, input_data_paths: list, context: dict, constraint: dict, format: str, self_generate = False,train_flag = False):
        '''
        This function generates a question for the causal relation prediction task.
        input_data_paths: list of input data paths
        '''
        super().__init__(input_data_paths,context, constraint, format, self_generate,train_flag)

    def generate(self):
        if self.self_generate:
            self.create_context_constraint()
        else:
            self.get_input_datas()
        relation_matrix = np.array(self.json_data['relation_matrix'])
        relation_ratio = self.json_data['relation_ratio']
        data_path = self.json_data['data_path']
        self.data = pd.read_csv(data_path,index_col=0).round(3)

        # Randomly select at least 10% of the data
        num_rows_to_select = max(1, int(0.1 * len(self.data)))
        start_idx = np.random.randint(0, len(self.data) - num_rows_to_select + 1)
        random_segment = self.data.iloc[start_idx : start_idx + num_rows_to_select]
        self.data = random_segment

        all_vars = self.data.columns.tolist()
        influence_vars_str = ", ".join(all_vars)
        self.data = self.data[all_vars].values

        data_str = ""
        # import pdb; pdb.set_trace()
        for i in range(len(all_vars)):
            data_str+=f"The historical {all_vars[i]} data for the past time steps is: ["
            data_str += ", ".join([str(round(x,2)) for x in self.data[:,i]]) + "]. "

        prompt = f"I have historical {influence_vars_str} data and want to get the causal relationship between each pair of the variables. "
        
        constriant_description = f"I know that {relation_ratio*100}% of the variable pairs have relationship. Self-causalation is not considered. "

        prompt += constriant_description

        instruction = f"Consider the potential influence of each variable on the others in this variable list: {all_vars}. " + \
            f"Please provide 2d numpy matrix with binary values to indicate whether each pair of variables has a relationship. The data for all variables are stored in variable VAL. "
        
        output_requirement = "Answer: \n prediction = np.array([[...], [...], ...],dtype=np.float64)\n prediction.shape = (number of variables, number of variables)\n Please return a 2D numpy matrix with 1 indicating a relationship and 0 indicating no relationship."

        self.constraint = {"relation_ratio": relation_ratio}
        
        if self.format =="prompt":
            prompt += data_str
            prompt += instruction
            return {"prompt": prompt,  "ground_truth_data": relation_matrix, \
                     "context": self.context, "constraint": self.constraint, "data_str": data_str, \
                       "executor_variables":{"VAL": self.data,"RATIO": relation_ratio, "constraint_str": constriant_description},\
                         "output_requirement": output_requirement}
        elif self.format == "df":
            data_str = prompt + data_str + instruction
            prompt += instruction
            return {"prompt": prompt, "output_requirement": output_requirement,"ground_truth_data": relation_matrix,  "context": self.context,\
                    "constraint": self.constraint, "data_str": data_str, "executor_variables":{"VAL": self.data,"RATIO": relation_ratio, "constraint_str": constriant_description}}
        elif self.format == "json":
            my_dict_serializable = {key: list(value) for key, value in zip(all_vars, self.data.T)}
            data = json.dumps(my_dict_serializable)
            return {"prompt": prompt, "ground_truth_data": relation_matrix, 'instruction': instruction, "context": self.context,\
                          "constraint": self.constraint, "data_str": data_str, "output_requirement": output_requirement,
                       "executor_variables":{"VAL": self.data,"RATIO": relation_ratio, "constraint_str": constriant_description}}   

    def get_input_datas(self):
        assert "data_idx" in self.context
        self.data_idx = self.context['data_idx']
        # self.data = pd.read_csv(self.input_data_paths[loc],index_col=0) 
        # Read the JSON file
        with open("synthetic_data_results.json", 'r') as file:
            json_data = json.load(file)
        # loc = np.random.randint(0, len(json_data))
        self.json_data = json_data[self.data_idx]
    
    def create_context_constraint(self):
        with open('synthetic_data_results.json', 'r') as file:
            json_data = json.load(file)
        loc = np.random.randint(0, len(json_data))
        self.json_data = json_data[loc]
        return


class Causal_Knowledge_Question_Generator(Causal_Relation_Question_Generator):
    def __init__(self, input_data_paths: list, context: dict, constraint: dict, format: str, self_generate=False,train_flag = False):
        '''
        This function generates a question for the causal relation prediction task.
        input_data_paths: list of input data paths
        '''
        super().__init__(input_data_paths, context, constraint, format, self_generate,train_flag)

    def generate(self):
        if self.self_generate:
            self.create_context_constraint()
        else:
            self.get_input_datas()
        
        relation_matrix = np.array(self.json_data['relation_matrix'])
        data_path = self.json_data['data_path']
        self.data = pd.read_csv(data_path, index_col=0).round(3)

        # Randomly select at least 10% of the data
        num_rows_to_select = max(1, int(0.1 * len(self.data)))
        start_idx = np.random.randint(0, len(self.data) - num_rows_to_select + 1)
        random_segment = self.data.iloc[start_idx : start_idx + num_rows_to_select]
        self.data = random_segment

        all_vars = self.data.columns.tolist()
        influence_vars_str = ", ".join(all_vars)
        self.data = self.data[all_vars].values

        data_str = ""
        for i in range(len(all_vars)):
            data_str += f"The historical {all_vars[i]} data for the past time steps is: ["
            data_str += ", ".join([str(round(x, 2)) for x in self.data[:, i]]) + "]. "

        prompt = f"I have historical {influence_vars_str} data and want to get the causal relationship between each pair of the variables. "

        # Randomly select a pair of related variables from the relation matrix
        related_pairs = np.argwhere(relation_matrix == 1)
        if len(related_pairs) > 0:
            selected_pair = related_pairs[np.random.randint(len(related_pairs))]
            var1, var2 = all_vars[selected_pair[0]], all_vars[selected_pair[1]]
            domain_knowledge = f"Among variables {','.join(all_vars)}, I know that {var1} influences {var2}. "
        else:
            domain_knowledge = "I do not have any specific relationship knowledge. "
        
        prompt += domain_knowledge

        instruction = f"Consider the potential influence of each variable on the others in this variable list: {all_vars}. " + \
                      f"Please provide a 2D numpy matrix with binary values to indicate whether each pair of variables has a relationship. The data for all variables are stored in variable VAL. "

        output_requirement = "Answer: \n prediction = np.array([[...], [...], ...],dtype=np.float64)\n prediction.shape = (number of variables, number of variables)\n Please return a 2D numpy matrix with 1 indicating a relationship and 0 indicating no relationship."

        self.constraint = {"domain_pair":selected_pair}

        if self.format == "prompt":
            prompt += data_str
            prompt += instruction
            return {"prompt": prompt, "ground_truth_data": relation_matrix, \
                    "context": self.context, "constraint": self.constraint, "data_str": data_str, \
                    "executor_variables": {"VAL": self.data, "constraint_str": domain_knowledge},
                    "output_requirement": output_requirement}
        elif self.format == "df":
            data_str = prompt + data_str + instruction
            prompt += instruction
            return {"prompt": prompt, "output_requirement": output_requirement, "ground_truth_data": relation_matrix, \
                    "context": self.context, "constraint": self.constraint, "data_str": data_str, \
                    "executor_variables": {"VAL": self.data, "constraint_str": domain_knowledge}}
        elif self.format == "json":
            my_dict_serializable = {key: list(value) for key, value in zip(all_vars, self.data.T)}
            data = json.dumps(my_dict_serializable)
            return {"prompt": prompt, "ground_truth_data": relation_matrix, 'instruction': instruction, "context": self.context, \
                    "constraint": self.constraint, "data_str": data_str, "output_requirement": output_requirement,
                    "executor_variables": {"VAL": self.data, "constraint_str": domain_knowledge}}

    
class Causal_Relation_Question_Evaluator(Evaluator):
    def __init__(self, response: dict, ground_truth_data: np.ndarray, context: dict, constraint: dict):
        '''
        This function evaluates the response for the causal relation prediction task.
        response: dictionary containing the response with key "relation_matrix"
        ground_truth_data: numpy array containing the ground truth relation matrix
        context: dictionary containing the context information (not used in this evaluator)
        constraint: dictionary containing the constraints for the question (not used in this evaluator)
        '''
        super().__init__(response, ground_truth_data, context, constraint)
    
    def evaluate(self):
        try:
            assert np.any(np.isnan(self.response)) == False, "Response contains NaN values"
            response_matrix = self.response
            ground_truth_matrix = self.ground_truth_data

            assert response_matrix.shape == ground_truth_matrix.shape, "Response matrix shape does not match ground truth matrix shape"

            correct_predictions = np.sum(response_matrix == ground_truth_matrix)
            total_predictions = response_matrix.size
            accuracy = correct_predictions / total_predictions
            
            
            incorrect_pairs = []
            for i in range(response_matrix.shape[0]):
                for j in range(response_matrix.shape[1]):
                    if response_matrix[i, j] != ground_truth_matrix[i, j]:
                        incorrect_pairs.append((i, j))

            status = 1 if accuracy == 1.0 else 0
            message = ""

            assert set(response_matrix.flatten()) == {0, 1}, "Response matrix should contain 0 and 1 values only"
            if self.constraint.get("domain_pair", None) is not None:
                domain_pair = self.constraint["domain_pair"]
                if response_matrix[domain_pair[0],domain_pair[1]] != 1:
                    message += f"\nThe predicted relationship for the domain pair {domain_pair} is incorrect. "
                    return {"status": 0, "message": message}
                else:
                    message += f"\nThe predicted relationship for the domain pair {domain_pair} is correct. "

            
            if incorrect_pairs:
                message += "\nIncorrect predictions for pairs:"
                for pair in incorrect_pairs:
                    message += f"\n  ({pair[0]}, {pair[1]}): Predicted {response_matrix[pair[0], pair[1]]}, Actual {ground_truth_matrix[pair[0], pair[1]]}"


            return {"status": 1, "strict_success": status, "accuracy": accuracy, "message": message}
        except Exception as e:
            return {"status": 0, "message": str(e), "error": 1}


class Electricity_Prediction_Question_Generator_single(Question_Generator):
    def __init__(self, input_data_paths: list, context: dict, constraint: dict, format: str, self_generate = False, train_flag = False):
        '''
        This function generates a question for the electricity consumption forecasting task.
        input_data_paths: list of input data paths
        context: dictionary containing the context information, it should contain keys "hist_len", "future_len", "influence_vars", "target_var"
            hist_len: length of historical data; int
            future_len: length of future data; int
            target_var: target variable; string
        constraint: dictionary containing the constraints for the question
            max_load: Maximum allowable system load; float
            min_load: Minimum system load; float
            load_ramp_rate: Maximum allowable load change rate; float
            load_variability_limit: Maximum allowable variability in load over a given period; float
        '''
        super().__init__(input_data_paths, context, constraint, format, self_generate, train_flag)
        self.input_data_paths = input_data_paths
        if len(self.input_data_paths) == 0:
            dir_name = "TS-Reasoning/energy_data/by_geo"
            self.input_data_paths = self.load_paths(dir_name)
        self.possible_case = ["max_load", "min_load", "load_ramp_rate", "load_variability_limit"]
        self.possible_target_var = ["load_power","wind_power"]

    def load_paths(self, dir_name):
        paths = sorted([path for path in os.listdir(dir_name) if path.endswith(".csv")])

        # Group by energy grid
        grid_groups = {}
        for path in paths:
            grid_name = path.split("_")[0]  # Extract grid name (e.g., CAISO, ERCOT, MISO)
            if grid_name not in grid_groups:
                grid_groups[grid_name] = []
            grid_groups[grid_name].append(os.path.join(dir_name, path))

        # Select first or second half based on train_flag
        selected_paths = []
        for grid, grid_paths in grid_groups.items():
            mid = len(grid_paths) // 2
            if self.train_flag:
                selected_paths.extend(grid_paths[:mid])  # First half for training
            else:
                selected_paths.extend(grid_paths[mid:])  # Latter half for testing

        return selected_paths

    def generate(self):
        if self.self_generate:
            self.create_context()
        hist_data, future_data,geolocation = self.get_input_datas()
        assert np.sum(future_data[:,-1]) != 0 # Ensure that future_target_var is not all zeros.
        if self.self_generate:
            self.create_constraint(future_data[:,-1])
        
        
        hist_len = self.context["hist_len"]
        future_len = self.context["future_len"]
        target_var = self.context["target_var"]
        all_vars = target_var
        constriant_key = list(self.constraint.keys())[0]
        constriant_value = self.constraint[constriant_key]
        assert len(self.constraint.keys()) == 1
        assert len(target_var) == 1
        target_var = target_var[0]
        constriant_prompt_map = {"max_load": f"I need to ensure that the maximum allowable system load does not exceed {constriant_value} MW. ",
                            "min_load": f"I require that the system load is maintained above a minimum of {constriant_value} MW. ",
                            "load_ramp_rate": f"I must monitor the load ramp rate to ensure it does not exceed {constriant_value} MW for each time step. ",
                            "load_variability_limit": f"I need to manage the load variability so that it does not exceed {constriant_value} MW over the complete time period (i.e the maximum change in load over the entire period). "
                            }


        prompt = f"I have {target_var} data for the past {hist_len} minutes. "
        constriant_description = constriant_prompt_map[constriant_key]        
        prompt += constriant_description        

        output_requirement = defaultdict(lambda: f"""Answer: \n prediction = np.array([...],dtype=np.float64) \n prediction.shape = (future length)""")
        
        instruction = f" Please give me a forecast for the next {future_len} minutes for {target_var}. Think about what could be relevant covariates that can help forecast {target_var}. Your goal is to make the most accurate forecast as possible, refine prediction result based on the constraint previously described, and return the result as a 1D numpy array. The historical data for {target_var} is saved in variable VAL."
        data_str = ""
        for i in range(len(all_vars)):
            data_str+=f"The historical {all_vars[i]} data for the past {hist_len} minutes is: ["
            data_str += ", ".join([str(round(x,2)) for x in hist_data[:,i]]) + "]. "
        if self.format =="prompt":
            prompt += data_str
            prompt += instruction
            return {"prompt": prompt, "output_requirement": output_requirement[0], "ground_truth_data": future_data[:,-1],
                    "context": self.context, "constraint": self.constraint, "data_str": data_str, "executor_variables":{"VAL": self.input_datas[target_var].values,"N": len(future_data),"geolocation":geolocation, "constraint_str": constriant_description}}
        elif self.format == "df":
            target_var_np = self.input_datas[target_var].values.reshape(-1, 1)
            val = np.hstack((target_var_np))
            df = pd.DataFrame(val, columns=all_vars,index=self.input_datas.index)
            data_str = prompt + data_str + instruction
            prompt += instruction
            return {"prompt": prompt, "output_requirement": output_requirement[0],  "ground_truth_data": future_data[:,-1], 
                    "context": self.context, "constraint": self.constraint, "data_str": data_str,
                      "executor_variables":{"VAL": df,"N": len(future_data),"geolocation":geolocation,"future_timestamp":self.all_data.index[-future_len:], "constraint_str": constriant_description}}
        elif self.format == "json":
            my_dict_serializable = {key: list(value) for key, value in zip(all_vars, hist_data.T)}
            data = json.dumps(my_dict_serializable)
            return {"prompt": prompt, "output_requirement": output_requirement[0], "data": data, "ground_truth_data": future_data[:,-1], 
                     "context": self.context, "constraint": self.constraint, "data_str": data_str, "executor_variables":{"VAL": self.input_datas[target_var].values,"N": len(future_data),"geolocation":geolocation, "constraint_str": constriant_description}}


    def get_input_datas(self):
        if self.context.get("target_var", None) is None:
            target_var = np.random.choice(self.possible_target_var, 1, replace=False).tolist()
            self.context["target_var"] = target_var
        else:
            target_var = self.context["target_var"]
        loc = np.random.randint(0, len(self.input_data_paths))
        all_data = pd.read_csv(self.input_data_paths[loc],index_col=0).fillna(0)
        while target_var[0] not in all_data.columns:
            target_var = np.random.choice(self.possible_target_var, 1, replace=False).tolist()
            self.context["target_var"] = target_var
        all_vars = target_var
        temp = all_data[all_vars].values[:, -1]
        flag = np.all(np.isclose(temp, 0))
        while flag==True:
            print("in while loop! read a new file")
            loc = np.random.randint(0, len(self.input_data_paths))
            all_data = pd.read_csv(self.input_data_paths[loc],index_col=0).fillna(0)
            while target_var[0] not in all_data.columns:
                target_var = np.random.choice(self.possible_target_var, 1, replace=False).tolist()
                self.context["target_var"] = target_var
                all_vars = target_var  
            temp = all_data[all_vars].values[:, -1]
            flag = np.all(np.isclose(temp, 0))
        geo_location = all_data['location'].values[0]+", "+all_data['state'].values[0]
        hist_len = self.context["hist_len"]
        future_len = self.context["future_len"]
        data = all_data[all_vars].values
        future_target_var = np.zeros((future_len, 1))
        #don't want all zero cases
        daylight_savings = False
        while np.all(np.isclose(future_target_var, 0)) or daylight_savings:
            start = np.random.randint(0, len(data)-hist_len-future_len)
            hist_data = data[start:start+hist_len]
            future_data = data[start+hist_len:start+hist_len+future_len]
            future_target_var = future_data[:,-1]
            self.input_datas = all_data[all_vars][start:start+hist_len]
            daylight_savings = pd.infer_freq(all_data[start:start+hist_len+future_len].index) is None
        self.all_data = all_data[all_vars][start:start+hist_len+future_len]
        return hist_data, future_data, geo_location

    def create_context(self):
        hist_len = np.random.randint(50,200)
        future_len = np.random.randint(10, 90)
        
        self.context = {"hist_len": hist_len, "future_len": future_len}

        # print("context: ", self.context)

    def create_constraint(self, future_tar_data):
        if self.constraint:
            assert len(self.constraint.keys()) == 1
            assert list(self.constraint.keys())[0] in self.possible_case
            chosen_constraints = list(self.constraint.keys())[0]
        else: 
            chosen_constraints = np.random.choice(self.possible_case)
        
        # Ensure future target data satisfies constraints
        if chosen_constraints=="max_load":
            max_load = max(future_tar_data)
            constraint_value =  np.random.uniform(max_load, max_load*1.1)
        elif chosen_constraints=="min_load":
            min_load = min(future_tar_data)
            constraint_value =  np.random.uniform(0.9*min_load, min_load)
        elif chosen_constraints=="load_ramp_rate":
            load_changes = [future_tar_data[i] - future_tar_data[i - 1] for i in range(1, len(future_tar_data))]
            pred_ramp_rate = max([abs(change) for change in load_changes])
            constraint_value =  np.random.uniform(pred_ramp_rate, pred_ramp_rate*1.1)
        elif chosen_constraints=="load_variability_limit":
            forecast_variability = max(future_tar_data) - min(future_tar_data)
            pred_variability_limit = forecast_variability
            constraint_value =  np.random.uniform(pred_variability_limit, pred_variability_limit*1.1)
  
        self.constraint = {chosen_constraints: constraint_value}


class Climate_Anomaly_Question_Generator(Question_Generator):
    def __init__(self, input_data_paths: list, context: dict, format: str, constraint: dict = {}, self_generate = False, train_flag = False):
        '''
        This function generates a question for the climate anomaly detection task.
        input_data_paths: list of input data paths
        context: dictionary containing the context information, it should contain keys "total_len", "num_normal", "target_var"
            length: length of data; int
            num_normal: number of normal samples; int
        '''
        super().__init__(input_data_paths, context, constraint, format, self_generate,train_flag)
        dir_name = "TS-Reasoning/climate_data/extreme"
        self.dir_name = dir_name
        if len(self.input_data_paths) == 0:
            paths = os.listdir(dir_name)
            self.input_data_paths = [path for path in paths if path.endswith(".npz") and "climatology" not in path] #sort data
        self.input_data_paths = sorted(self.input_data_paths,
                    key=lambda x: tuple(map(int, re.match(r"(\d{4})_(\d{1,2})\.npz", x).groups())))
        self.lat = np.load(os.path.join(self.dir_name.replace("extreme","processed"), "lat.npy"))
        self.lon = np.load(os.path.join(self.dir_name.replace("extreme","processed"), "lon.npy"))

    def generate(self):
        if self.self_generate:
            self.create_context()
        data, labels, normal_samples = self.get_input_datas() #data: total_len x 1, labels: total_len x 1, normal_samples: num_normal x total_len x 1
        
        total_len = self.context["total_len"]
        target_var = self.context["target_var"]
        all_vars = target_var
        assert len(target_var) == 1
        target_var = target_var[0]


        prompt = f"I have {target_var} data that spans {total_len} hours. "          

        output_requirement = defaultdict(lambda: f"""Answer: \n prediction = np.array([...],dtype=np.float64) \n prediction.shape = (sequence length)""")
        
        instruction = f" Please tell me whether there are anomalies (extreme weather events) and where are anomalies if present in this sequence. Please return a 1D numpy array with 1 indicating an anomaly and 0 indicating no anomaly. The data is stored in variable VAL and some anomaly-free normal samples are stored in variable NORM_VAL."
        data_str = ""
        for i in range(len(all_vars)):
            data_str+=f"The {all_vars[i]} data is: ["
            data_str += ", ".join([str(round(x,2)) for x in data[:,i]]) + "]. "
        if self.format =="prompt":
            prompt += data_str
            prompt += instruction
            return {"prompt": prompt, "output_requirement": output_requirement[0],  "ground_truth_data": labels,
                     "context": self.context, "constraint": self.constraint, "data_str": data_str, "executor_variables":{"VAL": data,"NORM_VAL": normal_samples}}
        elif self.format == "df":
            data_str = prompt + data_str + instruction
            prompt += instruction
            return {"prompt": prompt, "output_requirement": output_requirement[0],  "ground_truth_data": labels, 
                    "context": self.context, "constraint": self.constraint, "data_str": data_str,
                      "executor_variables":{"VAL": data,"NORM_VAL": normal_samples}}
        elif self.format == "json":
            my_dict_serializable = {key: list(value) for key, value in zip(all_vars, data.T)}
            data = json.dumps(my_dict_serializable)
            return {"prompt": prompt, "output_requirement": output_requirement[0], "data": data, "ground_truth_data": labels, 
                     "context": self.context, "constraint": self.constraint, "data_str": data_str, "executor_variables":{"VAL": data,"NORM_VAL": normal_samples}}


    def get_input_datas(self):
        total_len = self.context["total_len"]
        target_var = self.context["target_var"][0]

        loc = np.random.randint(160, len(self.input_data_paths)) #save the first year for guaranteed historical data
        # print("this file path: ", self.input_data_paths[loc])
        start,end = get_temporal_span(self.input_data_paths[loc])
        data = np.load(os.path.join(self.dir_name, self.input_data_paths[loc]))
        all_data = data[target_var]
        mask = data["2m_temperature_extreme_mask"]
        time_range = pd.date_range(start, end, freq='H')
        while True:
            #choose a grid id that contains anomaly
            lat_index = np.random.randint(0, len(self.lat))
            lon_index = np.random.randint(0, len(self.lon))
            grid_id = (lat_index, lon_index)
            this_mask = mask[:,:,grid_id[0],grid_id[1]] #seq_len x 1
            seq_len = this_mask.shape[0]
            if np.any(this_mask):
                begin = np.random.randint(0, seq_len-total_len)
                if np.any(this_mask[begin:begin+total_len]):
                    break
        all_data = all_data[begin:begin+total_len,:,grid_id[0],grid_id[1]] #total_len x 1
        this_mask = this_mask[begin:begin+total_len] #total_len x 1
        this_mask = np.squeeze(this_mask) #total_len
        time_range = time_range[begin:begin+total_len] 
        delta_lon = self.lon[1] - self.lon[0]
        geolocation = (self.lat[lat_index], self.lon[lon_index]+delta_lon/2) #centroid of the grid cell

        # get some normal data for the same grid cell from previous time steps
        normal_samples = []
        num_normal = self.context["num_normal"]
        possible_files = list(range(loc%16,loc-16,16))
        random.shuffle(possible_files)
        for i in range(len(possible_files)):
            file_loc = possible_files[i]
            # print("file path: ", self.input_data_paths[file_loc])
            data = np.load(os.path.join(self.dir_name, self.input_data_paths[file_loc]))
            all_data_cur = data[target_var]
            mask_cur = data["2m_temperature_extreme_mask"]
            mask_cur = mask_cur[:,:,grid_id[0],grid_id[1]]
            seq_len = mask_cur.shape[0]
            if not np.any(mask_cur[begin:begin+total_len]):
                # print(np.sum(mask_cur[begin:begin+total_len]))
                normal_samples.append(all_data_cur[begin:begin+total_len,:,grid_id[0],grid_id[1]])
            if len(normal_samples) == num_normal:
                break
        normal_samples = np.array(normal_samples)
        return all_data, this_mask, normal_samples


    def create_context(self):
        total_len = np.random.randint(168,336)
        self.context = {"total_len": total_len}
        self.context["target_var"] = ["2m_temperature"]
        if self.context.get("num_normal", None) is None:
            self.context["num_normal"] = 10
        # print("context: ", self.context)

class Anomaly_Evaluator(Evaluator):
    def __init__(self, response: np.ndarray, ground_truth_data: np.ndarray, context: dict = {}, constraint: dict = {}):
        '''
        This function evaluates the response for the anomaly detection.
        response: numpy array containing the response
        ground_truth_data: numpy array containing the ground truth data
        '''
        super().__init__(response, ground_truth_data, context, constraint)

    def evaluate(self):
        try:
            prediction = self.response
            future_data = self.ground_truth_data
            threshold = self.constraint.get("threshold",1e-6 )
            if len(prediction.shape) == 1:
                prediction = prediction.reshape(-1, 1)
            assert np.any(np.isnan(self.response)) == False, "Response contains NaN values"
            assert len(prediction) == len(future_data), "Prediction and ground truth data have different lengths"
            gt, pred = adjustment(future_data, prediction)
            accuracy = accuracy_score(gt, pred) 
            precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
            print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
                accuracy, precision,
                recall, f_score))
            if f_score >= threshold:
                return {"status": 1, "f1":f_score, "message": f"F1 score is above threshold {threshold}."}
            else:
                return {"status": 0, "message": f"F1 score is below threshold {threshold}, trivial prediction."}
        except Exception as e:
            return {"status": 0, "message": str(e), "error": 1}

        
class Climate_Anomaly_Question_Generator_Multi(Question_Generator):
    def __init__(self, input_data_paths: list, context: dict, format: str, constraint: dict = {}, self_generate = False,train_flag = False):
        '''
        This function generates a question for the climate anomaly detection task.
        input_data_paths: list of input data paths
        context: dictionary containing the context information, it should contain keys "total_len", "num_normal", "target_var"
            length: length of data; int
            num_normal: number of normal samples; int
        '''
        super().__init__(input_data_paths, context, constraint, format, self_generate,train_flag)
        dir_name = "TS-Reasoning/climate_data/extreme"
        self.dir_name = dir_name
        if len(self.input_data_paths) == 0:
            paths = os.listdir(dir_name)
            self.input_data_paths = [path for path in paths if path.endswith(".npz") and "climatology" not in path] #sort data
        self.input_data_paths = sorted(self.input_data_paths,
                    key=lambda x: tuple(map(int, re.match(r"(\d{4})_(\d{1,2})\.npz", x).groups())))
        self.lat = np.load(os.path.join(self.dir_name, "lat.npy"))
        self.lon = np.load(os.path.join(self.dir_name, "lon.npy"))

    def generate(self):
        if self.self_generate:
            self.create_context()
        data, labels, time_range = self.get_input_datas()
        data = np.squeeze(np.array(data)).T # (len, num_ts)
        labels = np.squeeze(np.array(labels)).T # (len, num_ts)
        # print("data shape: ", data.shape, "labels shape: ", labels.shape)

        
        total_len = self.context["total_len"]
        target_var = self.context["target_var"]
        num_ts = self.context['num_ts']
        all_vars = target_var
        assert len(target_var) == 1
        target_var = target_var[0]


        prompt = f"I have {num_ts} time series data for {target_var} of different locations that spans {total_len} hours. "          

        output_requirement = defaultdict(lambda: f"""Answer: \n prediction = np.array([...],dtype=np.float64) \n prediction.shape = (seq length, num_ts)""")
        
        instruction = f"Please tell me whether there are anomalies (extreme weather events) and where are anomalies if present in each time series sequence. Please return a 2D numpy array with 1 indicating an anomaly and 0 indicating no anomaly. The data is stored in variable VAL and anomaly rate for each location is stored in variable ANOMALY_RATE."
        data_str = ""
        data_str+=f"The {all_vars[0]} data for each location is: ["
        for j in range(self.context['num_ts']):
            data_str += '['
            data_str += ", ".join([str(round(x,2)) for x in data[:,j]]) + '],'
        data_str += "]. "
        anom_rate = np.mean(labels, axis=0)
        if self.format =="prompt":
            prompt += data_str
            prompt += instruction
            return {"prompt": prompt, "output_requirement": output_requirement[0],  "ground_truth_data": labels,
                    "context": self.context, "constraint": self.constraint, "data_str": data_str, "executor_variables":{"VAL": data,"ANOMALY_RATE": anom_rate}}
        elif self.format == "df":
            data_str = prompt + data_str + instruction
            data_df = pd.DataFrame(data=data, index=time_range, columns=[f"{i}" for i in range(data.shape[1])])
            prompt += instruction
            return {"prompt": prompt, "output_requirement": output_requirement[0],  "ground_truth_data": labels, 
                    "context": self.context, "constraint": self.constraint, "data_str": data_str,
                      "executor_variables":{"VAL": data_df,"ANOMALY_RATE": anom_rate}}
        elif self.format == "json":
            my_dict_serializable = {key: value.tolist() for key, value in zip(all_vars, np.transpose(data, (2, 0, 1)))}
            data = json.dumps(my_dict_serializable)
            return {"prompt": prompt, "output_requirement": output_requirement[0], "data": data, "ground_truth_data": labels, 
                    "context": self.context, "constraint": self.constraint, "data_str": data_str, "executor_variables":{"VAL": data,"ANOMALY_RATE": anom_rate}}


    def get_input_datas(self):
        
        total_len = self.context["total_len"]
        target_var = self.context["target_var"][0]
        num_ts = self.context['num_ts']

        step = int(total_len/546)

        loc = np.random.randint(0, len(self.input_data_paths))
        # print("this file path: ", self.input_data_paths[loc])
        start,_ = get_temporal_span(self.input_data_paths[loc])
        _, end = get_temporal_span(self.input_data_paths[loc+step])
        time_range = pd.date_range(start, end, freq='H')

        all_datas = [] 
        this_masks = []

        data = np.load(os.path.join(self.dir_name, self.input_data_paths[loc]))
        mask = data["2m_temperature_extreme_mask"]
        all_data = data[target_var]
        for j in range(step):
            data_ = np.load(os.path.join(self.dir_name, self.input_data_paths[loc+j+1]))
            mask_ = data_["2m_temperature_extreme_mask"]
            data_cur = data_[target_var]
            all_data = np.concatenate((all_data, data_cur), axis=0)
            mask = np.concatenate((mask, mask_), axis=0)
                
        exist_grid = set()
        while len(exist_grid) < num_ts:
            #choose a grid id that contains anomaly
            lat_index = np.random.randint(0, len(self.lat)) # 32
            lon_index = np.random.randint(0, len(self.lon)) # 64
            grid_id = (lat_index, lon_index)
            this_mask = mask[:,:,grid_id[0],grid_id[1]] #seq_len x 1
            seq_len = this_mask.shape[0]
            if grid_id not in exist_grid:
                if len(exist_grid) == 0: # first grid
                    begin = np.random.randint(0, seq_len-total_len) # for the first grid location, set a begin timepoint
                    time_range = time_range[begin:begin+total_len]
                exist_grid.add(grid_id)
            else:
                continue
            this_data = all_data[begin:begin+total_len,:,grid_id[0],grid_id[1]] #total_len x 1
            this_mask = this_mask[begin:begin+total_len] #total_len x 1

            all_datas.append(this_data)
            this_masks.append(this_mask)

        return all_datas, this_masks, time_range


    def create_context(self):
        # total_len = np.random.randint(168,336)
        total_len = np.random.randint(512, 1024)
        self.context = {"total_len": total_len}
        self.context["target_var"] = ["2m_temperature"]
        # print("context: ", self.context)
        self.context['num_ts'] = np.random.randint(100, 300)


class Anomaly_Evaluator_Multi(Evaluator):
    def __init__(self, response: np.ndarray, ground_truth_data: np.ndarray, context: dict = {}, constraint: dict = {}):
        '''
        This function evaluates the response for the anomaly detection.
        response: numpy array containing the response
        ground_truth_data: numpy array containing the ground truth data
        '''
        super().__init__(response, ground_truth_data, context, constraint)

    def evaluate(self):
        try:
            prediction = self.response
            future_data = self.ground_truth_data
            threshold = self.constraint.get("threshold", 1e-6)
            assert np.any(np.isnan(self.response)) == False, "Response contains NaN values"
            assert len(prediction) == len(future_data), "Prediction and ground truth data have different lengths"
            gt = []; pred =[]
            for i in range(prediction.shape[1]):
                gt_, pred_ = adjustment(future_data[:,i], prediction[:,i])
                gt.append(gt_); pred.append(pred_)
            gt = np.array(gt).reshape(-1,1); pred = np.array(pred).reshape(-1,1)
            accuracy = accuracy_score(gt, pred)
            precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
            print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
                accuracy, precision,
                recall, f_score))
            if f_score >= threshold:
                return {"status": 1, "f1":f_score, "message": f"F1 score is above threshold {threshold}."}
            else:
                return {"status": 0, "message": f"F1 score is below threshold {threshold}, trivial prediction."}
        except Exception as e:
            return {"status": 0, "message": str(e), "error": 1}

class Electricity_Prediction_Question_Generator_Multi(Electricity_Prediction_Question_Generator):
    def __init__(self, input_data_paths: list, context: dict, constraint: dict, format: str, self_generate = False, train_flag = False):
        '''
        This function generates a question for the electricity consumption forecasting task.
        input_data_paths: list of input data paths
        context: dictionary containing the context information, it should contain keys "hist_len", "future_len", "influence_vars", "target_var"
            hist_len: length of historical data; int
            future_len: length of future data; int
            target_var: target variable; string
        constraint: dictionary containing the constraints for the question
            mape: Mean absolute percentage error; float
            max_load: Maximum allowable system load; float
            min_load: Minimum system load; float
            load_ramp_rate: Maximum allowable load change rate; float
            load_variability_limit: Maximum allowable variability in load over a given period; float
        '''
        super().__init__(input_data_paths, context, constraint, format, self_generate, train_flag)

    def generate(self):
        
        if self.self_generate:
            self.create_context()
        hist_data, future_data = self.get_input_datas() # (num_ts, hist_len), (num_ts, future_len)
        if self.self_generate:
            self.create_constraint(future_data)
        future_data = future_data.T
        hist_data = hist_data.T
        
        hist_len = self.context["hist_len"]
        future_len = self.context["future_len"]
        num_ts = self.context['num_ts']
        target_var = self.context["target_var"]
        all_vars = target_var
        constriant_key = list(self.constraint.keys())[0]
        constriant_value = self.constraint[constriant_key]
        assert len(self.constraint.keys()) == 1
        assert len(target_var) == 1
        target_var = target_var[0]
        constriant_prompt_map = {"max_load": f"I need to ensure that the maximum allowable system load does not exceed {constriant_value} MW for each electric grid zone. ",
                            "min_load": f"I require that the system load is maintained above a minimum of {constriant_value} MW for each electric grid zone. ",
                            "load_ramp_rate": f"I must monitor the load ramp rate to ensure that the maximum change between each timestep does not exceed {constriant_value} MW for each electric grid zone. ",
                            "load_variability_limit": f"I need to manage the load variability so that it does not exceed {constriant_value} MW over the complete time period (i.e the maximum change in load over the entire period) for each electric grid zone. "
                            }

        prompt = f"I have {num_ts} electric grid zones' historical {target_var} data for the past {hist_len} minutes. "
        constriant_description = constriant_prompt_map[constriant_key]        
        prompt += constriant_description 
    

        output_requirement = defaultdict(lambda: f"""Answer: \n prediction = np.array([...],dtype=np.float64) \n prediction.shape = (future length, num_ts)""")
        instruction = f"Please give me a forecast for the next {future_len} minutes for {target_var}. Your goal is to make the most accurate forecast as possible, refine prediction result based on the constraint previously described, and return the result as a 2D numpy array. The historical data for {target_var} are saved in variable VAL."
        data_str = ""
        # for i in range(len(all_vars)):
        data_str+=f"The historical {target_var} data for the past {hist_len} minutes is: ["
        for j in range(self.context['hist_len']): 
            data_str += '['
            data_str += ", ".join([str(round(x,2)) for x in hist_data[j,:]]) + "], "
        data_str += ']. '
        if self.format =="prompt":
            prompt += data_str
            prompt += instruction
            val = np.array([self.input_datas[t][target_var].values for t in range(self.context['num_ts'])])
            return {"prompt": prompt, "output_requirement": output_requirement[0], "ground_truth_data": future_data.T, 
                     "context": self.context, "constraint": self.constraint, "data_str": data_str, 
                    "executor_variables":{"VAL": val,"N": len(future_data[0]),"MULVAL":future_data[:, :, :-1], "constraint_str": constriant_description}}
        elif self.format == "df":
            # target_var_np = np.array([ts[target_var].values for ts in self.input_datas]).T
            df = pd.DataFrame(hist_data, index=self.input_datas.index)
            data_str = prompt + data_str + instruction
            prompt += instruction
            return_dict =  {"prompt": prompt, "output_requirement": output_requirement[0], "ground_truth_data": future_data, 
                    "context": self.context, "constraint": self.constraint, "data_str": data_str,
                      "executor_variables":{"VAL": df,"N": len(future_data), "constraint_str": constriant_description}}
            return return_dict
            
        elif self.format == "json":
            # my_dict_serializable = {key: list(value) for key, value in zip(all_vars, hist_data.T)}
            my_dict_serializable = {}
            for i, all_v in enumerate(all_vars):
                my_dict_serializable[all_v] = hist_data[:,:,i].tolist()
            data = json.dumps(my_dict_serializable)
            val = np.array([self.input_datas[t][target_var].values for t in range(self.context['num_ts'])])
            return {"prompt": prompt, "output_requirement": output_requirement[0], "data": data, 
                    "ground_truth_data": future_data[:,:,-1], "context": self.context, 
                    "constraint": self.constraint, "data_str": data_str, 
                    "executor_variables":{"VAL": val,"N": len(future_data[0]),"MULVAL":future_data[:, :, :-1], "constraint_str": constriant_description}}

    def get_input_datas(self):
        
        target_var = self.context["target_var"]
        all_vars = target_var
        hist_len = self.context["hist_len"]
        future_len = self.context["future_len"]
        num_ts = self.context["num_ts"]
        
        datas = []
        locs_ = np.random.choice(np.arange(len(self.input_data_paths)), size=num_ts, replace=False)
        for i in range(num_ts):
            loc = locs_[i]
            all_data = pd.read_csv(self.input_data_paths[loc],index_col=0).fillna(0) 
            temp = all_data[all_vars].values[:, -1]
            flag = np.all(np.isclose(temp, 0, atol=1e-6))
            while (flag==True):
                print(f"in while loop for file {self.input_data_paths[loc]}! read a new file")
                loc = np.random.randint(0, len(self.input_data_paths))
                while loc in locs_: loc = np.random.randint(0, len(self.input_data_paths))
                all_data = pd.read_csv(self.input_data_paths[loc],index_col=0).fillna(0)
                temp = all_data[all_vars].values[:, -1]
                flag = np.all(np.isclose(temp, 0, atol=1e-6))
            data = all_data[all_vars].values
            locs_[i] = loc
            datas.append(data)
        datas = np.squeeze(np.array(datas)) # (num_ts, seq_len)
        def year_check(h,f):
            h_time = pd.to_datetime(h.index)
            f_time = pd.to_datetime(f.index)
            if (h_time.year.nunique() != 1) or (f_time.year.nunique() != 1):
                return False
            if h_time.year.unique()[0] != f_time.year.unique()[0]:
                return False
            return True
        future_target_var = np.zeros((num_ts, future_len))
        start = np.random.randint(0, datas.shape[1]-hist_len-future_len)
        while (np.any(np.isclose(future_target_var,0,atol=1e-6),axis=1).sum()>5) or (not year_check(all_data[start:start+hist_len], all_data[start+hist_len:start+hist_len+future_len])):
            print("rechoosing start timestamp")
            start = np.random.randint(0, datas.shape[1]-hist_len-future_len)
            hist_data = datas[:, start:start+hist_len]
            future_data = datas[:, start+hist_len:start+hist_len+future_len]
            future_target_var = future_data
        #mask along num_ts axis if it contains zero
        mask = ~np.any(np.isclose(future_target_var,0,atol=1e-6),axis=1)
        future_data = future_data[mask]
        hist_data = hist_data[mask]
        self.context["num_ts"] = len(future_data)
        # print("num_ts: ", self.context["num_ts"])

        
        self.input_datas = pd.DataFrame(datas[:, start:start+hist_len].T, index=all_data[start:start+hist_len].index)
        self.all_data = datas[:, start:start+hist_len+future_len]
        self.chosen_file = [self.input_data_paths[i] for i in locs_]
        return hist_data, future_data

    def create_context(self):
        hist_len = np.random.randint(800,1440)
        future_len = np.random.randint(20,60)
        target_var = np.random.choice(self.possible_target_var, 1, replace=False).tolist()

        # context
        self.context = {"hist_len": hist_len, "future_len": future_len}
        self.context["target_var"] = target_var
        self.context['num_ts'] = np.random.randint(10, 20)

        # print("context: ", self.context)

    def create_constraint(self, future_tar_data):
        # print(future_tar_data.shape) # (num_ts, future_len)
        if self.constraint:
            assert len(self.constraint.keys()) == 1
            assert list(self.constraint.keys())[0] in self.possible_case
            chosen_constraints = list(self.constraint.keys())[0]
        else: 
            chosen_constraints = np.random.choice(self.possible_case)
        
        # Ensure future target data satisfies constraints
        if chosen_constraints=="max_load":
            max_load = max([max(future_tar_data[i]) for i in range(len(future_tar_data))])
            constraint_value =  np.random.uniform(max_load, max_load*1.1)
        elif chosen_constraints=="min_load":
            min_load = min([min(future_tar_data[i]) for i in range(len(future_tar_data))])
            constraint_value =  np.random.uniform(0.9* min_load, min_load)
        elif chosen_constraints=="load_ramp_rate":
            load_changes = []
            for j in range(len(future_tar_data)):
                load_changes.extend([future_tar_data[j, i] - future_tar_data[j, i - 1] for i in range(1, len(future_tar_data[j]))])
            pred_ramp_rate = max([abs(change) for change in load_changes])
            constraint_value =  np.random.uniform(pred_ramp_rate, pred_ramp_rate*1.1)
        elif chosen_constraints=="load_variability_limit":
            forecast_variability = []
            for j in range(len(future_tar_data)):
                forecast_variability.append(max(future_tar_data[j]) - min(future_tar_data[j]))
            pred_variability_limit = max(forecast_variability)
            constraint_value =  np.random.uniform(pred_variability_limit, pred_variability_limit*1.1)
  
        self.constraint = {chosen_constraints: constraint_value}


class Electricity_Prediction_Evaluator_Multi(Evaluator):
    def __init__(self, response: dict, ground_truth_data: np.ndarray, context: dict, constraint: dict):
        '''
        This function evaluates the response for the target variable prediction task.
        response: dictionary containing the response with keys target_var and "explanation"
        ground_truth_data: numpy array containing the ground truth data
        context: dictionary containing the context information
            "target_var": target variable
        constraint: dictionary containing the constraints for the question
            mape: Mean absolute percentage error; float
            max_load: Maximum allowable system load; float
            min_load: Minimum system load; float
            load_ramp_rate: Maximum allowable load change rate; float
            load_variability_limit: Maximum allowable variability in load over a given period; float
        '''
        super().__init__(response, ground_truth_data, context, constraint)
    
    def evaluate(self):
        try:
            assert np.any(np.isnan(self.response)) == False, "Response contains NaN values"
            assert len(self.context["target_var"]) == 1 , "Only one target variable is allowed"
            target_var = self.context["target_var"][0]
            
        
            # explanation = self.response["explanation"] # TODO: Future work - how influence_vars influence target_var
            
            # If the MAPE value is not set, it will be assigned a default value of 0.1.
            threshold = self.constraint.get("threshold", 1)
            self.constraint["threshold"] = threshold
            
            assert len(self.constraint.keys()) == 2, "Only one constraint is allowed in addition to the MAPE threshold value"
            assert self.response.shape == self.ground_truth_data.shape, "Prediction and Ground Truth data have different shapes"       
            
            max_load = self.constraint.get("max_load", None)
            min_load = self.constraint.get("min_load", None)
            load_ramp_rate = self.constraint.get("load_ramp_rate", None)
            load_variability_limit = self.constraint.get("load_variability_limit", None)
            
            # compute predicted constraint according to forecast
            pred_max_load = np.max(self.response)
            pred_min_load = np.min(self.response)

            load_changes = 0
            variability = 0
            num_ts = self.response.shape[-1]

            for i in range(num_ts):
                cur_load_changes = max([abs(self.response[j, i] - self.response[j - 1, i]) for j in range(1, len(self.response))])
                load_changes = max(load_changes, cur_load_changes)
                cur_variability = max(self.response[:,i])-min(self.response[:,i])
                variability = max(variability, cur_variability)
            pred_ramp_rate = load_changes
            pred_variability_limit = variability

            forecast = self.response.flatten()
            ground_truth_data = self.ground_truth_data.flatten()
            
            # print(pred_max_load, pred_min_load, pred_ramp_rate, pred_variability_limit)
            
            forecast += 1e-6
            ground_truth_data += 1e-6
            
            mape_calculated = np.mean(np.abs((forecast - ground_truth_data) / ground_truth_data),axis=0)
            mape_calculated = np.mean(mape_calculated)
            if max_load and (pred_max_load-max_load)>1e-4:
                return {"status": 0, "message": f"Predicted load exceeds the maximum load limit of {max_load} MW. ", "mape": mape_calculated}
            elif min_load and (min_load-pred_min_load)>1e-4:
                return {"status": 0, "message": f"Predicted load falls below the minimum load limit of {min_load} MW.", "mape": mape_calculated}
            elif load_ramp_rate is not None and (pred_ramp_rate - load_ramp_rate)>1e-4:
                return {"status": 0, "message": f"Predicted load ramp rate exceeds the maximum allowable ramp rate of {load_ramp_rate} MW/min. ", "mape": mape_calculated}
            elif load_variability_limit is not None and (pred_variability_limit - load_variability_limit)>1e-4:
                return {"status": 0, "message": f"Predicted load variability exceeds the maximum allowable limit of {load_variability_limit} MW. ", "mape": mape_calculated}
            else:
                if mape_calculated > threshold:
                    message  = f"The MAPE value is {mape_calculated}, which exceeds the threshold {threshold}."
                else:
                    message = "All constraints are satisfied and the MAPE value is within the threshold."
                return {"status": int(mape_calculated<=threshold), "message": message, "mape": mape_calculated}
        except Exception as e:
            return {"status": 0, "message": str(e), "error": 1}

    
class Stock_RV_Question_Generator(Question_Generator): #Return and Volatility
    def __init__(self, input_data_paths: list, context: dict, constraint: dict, format: str, self_generate = False, train_flag = False):
        '''
        This function generates a question for the stock investment strategy.
        input_data_paths: list of input data paths
        context: dictionary containing the context information, it should contain keys "seq_len", "weights"
            hist_len: length of historical data; int
            future_len: length of future data; int
            var: target variable; string, can be "Close", "Open", "High", "Low"
            target_metric: target metric; string, can be "return", "volatility" #annualized return and volatility
        constraint: dictionary containing the constraints for the question
        '''
        super().__init__(input_data_paths, context, constraint, format, self_generate, train_flag)
        if len(self.input_data_paths) == 0:
            dir_name = "TS-Reasoning/day_yahoo"
            paths = os.listdir(dir_name)
            self.input_data_paths = [os.path.join(dir_name, path) for path in paths if path.endswith(".csv")]
            self.input_data_paths = sorted(self.input_data_paths)
            if self.train_flag:
                self.input_data_paths = self.input_data_paths[:len(self.input_data_paths)//2]
            else:
                self.input_data_paths = self.input_data_paths[len(self.input_data_paths)//2:]
        self.possible_case = ["annualized return", "annualized volatility","maximum drawdown","calmar ratio", "sortino ratio", "sharpe ratio"]

    def generate(self):
        if self.self_generate:
            selected_paths = self.create_context_constraint()
        else:
            selected_paths = self.input_data_paths
        cur_data = self.get_input_datas(selected_paths)

        if self.constraint["target_metric"] == "annualized return":
            num_days = (cur_data.index[-1] - cur_data.index[0]).days+1
            cur_val = cur_data.mul(self.context["weights"]).sum(axis=1).values
            end_metric = (cur_val[-1]/cur_val[0])** (365/num_days) - 1
        elif self.constraint["target_metric"] == "annualized volatility":
            returns = cur_data.pct_change().dropna()
            returns = returns.mul(self.context["weights"]).sum(axis=1).values
            end_metric = returns.std() * np.sqrt(252)
        elif self.constraint["target_metric"] == "sharpe ratio":
            risk_free_rate = round(np.random.uniform(0.01, 0.05),2)
            returns = cur_data.pct_change().dropna()
            returns = returns.mul(self.context["weights"]).sum(axis=1).values
            excess_returns = returns - risk_free_rate
            # print(np.mean(excess_returns), np.std(excess_returns))
            end_metric = np.mean(excess_returns) / np.std(excess_returns) #* np.sqrt(252)
        elif self.constraint["target_metric"] == "sortino ratio":
            risk_free_rate = round(np.random.uniform(0.01, 0.05),2)
            returns = cur_data.pct_change().dropna()
            returns = returns.mul(self.context["weights"]).sum(axis=1).values
            excess_returns = returns - risk_free_rate
            negative_returns = excess_returns[excess_returns < 0] #only consider downside volatility
            # print(np.mean(excess_returns), np.std(negative_returns), np.std(excess_returns))
            end_metric = np.mean(excess_returns) / np.std(negative_returns) #* np.sqrt(252)
        elif self.constraint["target_metric"] == "maximum drawdown":
            prices = cur_data.mul(self.context["weights"]).sum(axis=1).values
            peak = np.maximum.accumulate(prices)
            drawdown = (prices - peak) / peak
            max_drawdown = np.min(drawdown)
            end_metric = abs(max_drawdown)
        elif self.constraint["target_metric"] == "calmar ratio":
            num_days = (cur_data.index[-1] - cur_data.index[0]).days+1
            cur_val = cur_data.mul(self.context["weights"]).sum(axis=1).values
            annualized_return = (cur_val[-1]/cur_val[0])** (365/num_days) - 1
            peak = np.maximum.accumulate(cur_val)
            drawdown = (cur_val - peak) / peak
            max_drawdown = abs(np.min(drawdown))
            end_metric = annualized_return / max_drawdown
        else:
            raise ValueError("Invalid target metric")
    
        num_ts = len(self.context["names"])
        name_str = ", ".join(self.context["names"])
        names = self.context["names"]

        weights_str = ", ".join([str(round(x,3)) for x in self.context["weights"]])

        prompt = f"I have {num_ts} historical closing price data for the following stocks: {name_str}. The stocks are held with the following weights:[{weights_str}]. "
        instruction = f"Please calculate the {self.constraint['target_metric']} for this investment portfolio. "
        if self.constraint["target_metric"] == "sharpe ratio" or self.constraint["target_metric"] == "sortino ratio":
            prompt += f"Assume the risk-free rate is {risk_free_rate}. "
        prompt += "The stock data is stored in variable VAL."
        output_requirement = """Answer: \n prediction: float"""
        data_str = ""
        for i in range(len(names)):
            name = names[i]
            data_str+= f"The historical stock value data of {name} for the past {len(cur_data)} trading {self.context['resolution']}s is: ["
            data_str += ", ".join([str(round(x,2)) for x in cur_data.values[:,i]]) + "]. "
        if self.format =="prompt":
            prompt += data_str
            prompt += instruction
            return {"prompt": prompt, "output_requirement": output_requirement, "ground_truth_data": end_metric, "context": self.context, "constraint": self.constraint, "data_str": data_str}
        elif self.format == "df":
            data_str = prompt + data_str + instruction
            prompt += instruction
            return {"prompt": prompt, "output_requirement": output_requirement, "ground_truth_data": end_metric, "context": self.context, "constraint": self.constraint, "data_str": data_str, 
                    "executor_variables":{"VAL": cur_data}}
        elif self.format == "json":
            cur_data = cur_data.to_json(orient='index')
            return {"prompt": prompt, "output_requirement": output_requirement, "ground_truth_data": end_metric, "context": self.context, "constraint": self.constraint, "data_str": data_str, 
                    "data":cur_data}
        
    def get_input_datas(self,ps):
        var = self.context.get("var","Close")
        self.input_datas = []
        # slice the last min_len data to ensure same temporal span, all data ends on 2024-09-17 but have various beginning dates
        min_len = np.inf
        for i in range(len(ps)):
            temp = pd.read_csv(ps[i],index_col=0,parse_dates=True)[var]
            if len(temp) < min_len:
                min_len = len(temp)
            self.input_datas.append(temp)
        for i in range(len(self.input_datas)):
            self.input_datas[i] = self.input_datas[i][-min_len:]
        seq_len = self.context["seq_len"]
        stock_names = []
        for i in range(len(self.input_datas)):
            stock_names.append(ps[i].split("/")[-1].split(".")[0])
        if len(self.input_datas)>1:
            self.input_datas = pd.concat(self.input_datas, axis=1)
        else:
            self.input_datas = pd.DataFrame(self.input_datas[0])
        self.input_datas.columns = stock_names
        start = np.random.randint(0, len(self.input_datas) - seq_len)
        cur_data = self.input_datas.iloc[start:start+seq_len]
        self.context["names"] = stock_names
        return cur_data


    def create_context_constraint(self):
        seq_len = np.random.randint(130, 300)
        num_stocks = np.random.randint(1, 5)
        paths = np.random.choice(self.input_data_paths, num_stocks, replace=False)
        weight = np.random.rand(num_stocks)
        weight = weight / weight.sum()
        self.context = {"seq_len": seq_len, "resolution": "day","weights": weight}
        if self.constraint:
            assert len(self.constraint.keys()) == 1
            assert list(self.constraint.keys())[0] in self.possible_case
            target_metric = list(self.constraint.keys())[0]
        else: 
            target_metric = np.random.choice(self.possible_case)
        self.constraint["target_metric"] = target_metric
        return paths



class Stock_IG_Question_Generator(Stock_RV_Question_Generator):
    def __init__(self, input_data_paths: list, context: dict, constraint: dict, format: str, self_generate = False, train_flag = False):
        super().__init__(input_data_paths, context, constraint, format, self_generate, train_flag)
        self.market_data_path = "TS-Reasoning/market_indices/"
        self.possible_case = None
        if len(self.input_data_paths) == 0:
            dir_name = "TS-Reasoning/day_yahoo"
            paths = os.listdir(dir_name)
            self.input_data_paths = [os.path.join(dir_name, path) for path in paths if path.endswith(".csv")]
            self.input_data_paths = sorted(self.input_data_paths)
            if self.train_flag:
                self.input_data_paths = self.input_data_paths[:len(self.input_data_paths)//2]
            else:
                self.input_data_paths = self.input_data_paths[len(self.input_data_paths)//2:]

    def generate(self):
        if self.self_generate:
            selected_paths = self.create_context_constraint()
        else:
            selected_paths = self.input_data_paths
        cur_data,market_data = self.get_input_datas(selected_paths)

        stock_return = cur_data.pct_change().dropna().values
        market_return = market_data.pct_change().dropna().values
        excess_return = stock_return - market_return
        ir = np.mean(excess_return) / np.std(excess_return)
    
        num_ts = len(self.context["names"])
        name_str = ", ".join(self.context["names"])
        names = self.context["names"]

        prompt = f"I have {num_ts} historical closing price data for the following stocks: {name_str}. I also have the {self.context['target_index']} market index data from the same period. "
        instruction = f"Please calculate the information ratio (IR) of this stock asset with respect to the {self.context['target_index']} market index. The stock data is stored in variable VAL and the market index data is stored in variable VALM. "
        output_requirement = """Answer: \n prediction: float"""
        data_str = ""
        for i in range(len(names)):
            name = names[i]
            data_str+= f"The historical stock value data of {name} for the past {len(cur_data)} trading {self.context['resolution']}s is: ["
            data_str += ", ".join([str(round(x,2)) for x in cur_data.values[:,i]]) + "]. "
        data_str+= f"The historical market index data of {self.context['target_index']} for the past {len(cur_data)} trading {self.context['resolution']}s is: ["
        data_str += ", ".join([str(round(x,2)) for x in market_data.values[:,i]]) + "]. "
        if self.format =="prompt":
            prompt += data_str
            prompt += instruction
            return {"prompt": prompt, "output_requirement": output_requirement, "ground_truth_data": ir, "context": self.context, "constraint": self.constraint, "data_str": data_str}
        elif self.format == "df":
            data_str = prompt + data_str + instruction
            prompt += instruction
            return {"prompt": prompt, "output_requirement": output_requirement, "ground_truth_data": ir, "context": self.context, "constraint": self.constraint, "data_str": data_str,
                    "executor_variables":{"VAL": cur_data,"VALM": market_data}}
        elif self.format == "json":
            cur_data = cur_data.to_json(orient='index')
            return {"prompt": prompt, "output_requirement": output_requirement, "ground_truth_data": ir, "context": self.context, "constraint": self.constraint, "data_str": data_str, 
                    "data":cur_data}

    def get_input_datas(self,ps):
        var = self.context.get("var","Close")
        self.input_datas = []
        for i in range(len(ps)):
            self.input_datas.append(pd.read_csv(ps[i],index_col=0,parse_dates=True)[var])
        seq_len = self.context["seq_len"]
        stock_names = []
        for i in range(len(self.input_datas)):
            stock_names.append(ps[i].split("/")[-1].split(".")[0])
        if len(self.input_datas)>1:
            self.input_datas = pd.concat(self.input_datas, axis=1)
        else:
            self.input_datas = pd.DataFrame(self.input_datas[0])
        self.input_datas.columns = stock_names
        market_data = pd.read_csv(self.market_data_path + f"{self.context['target_index']}.csv",index_col=0,parse_dates=True)
        market_data = market_data[[var]]
        earliest_market_data = market_data.index[0]
        self.input_datas.index = pd.to_datetime(self.input_datas.index, utc=True).tz_convert(None)
        self.input_datas.index = self.input_datas.index.date
        #inner join input data and market data
        joined_data = pd.merge(self.input_datas, market_data, left_index=True, right_index=True, how='inner')
        start = np.random.randint(0, len(joined_data) - seq_len)
        cur_data = joined_data.iloc[start:start+seq_len][stock_names]
        market_data = market_data.loc[cur_data.index]
        market_data.columns = [self.context['target_index']]
        self.context["names"] = stock_names
        return cur_data, market_data

    def create_context_constraint(self):
        seq_len = np.random.randint(130, 252)
        paths = np.random.choice(self.input_data_paths, 1, replace=False)
        self.context = {"seq_len": seq_len, "resolution": "day"}
        possible_target_index = os.listdir(self.market_data_path)
        target_index = np.random.choice(possible_target_index)
        self.context["target_index"] = target_index.split("/")[-1].split(".")[0]
        return paths

class Stock_Threshold_Evaluator(Evaluator):
    def __init__(self, response: float, ground_truth_data: np.ndarray, context: dict, constraint: dict):
        '''
        This function evaluates the response for the stock investment strategy.
        response: float containing the response
        ground_truth_data: numpy array containing the ground truth data
        '''
        super().__init__(response, ground_truth_data, context, constraint)

    def evaluate(self):
        try:
            assert np.any(np.isnan(self.response)) == False, "Response contains NaN values"
            #check response is numerical type
            assert isinstance(self.response, (int, float)), "Response is not a numerical value"
            self.response = float(self.response)
            threshold = self.constraint.get("threshold", 0.05)
            diff = abs(self.response - self.ground_truth_data)
            if diff <= threshold:
                return {"status": 1, "message": f"Computed metric is within the error threshold {threshold}","absolute_diff": diff}
            else:
                return {"status": 0, "message": f"Computed metric exceeds the error threshold {threshold}","absolute_diff": diff}
        except Exception as e:
            return {"status": 0, "message": str(e), "error": 1}

class ECG_Anomaly_Question_Generator(Question_Generator):
    def __init__(self, input_data_paths: list, context: dict, constraint: dict, format: str,self_generate=False,train_flag = False):
        '''
        This function generates a question for the ECG anomaly detection task.
        input_data_paths: list of input data paths
        context: dictionary containing the context information, it should contain keys "hist_len", "future_len", "influence_vars", "target_var"
            length: length of data; int
            num_normal: number of normal samples; int
        '''
        super().__init__(input_data_paths, context, constraint, format, self_generate,train_flag)
        dir_name = "TS-Reasoning/ecg_data/mitbh_combine"
        self.dir_name = dir_name
        if len(self.input_data_paths) == 0:
            paths = os.listdir(dir_name)
            self.input_data_paths = [os.path.splitext(os.path.basename(path))[0] for path in paths if path.endswith(".atr")]

    def generate(self):
        if self.self_generate:
            self.create_context()
        data, labels, normal_samples = self.get_input_datas() #data: total_len x 2, labels: total_len, normal_samples: num_normal x total_len x 2

        total_len = self.context["total_len"]
        all_vars = ["ECG1", "ECG2"]
        prompt = f"I have two lead/channel ECG data spanning across {str(data.shape[0])} timesteps that covers about {total_len} cardiac cycles."
        output_requirement = defaultdict(
            lambda: f"""Answer: \n prediction = np.array([...],dtype=np.float64) \n prediction.shape = (sequence length)""")
        instruction = f" Please tell me whether there are anomalies (Arrhythmia events) and where are anomalies if present in this sequence. Please return a 1D numpy array of sequence length timesteps with 1 indicating an anomaly and 0 indicating no anomaly. The data is stored in variable VAL and some anomaly-free normal samples are stored in variable NORM_VAL."
        data_str = ""
        for i in range(len(all_vars)):
            data_str += f"The {all_vars[i]} data is: ["
            data_str += ", ".join([str(round(x, 2)) for x in data[:, i]]) + "]. "

        if self.format == "prompt":
            prompt += data_str
            prompt += instruction
            return {"prompt": prompt, "output_requirement": output_requirement[0], "ground_truth_data": labels,
                     "context": self.context, "constraint": self.constraint,
                    "data_str": data_str, "executor_variables": {"VAL": data, "NORM_VAL": normal_samples}}
        elif self.format == "df":
            data_str = prompt + data_str + instruction
            prompt += instruction
            return {"prompt": prompt, "output_requirement": output_requirement[0], "ground_truth_data": labels,
                    "context": self.context, "constraint": self.constraint, "data_str": data_str,
                    "executor_variables": {"VAL": data, "NORM_VAL": normal_samples}}
        elif self.format == "json":
            my_dict_serializable = {key: list(value) for key, value in zip(all_vars, data.T)}
            data = json.dumps(my_dict_serializable)
            return {"prompt": prompt, "output_requirement": output_requirement[0], "data": data,
                    "ground_truth_data": labels, "context": self.context, "constraint": self.constraint,
                    "data_str": data_str, "executor_variables": {"VAL": data, "NORM_VAL": normal_samples}}

    def get_input_datas(self):

        total_len = self.context["total_len"]
        loc = np.random.randint(0, len(self.input_data_paths))
        record = wfdb.rdrecord(os.path.join(self.dir_name, self.input_data_paths[loc]))
        all_data = record.p_signal
        annotation = wfdb.rdann(os.path.join(self.dir_name, self.input_data_paths[loc]), 'atr')
        label = annotation.symbol
        while True:
            seq_len = len(label)
            begin = np.random.randint(5, seq_len - total_len - 1)
            if any(x != 'N' for x in label[begin:begin + total_len]) and '~' not in label[begin - 5:begin + total_len]:
                break
        all_label = [0] * all_data.shape[0]
        left = (annotation.sample[begin - 1] + annotation.sample[begin]) // 2
        right = (annotation.sample[begin + total_len - 1] + annotation.sample[begin + total_len]) // 2
        all_data = all_data[left:right, :]
        for i in range(total_len):
            if label[begin + i] != 'N':
                left_temp = (annotation.sample[(begin + i - 1)] + annotation.sample[(begin + i)]) // 2
                right_temp = (annotation.sample[(begin + i + 1)] + annotation.sample[(begin + i)]) // 2
                for j in range(left_temp, right_temp):
                    all_label[j] = 1
        all_label = all_label[left:right]

        input_data_paths_new = self.input_data_paths[:loc] + self.input_data_paths[loc + 1:]
        normal_samples = []
        num_normal = self.context["num_normal"]
        possible_files = list(range(len(input_data_paths_new)))
        random.shuffle(possible_files)
        sample_length = len(all_label)
        for i in range(len(possible_files)):
            file_loc = possible_files[i]
            record = wfdb.rdrecord(os.path.join(self.dir_name, input_data_paths_new[file_loc]))
            data = record.p_signal
            annotation = wfdb.rdann(os.path.join(self.dir_name, input_data_paths_new[file_loc]), 'atr')
            label = annotation.symbol
            label_indexes = [annotation.sample[i] for i in range(len(label))]
            for j in range(5, len(label)):
                last_label_index = None
                for k in range(j, len(label_indexes)):
                    if label_indexes[k] >= annotation.sample[j] + sample_length:
                        last_label_index = k
                        break
                if last_label_index is None:
                    break
                if all(x == 'N' for x in label[j:last_label_index+1]) and '~' not in label[j - 5:last_label_index+1]:
                    normal_samples.append(data[annotation.sample[j]:annotation.sample[j] + sample_length, :])
                    break
            if len(normal_samples) == num_normal:
                break

        all_label = np.array(all_label)
        normal_samples = np.array(normal_samples)
        return all_data, all_label, normal_samples

    def create_context(self):
        total_len = np.random.randint(5, 8)
        self.context = {"total_len": total_len}
        if self.context.get("num_normal", None) is None:
            self.context["num_normal"] = 10

class Energy_Anomaly_Question_Generator(Question_Generator):
    def __init__(self, input_data_paths: list, context: dict, constraint: dict, format: str,self_generate=False,train_flag = False):
        super().__init__(input_data_paths, context, constraint, format, self_generate,train_flag)
        self.dir_name = "TS-Reasoning/energy_data/anomaly_data"
        if len(self.input_data_paths) == 0:
            self.input_data_paths = [os.path.join(self.dir_name, path) for path in ["train.csv"]]
    
    def generate(self):
        if self.self_generate:
            self.create_context()
        data,label = self.get_input_datas() #total_len, num_ts
        self.context["num_ts"] = data.shape[1]
        prompt = f"I have {self.context['num_ts']} time series data for energy consumption of different buildings spanning {self.context['total_len']} hours. There might be missing values in each time series."
        instruction = " Please tell me whether there are anomalies (abnormal usage pattern) for each building and where are anomalies if present. Please return a 2D numpy array with 1 indicating an anomaly and 0 indicating no anomaly. The data is stored in variable VAL and anomaly rate for each location is stored in variable ANOMALY_RATE. "         

        output_requirement = defaultdict(lambda: f"""Answer: \n prediction = np.array([...],dtype=np.float64) \n prediction.shape = (seq length, num_ts)""")
        data_str = ""
        for i in range(self.context["num_ts"]):
            data_str += f"The energy consumption data of building id {data.columns[i]} is: ["
            data_str += ", ".join([str(round(x, 2)) for x in data.iloc[:,i]]) + "]. "
        if self.format == "prompt":
            prompt += data_str
            prompt += instruction
            return {"prompt": prompt, "output_requirement": output_requirement[0], "ground_truth_data": label,
                     "context": self.context, "constraint": self.constraint,
                    "data_str": data_str, "executor_variables": {"VAL": data, "ANOMALY_RATE": np.mean(label, axis=0)}}
        elif self.format == "df":
            data_str = prompt + data_str + instruction
            prompt += instruction
            return {"prompt": prompt, "output_requirement": output_requirement[0], "ground_truth_data": label,
                    "context": self.context, "constraint": self.constraint, "data_str": data_str,
                    "executor_variables": {"VAL": data, "ANOMALY_RATE": np.mean(label, axis=0)}}
        elif self.format == "json":
            my_dict_serializable = {key: list(value) for key, value in zip(self.context["building_ids"], data)}
            data = json.dumps(my_dict_serializable)
            return {"prompt": prompt, "output_requirement": output_requirement[0], "data": data,
                    "ground_truth_data": label, "context": self.context, "constraint": self.constraint,
                    "data_str": data_str, "executor_variables": {"VAL": data, "ANOMALY_RATE": np.mean(label, axis=0)}}
        
    def get_input_datas(self):
        
        total_len = self.context["total_len"]
        num_ts = self.context['num_ts']
        file = np.random.choice(self.input_data_paths)
        data = pd.read_csv(file)
        anomaly = data.pivot(index="timestamp", columns="building_id", values="anomaly")
        data = data.pivot(index="timestamp", columns="building_id", values="meter_reading")
        anomaly.index = pd.to_datetime(anomaly.index)
        data.index = pd.to_datetime(data.index)
        all_datas = [] 
        this_masks = []
        selected_building = []
        while len(selected_building) < num_ts:
            building = np.random.randint(0, data.shape[1])
            if building not in selected_building:
                if len(selected_building) == 0: # first building
                    begin = np.random.randint(0, len(data) - total_len) # for the first building, set a begin timepoint
                all_datas.append(data.iloc[begin:begin+total_len,building].values)
                this_masks.append(anomaly.iloc[begin:begin+total_len,building].values)
                selected_building.append(building)
        all_datas = np.array(all_datas).T
        this_masks = np.array(this_masks).T
        # for anomaly label, assum NaN values are 0
        this_masks[np.isnan(this_masks)] = 0
        #note that data contains NaN values
        # check if any column in all_datas is all nan and remove this column if so
        is_nan = np.isnan(all_datas).astype(int).sum(axis=0) == all_datas.shape[1]
        first_is_nan = np.isnan(all_datas[0,:])
        last_is_nan = np.isnan(all_datas[-1,:])
        is_nan = np.logical_or(is_nan, first_is_nan)
        is_nan = np.logical_or(is_nan, last_is_nan)
        all_datas = all_datas[:,~is_nan]
        this_masks = this_masks[:,~is_nan]
        selected_building = np.array(selected_building)[~is_nan]
        selected_building = selected_building.tolist()
        all_datas = pd.DataFrame(all_datas,index=data.index[begin:begin+total_len],columns=selected_building)
        return all_datas, this_masks


    def create_context(self):
        total_len = np.random.randint(336, 512)
        self.context = {"total_len": total_len}
        self.context['num_ts'] = np.random.randint(25,50)

Question_Type_MAP = {
                    "easy_stock": (Easy_Stock_Question_Generator, Easy_Stock_Evaluator),
                     "electricity_prediction": (Electricity_Prediction_Question_Generator, Electricity_Prediction_Evaluator),
                     "electricity_prediction_single": (Electricity_Prediction_Question_Generator_single, Electricity_Prediction_Evaluator),
                     "electricity_prediction_large": (Electricity_Prediction_Question_Generator_Multi, Electricity_Prediction_Evaluator_Multi),
                     "climate_anomaly": (Climate_Anomaly_Question_Generator, Anomaly_Evaluator),
                     "climate_anomaly_large": (Climate_Anomaly_Question_Generator_Multi, Anomaly_Evaluator_Multi),
                     "causal_relation": (Causal_Relation_Question_Generator, Causal_Relation_Question_Evaluator),
                     "causal_knowledge": (Causal_Knowledge_Question_Generator, Causal_Relation_Question_Evaluator),
                     "stock_rv_estimation": (Stock_RV_Question_Generator, Stock_Threshold_Evaluator),
                     "stock_ir_estimation": (Stock_IG_Question_Generator, Stock_Threshold_Evaluator),
                     "ecg_anomaly": (ECG_Anomaly_Question_Generator, Anomaly_Evaluator),
                     "energy_anomaly": (Energy_Anomaly_Question_Generator, Anomaly_Evaluator_Multi),
                     "stock_investment": (Stock_Question_Generator_Single, Stock_Evaluator_Single),}


if __name__ == "__main__":

    data = []
    question_id = 0 
    questions_of_interest = list(Question_Type_MAP.keys())
    sample_num = 1000//len(questions_of_interest)
    for task in questions_of_interest:
        print(task)
        begin = time.time()
        use_class = Question_Type_MAP.get(task)
        generator_class = use_class[0]
        sub_cases = generator_class(input_data_paths=[], context={}, constraint={},format="df",self_generate=True,train_flag=False).possible_case
        print(sub_cases)
        if sub_cases:
            small_sumple_num = sample_num//len(sub_cases)
            for case in sub_cases:
                questions = Parallel(n_jobs=-1)(delayed(generator_class(input_data_paths=[], context={}, constraint={case:True},format="df",self_generate=True,train_flag=False).generate)() for _ in range(small_sumple_num))
                questions = [dict(**q, question_type=task+"-"+case, question_id=question_id + i) for i, q in enumerate(questions)]
                question_id += len(questions)  # Update question_id counter
                data.extend(questions)
        else:
            questions = Parallel(n_jobs=-1)(delayed(generator_class(input_data_paths=[], context={}, constraint={},format="df",self_generate=True,train_flag=False).generate)() for _ in range(sample_num))
            questions = [dict(**q, question_type=task, question_id=question_id + i) for i, q in enumerate(questions)]
            question_id += len(questions)  # Update question_id counter
            data.extend(questions)
        end = time.time()
        print(f"Time taken for {task} is {end-begin}")
    import pickle as pkl
    pkl.dump(data, open("datav6.pkl", "wb"))
    print(len(data))