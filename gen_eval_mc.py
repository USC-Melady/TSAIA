import pandas as pd
import numpy as np
import json
import os
import ast
import re
import random
# from googletrans import Translator
# from langdetect import detect
# from Annotation import trend_detect
from sklearn.linear_model import LinearRegression
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support,accuracy_score

VALI_NUM = 10


class MC_Question_Generator:
    def __init__(self, input_data_paths: list, context: dict, constraint: dict, self_generate = True):
        '''
        format: str, the format of the output, can be "prompt", "df", "json"
                promt: data is in the prompt, data is in the form of string
                df: data is in the form of pandas dataframe and returned separately
                json: data is in the form of json and returned separately
        '''
        self.input_data_paths = input_data_paths
        self.context = context
        self.constraint = constraint
        self.possible_case = None

    def generate(self):
        return NotImplementedError
    
    def multi_generate(self, num):
        '''
        Generate multiple questions
        '''
        for i in range(num):
            yield self.generate()
            if i ==0 and self.self_generate:
                self.self_generate = False




class Stock_VaR_Question_Generator(MC_Question_Generator):
    def __init__(self, input_data_paths: list = None, context: dict= None,constraint: dict = None, self_generate = True):
        '''
        This function generates a question for the stock investment strategy.
        input_data_paths: list of input data paths
        context: dictionary containing the context information, it should contain keys "seq_len", "target_var"
            hist_len: length of historical data; int
            future_len: length of future data; int
            var: target variable; string, can be "Close", "Open", "High", "Low"
        constraint: dictionary containing the constraints for the question
            confidence_level: confidence level for the VaR; float
        '''
        if not self_generate:
            assert input_data_paths is not None
            assert context is not None
            assert constraint is not None
        super().__init__(input_data_paths, context, constraint)
        self.input_data_paths = input_data_paths
        self.self_generate = self_generate
        self.possible_case = ["confidence_level"] 

    def multi_generate(self, num):
        if self.self_generate:
            self.create_context_constraint()

        options = []
        data_info = []
        execution_variables = {}

        portfolio_info_list = []  # Store portfolio info before assigning letters

        for i in range(num):
            self.choose_stock()
            this_portfolio_info = self.generate()
            portfolio_info_list.append(this_portfolio_info)

        # Extract ground truth values (VaR)
        ground_truth_values = [info[1] for info in portfolio_info_list]

        # Find the portfolio with the **lowest VaR**
        best_portfolio_index = ground_truth_values.index(min(ground_truth_values))

        # Randomly shuffle the order of options (so the best portfolio is not always in the same position)
        shuffled_indices = list(range(num))
        random.shuffle(shuffled_indices)  # Randomly permute indices

        # Map shuffled indices to portfolio info
        letter_map = {}  # Maps 'A', 'B', 'C'... to actual portfolio data
        options = []
        data_info = []

        for i, shuffled_index in enumerate(shuffled_indices):
            choice_chr = chr(ord('A') + i)  # Assign A, B, C, etc.
            portfolio_info = portfolio_info_list[shuffled_index]

            option_str = f"Option {choice_chr}: {portfolio_info[2]}"  # Portfolio description
            options.append(option_str)
            data_info.append(f"Option {choice_chr}: {portfolio_info[3]}")
            execution_variables[f"VAL_{choice_chr}"] = portfolio_info[0]
            execution_variables[f"WEIGHTS_{choice_chr}"] = portfolio_info[4]

            letter_map[choice_chr] = shuffled_index

        # Determine which letter corresponds to the best portfolio
        answer = chr(ord('A') + shuffled_indices.index(best_portfolio_index))  # Correct letter
        ground_truth_values = [ground_truth_values[i] for i in shuffled_indices]

        self.context["future_len"] -= 1 # when calculating the VaR, we need to use the profit between each day, so we need to reduce the future length by 1
        prompt = (f"I have {num} portfolios to choose from for investment. "
                f"Each portfolio contains {self.context['num_stocks']} stocks' historical price data. "
                f"I want to know which investment portfolio has the lowest value at risk (VaR) over the next {self.context['future_len']} trading {self.context['resolution']}s. "
                f"The options are: [{'    '.join(options)}]. Please choose the option that you think has the lowest VaR. "
                f"Only respond with the letter of the option.")

        whole_data_str = "\n".join(data_info)

        return {
            "prompt": prompt,
            "options": options,
            "answer": answer,
            "data_info": whole_data_str,
            "executor_variables": execution_variables,
            "answer_info":ground_truth_values
        }


    def generate(self):
        hist_data, future_data = self.get_input_datas() # (hist_len, num_ts), (future_len, num_ts)
        future_data = future_data.pct_change().dropna()
        future_data["weighted_sum"] = future_data.mul(self.context["weights"]).sum(axis=1)
        future_data = future_data["weighted_sum"].values
        VaR = -np.percentile(future_data, 100-self.constraint["confidence_level"])
        names = self.context["names"]
        weights_str = ", ".join([str(round(x,3)) for x in self.context["weights"]])
        portfolio_str = f"The portfolio contains the following stocks: {', '.join(names)} weighted by : [{weights_str}]. "
        data_str = ""
        for i in range(len(names)):
            name = names[i]
            data_str+= f"The historical stock value data of {name} for the past {len(hist_data)} {self.freq}s is: ["
            data_str += ", ".join([str(round(x,2)) for x in hist_data.values[:,i]]) + "]. "
        return hist_data, VaR, portfolio_str, data_str, self.context["weights"]
        
    def get_input_datas(self):
        var = self.context.get("var","Close")
        self.input_datas = []
        for i in range(len(self.input_data_paths)):
            self.input_datas.append(pd.read_csv(self.input_data_paths[i],index_col=0,parse_dates=True)[var])
        hist_len = self.context["hist_len"]
        future_len = self.context["future_len"]
        if (self.input_datas[0].index[1] - self.input_datas[0].index[0]).days == 1:
            self.freq = "day"
            total_len = 365
        elif (self.input_datas[0].index[1] - self.input_datas[0].index[0]).seconds == 3600:
            self.freq = "hour"
            total_len = 7*5 #7 trading hours for 5 days
        stock_names = []
        for i in range(len(self.input_datas)):
            self.input_datas[i] = self.input_datas[i][-total_len:]
            stock_names.append(self.input_data_paths[i].split("/")[-1].split(".")[0])
        self.input_datas = pd.concat(self.input_datas, axis=1)
        self.input_datas.columns = stock_names
        self.input_datas.index = pd.to_datetime(self.input_datas.index,utc=True).tz_convert(None)
        if self.freq == "day":
            self.input_datas.index = self.input_datas.index.date
        hist_data = self.input_datas.iloc[self.start:self.start+hist_len]
        future_data = self.input_datas.iloc[self.start+hist_len:self.start+hist_len+future_len]   
        self.context["names"] = stock_names
        self.context["resolution"] = self.freq
        return hist_data, future_data
    
    def choose_stock(self):
        if self.context["resolution"] == "day":
            dir_name = "TS-Reasoning/day_yahoo"
        else:
            dir_name = "TS-Reasoning/hour_yahoo"
        available_stocks = os.listdir(dir_name)
        num_stocks = self.context["num_stocks"]
        paths = np.random.choice(available_stocks, num_stocks, replace=False)   
        random_values = np.random.rand(num_stocks)
        probabilities = random_values / random_values.sum()
        self.context["weights"] = probabilities
        self.input_data_paths = [os.path.join(dir_name, path) for path in paths]

    def create_context_constraint(self):
        possible_resolutions = ["day", "hour"]
        chosen_resolution = np.random.choice(possible_resolutions)
        if chosen_resolution == "day":
            hist_len = np.random.randint(30, 100)
            future_len = np.random.randint(10, 30)
            start = np.random.randint(0, 252-hist_len-future_len)
        else:
            hist_len = np.random.randint(10, 20)
            future_len = np.random.randint(5, 35-hist_len)
            start = np.random.randint(0, 35-hist_len-future_len)
        self.start = start
        confidence_level = np.random.randint(90, 99)

        possible_constraints = {"confidence_level": confidence_level}
        if self.constraint:
            assert len(self.constraint.keys()) == 1
            assert list(self.constraint.keys())[0] in possible_constraints
            chosen_constraints = list(self.constraint.keys())[0]
        else:
            chosen_constraints = np.random.choice(list(possible_constraints.keys()))
        constraint_value = possible_constraints[chosen_constraints]
        num_stocks = np.random.randint(1, 5)
        self.context = {"hist_len": hist_len, "future_len": future_len,  "resolution": chosen_resolution,"num_stocks": num_stocks}
        self.constraint = {chosen_constraints: constraint_value}
        


class Stock_SharpeRatio_Question_Generator(MC_Question_Generator):
    def __init__(self, input_data_paths: list = None, context: dict= None,constraint: dict = None, self_generate = True):
        '''
        This function generates a question for the stock investment strategy.
        input_data_paths: list of input data paths
        context: dictionary containing the context information, it should contain keys "seq_len", "target_var"
            hist_len: length of historical data; int
            future_len: length of future data; int
            var: target variable; string, can be "Close", "Open", "High", "Low"
        constraint: dictionary containing the constraints for the question
            confidence_level: confidence level for the VaR; float
        '''
        if not self_generate:
            assert input_data_paths is not None
            assert context is not None
            assert constraint is not None
        super().__init__(input_data_paths, context, constraint)
        self.input_data_paths = input_data_paths
        self.self_generate = self_generate

    def multi_generate(self, num):
        if self.self_generate:
            self.create_context_constraint()

        options = []
        ground_truth = []
        data_info = []
        execution_variables = {}

        portfolio_info_list = []  # Store portfolio info before assigning letters

        for i in range(num):
            self.choose_stock()
            this_portfolio_info = self.generate()
            portfolio_info_list.append(this_portfolio_info)

        # Extract ground truth values (VaR)
        ground_truth_values = [info[1] for info in portfolio_info_list]

        # Find the portfolio with the **highest Sharpe Ratio**
        best_portfolio_index = ground_truth_values.index(max(ground_truth_values))

        # Randomly shuffle the order of options (so the best portfolio is not always in the same position)
        shuffled_indices = list(range(num))
        random.shuffle(shuffled_indices)  # Randomly permute indices

        # Map shuffled indices to portfolio info
        letter_map = {}  # Maps 'A', 'B', 'C'... to actual portfolio data
        options = []
        data_info = []

        for i, shuffled_index in enumerate(shuffled_indices):
            choice_chr = chr(ord('A') + i)  # Assign A, B, C, etc.
            portfolio_info = portfolio_info_list[shuffled_index]

            option_str = f"Option {choice_chr}: {portfolio_info[2]}"  # Portfolio description
            options.append(option_str)
            data_info.append(f"Option {choice_chr}: {portfolio_info[3]}")
            execution_variables[f"VAL_{choice_chr}"] = portfolio_info[0]
            execution_variables[f"WEIGHTS_{choice_chr}"] = portfolio_info[4]
            execution_variables[f"RiskFreeRate_{choice_chr}"] = portfolio_info[5]


            letter_map[choice_chr] = shuffled_index

        # Determine which letter corresponds to the best portfolio
        answer = chr(ord('A') + shuffled_indices.index(best_portfolio_index))  # Correct letter
        ground_truth_values = [ground_truth_values[i] for i in shuffled_indices]

        self.context["future_len"] -= 1 # when calculating the VaR, we need to use the profit between each day, so we need to reduce the future length by 1
        prompt = (f"I have {num} portfolios to choose from for investment. "
                f"Each portfolio contains {self.context['num_stocks']} stocks' price data for {self.context['future_len']} trading {self.context['resolution']}s and the corresponding risk-free rate from the same period. "
                f"I want to know which investment portfolio has the best sharpe ratio."
                f"The options are: [{'    '.join(options)}]. Please choose the option that you think has the best sharpe ratio. "
                f"Only respond with the letter of the option.")

        whole_data_str = "\n".join(data_info)

        return {
            "prompt": prompt,
            "options": options,
            "answer": answer,
            "data_info": whole_data_str,
            "executor_variables": execution_variables,
            "answer_info":ground_truth_values
        }


    def generate(self):
        risk_free_rate = pd.read_csv("TS-Reasoning/market_indices/risk_free_rate.csv",index_col=0,parse_dates=True)[["Close"]]
        hist_data, future_data = self.get_input_datas() # (hist_len, num_ts), (future_len, num_ts)
        if self.freq == "day":
            risk_free_rate = risk_free_rate.loc[future_data.index[0]:future_data.index[-1]].values
        else:
            risk_free_rate = risk_free_rate.loc[future_data.index[0].date():future_data.index[-1].date()].values
            risk_free_rate = np.repeat(risk_free_rate, future_data.groupby(future_data.index.date).count().values[:,0])
        risk_free_rate = risk_free_rate[1:]*0.01
        data_to_return = future_data.copy()
        future_data = future_data.pct_change().dropna()
        future_data["weighted_sum"] = future_data.mul(self.context["weights"]).sum(axis=1)
        future_data = future_data["weighted_sum"].values
        excess_return = future_data - risk_free_rate
        sharpe_ratio = excess_return.mean() / excess_return.std()
        names = self.context["names"]
        weights_str = ", ".join([str(round(x,3)) for x in self.context["weights"]])
        portfolio_str = f"The portfolio contains the following stocks: {', '.join(names)} weighted by : [{weights_str}]. "
        data_str = ""
        for i in range(len(names)):
            name = names[i]
            data_str+= f"The historical stock value data of {name} for the past {len(hist_data)} {self.freq}s is: ["
            data_str += ", ".join([str(round(x,2)) for x in hist_data.values[:,i]]) + "]. "
        return data_to_return, sharpe_ratio, portfolio_str, data_str, self.context["weights"],risk_free_rate
        
    def get_input_datas(self):
        var = self.context.get("var","Close")
        self.input_datas = []
        for i in range(len(self.input_data_paths)):
            self.input_datas.append(pd.read_csv(self.input_data_paths[i],index_col=0,parse_dates=True)[var])
        hist_len = self.context["hist_len"]
        future_len = self.context["future_len"]
        if (self.input_datas[0].index[1] - self.input_datas[0].index[0]).days == 1:
            self.freq = "day"
            total_len = 252
        elif (self.input_datas[0].index[1] - self.input_datas[0].index[0]).seconds == 3600:
            self.freq = "hour"
            total_len = 7*5 #7 trading hours for 5 days
        stock_names = []
        for i in range(len(self.input_datas)):
            self.input_datas[i] = self.input_datas[i][-total_len:]
            stock_names.append(self.input_data_paths[i].split("/")[-1].split(".")[0])
        self.input_datas = pd.concat(self.input_datas, axis=1)
        self.input_datas.columns = stock_names
        # the dtype of self.input datas index is datetime64[ns], so we need to convert it to datetime64[ns, UTC]
        self.input_datas.index = pd.to_datetime(self.input_datas.index, utc=True).tz_convert(None)
        if self.freq == "day":
            self.input_datas.index = self.input_datas.index.date
        hist_data = self.input_datas.iloc[self.start:self.start+hist_len]
        future_data = self.input_datas.iloc[self.start+hist_len:self.start+hist_len+future_len]   
        self.context["names"] = stock_names
        self.context["resolution"] = self.freq
        return hist_data, future_data
    
    def choose_stock(self):
        if self.context["resolution"] == "day":
            dir_name = "TS-Reasoning/day_yahoo"
        else:
            dir_name = "TS-Reasoning/hour_yahoo"
        available_stocks = os.listdir(dir_name)
        num_stocks = self.context["num_stocks"]
        paths = np.random.choice(available_stocks, num_stocks, replace=False)   
        random_values = np.random.rand(num_stocks)
        probabilities = random_values / random_values.sum()
        self.context["weights"] = probabilities
        self.input_data_paths = [os.path.join(dir_name, path) for path in paths]

    def create_context_constraint(self):
        possible_resolutions = ["day", "hour"]
        chosen_resolution = np.random.choice(possible_resolutions)
        if chosen_resolution == "day":
            hist_len = np.random.randint(30, 100)
            future_len = np.random.randint(10, 30)
            start = np.random.randint(0, 252 - hist_len - future_len)
        else:
            hist_len = np.random.randint(10, 20)
            future_len = np.random.randint(5, 35-hist_len)
            start = np.random.randint(0, 7*5 - hist_len - future_len)
        self.start = start
        num_stocks = np.random.randint(1, 10)
        self.context = {"hist_len": hist_len, "future_len": future_len,  "resolution": chosen_resolution,"num_stocks": num_stocks}
        

class Stock_MarketAB_Question_Generator(MC_Question_Generator):
    def __init__(self, input_data_paths: list = None, context: dict= None,constraint: dict = None, self_generate = True):
        '''
        This function generates a question for the stock investment strategy.
        input_data_paths: list of input data paths
        context: dictionary containing the context information, it should contain keys "seq_len", "target_var"
            hist_len: length of historical data; int
            future_len: length of future data; int
            var: target variable; string, can be "Close", "Open", "High", "Low"
        constraint: dictionary containing the constraints for the question
            confidence_level: confidence level for the VaR; float
        '''
        if not self_generate:
            assert input_data_paths is not None
            assert context is not None
            assert constraint is not None
        super().__init__(input_data_paths, context, constraint)
        self.input_data_paths = input_data_paths
        self.self_generate = self_generate
        self.possible_case = ["alpha","beta"]

    def multi_generate(self, num):
        if self.self_generate:
            self.create_context_constraint()
        self.choose_stock()
        return self.generate()

    def generate(self):
        potential_indices = ["SPX", "COMP", "DJIA", "RUT","W5000"]
        market_name = np.random.choice(potential_indices)
        sp = pd.read_csv(f"TS-Reasoning/market_indices/{market_name}.csv",index_col=0,parse_dates=True)[["Close"]]
        hist_data = self.get_input_datas() # (hist_len, num_ts), (future_len, num_ts)
        sp = sp.loc[hist_data.index[0]:hist_data.index[-1]]
        sp_return = sp.pct_change().dropna()
        hist_return = hist_data.pct_change().dropna()
        lr = LinearRegression()
        lr.fit(sp_return.values.reshape(-1,1), hist_return.values.reshape(-1, 1))
        alpha = lr.intercept_[0]
        beta = lr.coef_[0][0]
        if self.constraint.get("alpha", False):
            options = ["Outperform the market after adjusting for risk", "Underperform the market after adjusting for risk"]#, "Matched market performance after adjusting for risk"]
            if alpha<0:
                answer_idx = 1
            else:
                answer_idx = 0
        else:
            options = ["Less volatile than the market", "More volatile than the market"]#, "Moves in line with the market"]
            if beta<1:
                answer_idx = 0
            else:
                answer_idx = 1
        names = self.context["names"]
        data_str = ""
        for i in range(len(names)):
            name = names[i]
            data_str+= f"The historical stock value data of {name} for the past {len(hist_data)} days is: ["
            data_str += ", ".join([str(round(x,2)) for x in hist_data.values[:,i]]) + "]. "

        #randomly shuffle the order of options (so the best portfolio is not always in the same position)
        shuffled_indices = list(range(len(options)))
        random.shuffle(shuffled_indices)  # Randomly permute indices
        # Map shuffled indices to portfolio info
        choice_options = []
        for i, shuffled_index in enumerate(shuffled_indices):
            choice_chr = chr(ord('A') + i)
            this_choice = options[shuffled_index]
            option_str = f"Option {choice_chr}: {this_choice}"  # Portfolio description
            choice_options.append(option_str)
        final_answer_idx = shuffled_indices.index(answer_idx)
        answer = chr(ord('A') + final_answer_idx)
        execution_variables = {"VAL": hist_data,"market_VAL": sp}
        prompt = f"I'm looking at the {self.context['names'][0]} stock and the market {market_name} index. "
        if self.constraint.get("alpha", False):
            prompt += f"I want to know how the {self.context['names'][0]} stock performs with respect to the market {market_name} index after adjusting for risk. "
        else:
            prompt += f"I want to know how the {self.context['names'][0]} stock performs with respect to the market {market_name} index in terms of volatility. "
        prompt += f"The options are: [{';'.join(choice_options)}]. Please choose the option that you think is the best. "
        prompt += f"Which option is more likely? Only respond with the letter of the option."
        return {
            "prompt": prompt,
            "options": choice_options,
            "answer": answer,
            "data_info": data_str,
            "executor_variables": execution_variables,
            "answer_info": {"alpha": alpha, "beta": beta}
        }
        
    def get_input_datas(self):
        var = self.context.get("var","Close")
        total_len = 252
        self.input_datas=pd.read_csv(self.input_data_paths,index_col=0,parse_dates=True)[[var]][-total_len:]
        hist_len = self.context["hist_len"]        
        stock_names = [self.input_data_paths.split("/")[-1].split(".")[0]]
        self.input_datas.columns = stock_names
        # the dtype of self.input datas index is datetime64[ns], so we need to convert it to datetime64[ns, UTC]
        self.input_datas.index = pd.to_datetime(self.input_datas.index, utc=True).tz_convert(None)
        self.input_datas.index = self.input_datas.index.date
        hist_data = self.input_datas.iloc[self.start:self.start+hist_len] 
        self.context["names"] = stock_names
        return hist_data
    
    def choose_stock(self):
        dir_name = "TS-Reasoning/day_yahoo"
        available_stocks = os.listdir(dir_name)
        paths = np.random.choice(available_stocks, 1, replace=False)   
        self.input_data_paths = os.path.join(dir_name, paths[0])

    def create_context_constraint(self):
        hist_len = np.random.randint(30, 252)
        start = np.random.randint(0, 252 - hist_len)
        self.start = start
        self.context = {"hist_len": hist_len}
        possible_constraints = self.possible_case
        if self.constraint:
            assert len(self.constraint.keys()) == 1
            assert list(self.constraint.keys())[0] in possible_constraints
            chosen_constraints = list(self.constraint.keys())[0]
        else:
            chosen_constraints = np.random.choice(possible_constraints)
        self.constraint = {chosen_constraints: True}

Question_Type_MAP = {
    "VaR": Stock_VaR_Question_Generator,
    "SharpeRatio": Stock_SharpeRatio_Question_Generator,
    "MarketAB": Stock_MarketAB_Question_Generator
}

if __name__ == "__main__":
    import time 
    from joblib import Parallel, delayed
    import pickle as pkl
    data = []
    question_id = 0 
    questions_of_interest = list(Question_Type_MAP.keys())
    sample_num = 50
    for task in questions_of_interest:
        print(task)
        begin = time.time()
        use_class = Question_Type_MAP.get(task)
        generator_class = use_class
        sub_cases = generator_class(input_data_paths=[], context={}, constraint={},self_generate=True).possible_case
        print(sub_cases)
        if sub_cases:
            small_sumple_num = sample_num//len(sub_cases)
            for case in sub_cases:
                questions = Parallel(n_jobs=-1)(delayed(generator_class(input_data_paths=[], context={}, constraint={case:True},self_generate=True).multi_generate)(3) for _ in range(small_sumple_num))
                questions = [dict(**q, question_type=task+"-"+case, question_id=question_id + i) for i, q in enumerate(questions)]
                question_id += len(questions)  # Update question_id counter
                data.extend(questions)
        else:
            questions = Parallel(n_jobs=-1)(delayed(generator_class(input_data_paths=[], context={}, constraint={},self_generate=True).multi_generate)(3) for _ in range(sample_num))
            questions = [dict(**q, question_type=task, question_id=question_id + i) for i, q in enumerate(questions)]
            question_id += len(questions)  # Update question_id counter
            data.extend(questions)
        end = time.time()
        print(f"Time taken for {task} is {end-begin}")
    import pickle as pkl
    pkl.dump(data, open("data_mc_v3.pkl", "wb"))
    print(len(data))