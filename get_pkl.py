import os
import pickle
import pandas as pd

raw_data_dir = 'data/raw_data'
aq_path = os.path.join(raw_data_dir, "analysis_questions.csv")      # analysis_questions_path
mc_path = os.path.join(raw_data_dir, "multiple_choice.csv")        # multiple_choice_path
aq_out_pkl = "data/analysis_questions.pkl"
mc_out_pkl = "data/multiple_choice.pkl"

# get analysis questions
aq_external_cols = ["executor_variables", "ground_truth_data", "context", "constraint"]
aq_df = pd.read_csv(aq_path)
for col in aq_external_cols:
    aq_df[col] = aq_df[col].apply(
        lambda rel: pickle.load(open(os.path.join(raw_data_dir, rel), "rb"))
                     if pd.notna(rel) else None
    )
aq_list = aq_df.to_dict(orient='records')

# get multiple choice
mc_external_cols = ["answer_info", "executor_variables"]
mc_df = pd.read_csv(mc_path)

for col in mc_external_cols:
    mc_df[col] = mc_df[col].apply(
        lambda rel: pickle.load(open(os.path.join(raw_data_dir, rel), "rb"))
                     if pd.notna(rel) else None
    )
mc_list = mc_df.to_dict(orient='records')

with open(aq_out_pkl, 'wb') as f:
    pickle.dump(aq_list, f)
print(f"Successfully get analysis_questions.pkl in {aq_out_pkl}")

with open(mc_out_pkl, 'wb') as f:
    pickle.dump(mc_list, f)
print(f"Successfully get multiple_choice.pkl in {mc_out_pkl}")