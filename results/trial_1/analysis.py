import pandas as pd
import pandas_profiling


df = pd.read_csv("results/first_trial/first_trial_rec.csv")
results_ = pd.read_csv("results/first_trial/admin_neo_v10_feedback.csv")

df.columns = ["target", "recommendation", "image_sim", "nlp_sim", "total_sim"]

df.describe()
df.info()

results_.describe()
results_.info()
results_.isna().sum()

results_.head()

results = pd.merge(
    df.drop_duplicates(),
    results_,
    how="right",
    left_on=["target", "recommendation"],
    right_on=["target", "recommendation"],
)


profile = results.profile_report()
profile.to_file(output_file="output.html")
