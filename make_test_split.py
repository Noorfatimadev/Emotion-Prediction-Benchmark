import pandas as pd
from sklearn.model_selection import train_test_split

# settings
INPUT_FILE = "emo_dataset.tsv"
OUTPUT_TEST = "emo_test.tsv"
LABEL_COL = "normalized"
TEST_SIZE = 0.15
SEED = 13
N_BINS = 10



# load data
df = pd.read_csv(INPUT_FILE, sep="\t")

# stratification for regression
ranks = df[LABEL_COL].rank(method="average")
bins = pd.qcut(ranks, q=N_BINS, labels=False, duplicates="drop")

# split (keep only test)
_, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=SEED,
    stratify=bins,
)

# save test
test_df.to_csv(OUTPUT_TEST, sep="\t", index=False)

print(f"Saved test set: {len(test_df)} rows -> {OUTPUT_TEST}")