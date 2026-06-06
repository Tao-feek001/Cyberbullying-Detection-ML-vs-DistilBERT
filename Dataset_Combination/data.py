import pandas as pd
import os

# Load datasets
df1 = pd.read_csv("cyberbullying_tweets.csv")
df1 = df1.rename(columns={'tweet_text':'text','cyberbullying_type':'label_text'})
print("Cyberbullying dataset shape:", df1.shape)

df2 = pd.read_csv("labeled_data.csv")
df2 = df2.rename(columns={'tweet':'text','class':'label_raw'})
print("Hate Speech dataset shape:", df2.shape)

df3 = pd.read_csv("train.csv")
df3 = df3.rename(columns={'comment_text':'text','toxic':'label_raw'})
print("Toxic Comment dataset shape:", df3.shape)

# Combine
df1 = df1[['text', 'label_text']]
df2 = df2[['text', 'label_raw']]
df3 = df3[['text', 'label_raw']]

combined = pd.concat([df1, df2, df3], ignore_index=True)

# ✅ Merge label columns
combined['label'] = combined['label_text'].combine_first(combined['label_raw'])
combined = combined[['text', 'label']].drop_duplicates(subset=['text']).dropna()

os.makedirs("output", exist_ok=True)
save_path = "output/combined_raw_english_datasets.csv"
combined.to_csv(save_path, index=False)

print("\n✅ Combined dataset shape:", combined.shape)
print(f"💾 Saved to: {save_path}")

# Preview sample safely
if len(combined) > 0:
    print(combined.sample(10))
else:
    print("⚠️ Combined dataset is empty — check column names or file contents.")
