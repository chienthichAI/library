import pandas as pd
import random

file_path = r'c:\Users\nampham\Desktop\library\library_intent_dataset_1000_v3_fixed_final.csv'
df = pd.read_csv(file_path)

# Ensure no duplicates in text
df = df.drop_duplicates(subset=['text'])

intents = ["policy_query", "check_stock", "find_book", "general_chat", "renew_book", "check_debt"]
target_per_intent = 1000 // len(intents) # 166

balanced_data = []

for it in intents:
    subset = df[df['intent'] == it]
    samples = subset.values.tolist()
    
    if len(samples) >= target_per_intent:
        # Downsample
        samples = random.sample(samples, target_per_intent)
    else:
        # Upsample if not enough Unique samples
        # Just repeat some until we reach 166
        while len(samples) < target_per_intent:
            samples.append(random.choice(samples))
            
    for s in samples:
        balanced_data.append({"text": s[0], "intent": s[1]})

# Fill remaining gaps to reach exactly 1000
while len(balanced_data) < 1000:
    it = random.choice(intents)
    pool = df[df['intent'] == it].values.tolist()
    s = random.choice(pool)
    balanced_data.append({"text": s[0], "intent": s[1]})

random.shuffle(balanced_data)
final_df = pd.DataFrame(balanced_data)

final_df.to_csv(r'c:\Users\nampham\Desktop\library\library_intent_dataset_1000_v3_fixed_final.csv', index=False, encoding='utf-8-sig')

print("Final Balanced Distribution:")
print(final_df['intent'].value_counts())
print("\nFirst 10 rows:")
print(final_df.head(10))
