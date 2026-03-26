import pandas as pd
import random

file_path = r'c:\Users\nampham\Desktop\library\library_intent_dataset_1000_v3_fixed_final.csv'
df = pd.read_csv(file_path)

# Distribution adjustment
intents = df['intent'].unique().tolist()
target_per_intent = 1000 // len(intents) # ~166.6

final_data = []

# First, collect unique corrected samples per intent
for it in intents:
    subset = df[df['intent'] == it].drop_duplicates(subset=['text'])
    samples = subset.values.tolist()
    
    if len(samples) > target_per_intent:
        samples = random.sample(samples, target_per_intent)
    else:
        # Upsample slightly to fill more
        pass
    
    for s in samples:
        final_data.append({"text": s[0], "intent": s[1]})

# Fill up to 1000 by adding samples to the underrepresented ones
per_intent_count = {it: len([x for x in final_data if x['intent'] == it]) for it in intents}

for it in intents:
    needed = target_per_intent - per_intent_count[it]
    if needed > 0:
        pool = df[df['intent'] == it].values.tolist()
        for _ in range(needed):
            final_data.append(random.choice(pool))

# Add a few more randomly to reach exactly 1000
while len(final_data) < 1000:
    it = random.choice(intents)
    pool = df[df['intent'] == it].values.tolist()
    final_data.append(random.choice(pool))

random.shuffle(final_data)
final_df = pd.DataFrame(final_data)
final_df = final_df.head(1000)

final_df.to_csv(r'c:\Users\nampham\Desktop\library\library_intent_dataset_1000_v3_fixed_final.csv', index=False, encoding='utf-8-sig')

print("Final Rebalanced Distribution (Balanced):")
print(final_df['intent'].value_counts())
