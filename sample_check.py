import pandas as pd

file_path = r'c:\Users\nampham\Desktop\library\library_intent_dataset_1000_v3.csv'
df = pd.read_csv(file_path)

with open(r'c:\Users\nampham\Desktop\library\sample_report.txt', 'w', encoding='utf-8') as f:
    f.write("Random samples for each intent:\n")
    for intent in df['intent'].unique():
        f.write(f"\nIntent: {intent}\n")
        samples = df[df['intent'] == intent]['text'].sample(10).tolist()
        for s in samples:
            f.write(f"- {s}\n")
print("Sample report generated.")
