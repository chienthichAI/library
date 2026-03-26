import pandas as pd
import json

file_path = r'c:\Users\nampham\Desktop\library\library_intent_dataset_1000_v3.csv'
try:
    df = pd.read_csv(file_path)
    
    results = {
        "total_rows": int(len(df)),
        "columns": df.columns.tolist(),
        "intent_distribution": df['intent'].value_counts().to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicates": int(df.duplicated(subset=['text']).sum())
    }
    
    # Text consistency check
    text_intent_groups = df.groupby('text')['intent'].nunique()
    inconsistent_text = text_intent_groups[text_intent_groups > 1]
    results["inconsistent_labels"] = []
    if not inconsistent_text.empty:
        for text in inconsistent_text.index:
            results["inconsistent_labels"].append({
                "text": text,
                "intents": df[df['text'] == text]['intent'].unique().tolist()
            })
            
    # Sample length stats
    df['text_len'] = df['text'].apply(lambda x: len(str(x).split()) if pd.notnull(x) else 0)
    results["text_length_stats"] = df['text_len'].describe().to_dict()
    
    # Check for empty strings in 'text'
    results["empty_text_entries"] = int((df['text'].str.strip() == "").sum())

    with open(r'c:\Users\nampham\Desktop\library\analysis_report.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print("Report generated successfully.")
except Exception as e:
    print(f"Error: {e}")
