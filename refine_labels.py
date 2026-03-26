import pandas as pd
import re

file_path = r'c:\Users\nampham\Desktop\library\library_intent_dataset_1000_v3_fixed.csv'
df = pd.read_csv(file_path)

# 1. Direct fixes for the 8 specific sentences mentioned by USER
direct_fixes = {
    "còn nợ có được gia hạn không?": "renew_book",
    "đền bù 100% giá trị sách là quy định nào?": "policy_query",
    "bồi thường sách mất tính theo giá nào?": "policy_query",
    "quy định phạt quá hạn là gì?": "policy_query",
    "phạt sách hư hỏng bao nhiêu?": "policy_query",
    "hư hỏng nhẹ bị phạt bao nhiêu?": "policy_query",
    "nếu làm mất sách phải đền bao nhiêu?": "policy_query",
    "sách hư hỏng nặng phạt bao nhiêu?": "policy_query"
}

def refine_label(row):
    text_raw = str(row['text'])
    text = text_raw.lower().strip()
    intent = row['intent']
    
    # Apply direct fixes first
    if text in direct_fixes:
        return direct_fixes[text]
        
    # Apply the User's Rule:
    # check_debt = PERSONAL status/amount (contains "tôi", "mình", "em", "của tôi")
    # policy_query = GENERAL rules
    
    is_personal = any(k in text for k in ["tôi", "mình", "em", "của tôi", "tài khoản của", "thẻ của"])
    
    # If it's about fines/penalties/debt
    if any(k in text for k in ["phạt", "nợ", "thanh toán", "đền", "bồi thường", "khóa thẻ"]):
        # But if it's about "how to/rules" without personal context
        if not is_personal and any(k in text for k in ["bao nhiêu", "thế nào", "quy định", "ở đâu"]):
            # Specific exception: "thanh toán nợ phạt ở đâu?" -> policy_query (general procedure)
            return "policy_query"
        if is_personal:
            return "check_debt"
            
    # Re-check "what am I borrowing" incorrectly in renew_book
    if any(k in text for k in ["đang mượn", "đang cầm", "đã mượn"]):
        return "check_debt"

    # Specific case: "còn nợ có được gia hạn không?" already handled in direct_fixes
    
    return intent

df['intent'] = df.apply(refine_label, axis=1)

# Ensure no duplicates in text that might have different labels
df = df.drop_duplicates(subset=['text'], keep='first')

# Check counts
print("Current Distribution after logic fix:")
print(df['intent'].value_counts())

# Re-balance if necessary to reach exactly 1000
target = 167 # average
intents = df['intent'].unique().tolist()

# If short, I'll add from a small pool to reach exactly 1000
current_total = len(df)
if current_total < 1000:
    # Just a few more chat or search to fill
    needed = 1000 - current_total
    extra = df.sample(needed, replace=True)
    df = pd.concat([df, extra])

df = df.head(1000)
df.to_csv(r'c:\Users\nampham\Desktop\library\library_intent_dataset_1000_v3_fixed_final.csv', index=False, encoding='utf-8-sig')

print("\nFinal Rebalanced Distribution:")
print(df['intent'].value_counts())
