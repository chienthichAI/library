import os

css_dir = r"d:\Antigravity\Library\library\frontend\src"

replacements = {
    # Replace white transparent borders/overlays with dark transparent ones
    "rgba(255, 255, 255, ": "rgba(0, 0, 0, ",
    
    # Replace specific dark backgrounds
    "rgba(30, 41, 59, 0.4)": "var(--bg-card)",
    "rgba(30, 41, 59, 0.6)": "var(--bg-card)",
    "rgba(30, 41, 59, 0.8)": "var(--bg-card)",
    "rgba(15, 23, 42, 0.8)": "rgba(255, 255, 255, 0.9)",
    "rgba(15, 23, 42, 0.85)": "rgba(255, 255, 255, 0.9)",
    "rgba(15, 23, 42, 0.6)": "rgba(255, 255, 255, 0.8)",
    
    # Text colors
    "color: #fff": "color: var(--text-primary)",
    "color: white": "color: var(--text-primary)",
    
    # Background fixes for specific screens
    "background: #000;": "background: #f8fafc;",
    
    # Gradients
    "linear-gradient(135deg, #fff 0%, #cbd5e1 100%)": "var(--gradient-primary)",
}

for root, dirs, files in os.walk(css_dir):
    for file in files:
        if file.endswith(".css"):
            file_path = os.path.join(root, file)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                
            for old, new in replacements.items():
                content = content.replace(old, new)
                
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

print("CSS replacement done.")
