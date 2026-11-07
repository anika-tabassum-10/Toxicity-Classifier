import os, zipfile

INCLUDE_FILES = ["toxicity_classifier.ipynb", "README.md", "requirements.txt",
                 "main.py", ".gitignore", "github_link.txt"]
INCLUDE_DIRS = ["results"]           
ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".csv", ".json", ".html", ".md"}
MAX_MB = 5

with zipfile.ZipFile("results.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in INCLUDE_FILES:
        if os.path.exists(f):
            z.write(f)
        else:
            print("missing:", f)
    for d in INCLUDE_DIRS:
        for root, _, files in os.walk(d):
            for name in files:
                p = os.path.join(root, name)
                if os.path.splitext(name)[1].lower() in ALLOWED_EXT and os.path.getsize(p) < MAX_MB * 2**20:
                    z.write(p)
print("Created results.zip.")
