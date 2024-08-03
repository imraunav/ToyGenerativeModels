from urllib.request import urlretrieve
import os


# helper functions
# -------------------------------------------------------------------------------------------------
def prep_1984(text):
    lines = text.split("\n")
    clipped_lines = lines[37:-8]
    corpus = "\n".join(clipped_lines)
    return corpus


# hyper params
# -------------------------------------------------------------------------------------------------
url = "https://gutenberg.net.au/ebooks01/0100021.txt"
filename = "gpt/1984.txt"
dataset_path = "gpt/dataset.txt"

# download
# -------------------------------------------------------------------------------------------------
if not os.path.exists(filename):
    print("Downloading file...")
    urlretrieve(url, filename)
    print("Done!")
else:
    print(f"Skipping download, file exists at : {filename}")

# corpus
# -------------------------------------------------------------------------------------------------
# build corpus by deleting some parts at the top and the end
with open(filename, mode="r") as f:
    text = f.read()
corpus = prep_1984(text)

with open(dataset_path, mode="w") as f:
    f.write(corpus)
print(f"dataset saved at, {dataset_path}")
