"""Real bash one-liners from NL2Bash (Lin et al. 2018, arXiv:1802.08979) --
12,607 real shell commands scraped from StackOverflow, paired with expert-
written English descriptions in the original dataset. Fetched via GitHub's
raw content (same "real public dataset, near-zero cost" pattern as
extract_chat_corpus.py's nltk.download and extract_gutenberg_corpus.py's
catalog fetch), not scraped directly. Only the commands themselves
(all.cm) are pulled here -- the genuinely new syntactic domain (flags,
paths, pipes, quoting) not covered by the existing text or code corpora;
the paired English descriptions (all.nl) are just more natural-language
text, already covered by the text domain.
"""
import urllib.request

OUT_PATH = __file__.rsplit("/", 1)[0] + "/../data/terminal/nl2bash_corpus.txt"
URL = "https://raw.githubusercontent.com/TellinaTool/nl2bash/master/data/bash/all.cm"


def main():
    with urllib.request.urlopen(URL, timeout=30) as resp:
        text = resp.read().decode("utf-8", errors="replace")
    with open(OUT_PATH, "w") as f:
        f.write(text)
    n_lines = text.count("\n")
    print(f"nl2bash_corpus.txt: {len(text):,} chars from {n_lines:,} commands")


if __name__ == "__main__":
    main()
