"""Real conversational-turn text from NLTK's nps_chat corpus (Forsyth &
Martell 2007) -- 10,567 posts from age-specific online chat rooms, fetched
via NLTK's own download mechanism (nltk.download), not scraped directly.
Small and informally styled (chat slang, not thoughtful dialogue), so this
is a structural supplement to the drama/dialogue Gutenberg additions, not
a replacement -- it's the one available source of genuinely modern,
turn-based conversational text, distinct from 19th-century literary prose.
"""
import nltk

OUT_PATH = __file__.rsplit("/", 1)[0] + "/../data/text/chat_corpus.txt"


def main():
    nltk.download("nps_chat", quiet=True)
    from nltk.corpus import nps_chat

    lines = [" ".join(post) for post in nps_chat.posts() if post]
    text = "\n".join(lines)
    with open(OUT_PATH, "w") as f:
        f.write(text)
    print(f"chat_corpus.txt: {len(text):,} chars from {len(lines)} posts")


if __name__ == "__main__":
    main()
