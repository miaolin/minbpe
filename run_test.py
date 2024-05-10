import os


if __name__ == "__main__":
    from minbpe.regex import RegexTokenizer, GPT4_SPLIT_PATTERN

    reg_ins = RegexTokenizer(pattern=GPT4_SPLIT_PATTERN)

    # open some text and train a vocab of 512 tokens
    text = open("tests/taylorswift.txt", "r", encoding="utf-8").read()

    # create a directory for models, so we don't pollute the current directory
    os.makedirs("models", exist_ok=True)

    reg_ins.train(text, vocab_size=300, verbose=True)
    prefix = "models/regex_GPT4_pattern"
    reg_ins.save(prefix)






