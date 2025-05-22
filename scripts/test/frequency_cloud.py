import argparse
import os
import pandas as pd
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
import string
import json

def clean_word(word):
    """Remove punctuation and preserve original capitalization"""
    return word.strip(string.punctuation)

# Analyze command-line parameters
parser = argparse.ArgumentParser(description="Generate word cloud for each file based on decoded texts.")
parser.add_argument("--work_dir", type=str, required=True, help="Path to the model results.")
parser.add_argument("--model", type=str, required=True, help="Path to the model")
args = parser.parse_args()
model = args.model

# load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model)

work_dir = args.work_dir
detail_dir = os.path.join(work_dir, "details")
target_words = {"But", "Wait", "Let", "Perhaps", "Yet", "Let's"}  
target_words_lower = {word.lower() for word in target_words}  # Lowercase set for matching

# Remove all forms of 'but' from default stop words
stopwords = set(STOPWORDS) - {"but", "But"}  


def work_amc(amc_dir):
    all_list_text = []
    word_count = Counter()
    total_token_length = 0
    with open(amc_dir, "r") as f:
        data_list = json.load(f)
        for data in data_list:
            decoded_text = data["generated_output"]
            total_token_length += len(tokenizer.encode(decoded_text, add_special_tokens=False))
            all_list_text.append(decoded_text)

            # Statistical logic: Keep the original word form but do lowercase matching
            words = decoded_text.split()
            for word in words:
                cleaned_word = clean_word(word)
                if cleaned_word.lower() in target_words_lower:  # Lowercase matching
                    word_count[cleaned_word.lower()] += 1  # Keep the original word form
    
    all_text = " ".join(all_list_text)
    wordcloud = WordCloud(
        collocations=False,
        width=800,
        height=400,
        background_color='white',
        stopwords=stopwords  # Use filtered stop words
    ).generate(all_text)
    
    # Display and save word clouds
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    file_name = "amc"
    plt.title(f"Word Cloud for {file_name}")
    frequency_dir = os.path.join(work_dir, "frequency")
    os.makedirs(frequency_dir, exist_ok=True)
    plt.savefig(os.path.join(frequency_dir, f"{file_name}_wordcloud.png"), bbox_inches='tight')
    plt.close()
    # Save word frequency statistics results
    result_file = os.path.join(frequency_dir, f"{file_name}_frequency.txt")
    with open(result_file, "w") as f:
        total = 0
        for word, count in word_count.items():
            f.write(f"{word}: {count}\n")
            total += count
        f.write(f"Total Word: {total}\n")
        f.write(f"Total Token Length: {total_token_length}\n")
        RWR = total / total_token_length
        f.write(f"RWR: {RWR}\n")
    print("Word cloud and frequency statistics saved for AMC data.")


work_amc(os.path.join(work_dir, "amc_output.json"))

# Traverse all .parquet files in the details folder
for root, dirs, files in os.walk(detail_dir):
    for file in files:
        if file.endswith(".parquet"):
            total_token_length = 0
            file_path = os.path.join(root, file)
            if "codegeneration" in file_path:
                continue
            if "old" in file_path:
                continue
            print("Loading file:", file_path)
            df = pd.read_parquet(file_path, columns=["cont_tokens"])

            word_count = Counter()
            all_list_text = []
            for token_list in df["cont_tokens"]:
                if len(token_list) > 0:
                    token_array = token_list[0]
                    for token_seq in token_array:
                        token_ids = token_seq.tolist()
                        total_token_length += len(token_ids)
                        decoded_text = tokenizer.decode(
                            token_ids, 
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=True
                        ).strip()
                        all_list_text.append(decoded_text)

                        words = decoded_text.split()
                        for word in words:
                            cleaned_word = clean_word(word)
                            if cleaned_word.lower() in target_words_lower:
                                word_count[cleaned_word.lower()] += 1
            
            all_text = " ".join(all_list_text)
            wordcloud = WordCloud(
                collocations=False,
                width=800,
                height=400,
                background_color='white',
                stopwords=stopwords
            ).generate(all_text)
            
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")

            file_name = file_path.split("|")[-2]
            title = f"Word Cloud for {file_name}"
            if "gpqa" in file_name:
                title = f"Saturn-7B Word Cloud for GPQA Diamond"
            plt.title(title)

            frequency_dir = os.path.join(work_dir, "frequency")
            os.makedirs(frequency_dir, exist_ok=True)

            plt.savefig(os.path.join(frequency_dir, f"{file_name}_wordcloud.png"), bbox_inches='tight')
            plt.close()

            result_file = os.path.join(frequency_dir, f"{file_name}_frequency.txt")
            with open(result_file, "w") as f:
                total = 0
                for word, count in word_count.items():
                    f.write(f"{word}: {count}\n")
                    total += count
                f.write(f"Total Word: {total}\n")
                f.write(f"Total Token Length: {total_token_length}\n")
                RWR = total / total_token_length
                f.write(f"RWR: {RWR}\n")