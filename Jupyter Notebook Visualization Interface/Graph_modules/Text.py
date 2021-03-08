import operator
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import string
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

# Load the file
# file = pd.read_csv(file_path, encoding="ISO-8859-1")

# Now let's plot some graphs

def text_bar_graph(file,n_top_words,word_label_in_dict,label_y_axis="Count",height_in_inches=10,label_x_axis="Words",width_in_inches=10):

    # Print a histogram containing the top N words, and print them and their counts.
    top_n = n_top_words

    # Load the file
    file = file

    # Parse into a single list (from a list of lists)
    content_list = [item for sublist in file[word_label_in_dict] for item in sublist.split()]

    # Remove whitespace so we can concatenate appropriately, and unify case
    content_list_strip = [str.strip().lower() for str in content_list]

    # Concatenate strings into a single string
    content_concat = ' '.join(content_list_strip)

    # Remove punctuation and new lines
    punct = set(string.punctuation)
    unpunct_content = ''.join(x for x in content_concat if x not in punct)

    # Split string into list of strings, again
    word_list = unpunct_content.split()
    for i in word_list:
        if i in STOPWORDS:
            word_list.remove(i)

    # Perform count
    counts_all = Counter(word_list)

    words, count_values = zip(*counts_all.items())

    # Sort both lists by frequency in values (Schwartzian transform) - thanks, http://stackoverflow.com/questions/9543211/sorting-a-list-in-python-using-the-result-from-sorting-another-list
    values_sorted, words_sorted = zip(*sorted(zip(count_values, words), key=operator.itemgetter(0), reverse=True))

    # Top N
    words_sorted_top = words_sorted[0:top_n]
    values_sorted_top = values_sorted[0:top_n]

    # Histogram

    # Make xticklabels comprehensible by matplotlib
    xticklabels = list(words_sorted_top)
    # Remove the single quotes, commas and enclosing square brackets
    xtlabs = [xstr.replace("'", "").replace(",", "").replace("]", "").replace("[", "") for xstr in xticklabels]

    indices = np.arange(len(words_sorted_top))
    width = 1
    fig = plt.figure(figsize = (width_in_inches , height_in_inches))
    fig.suptitle('Word Frequency Histogram, Top {0}'.format(top_n), fontsize=16)
    plt.xlabel(label_x_axis , fontsize=12)
    plt.ylabel(label_y_axis , fontsize=12)
    plt.bar(indices, values_sorted_top, width)
    plt.xticks(indices + width * 0.5, xtlabs, rotation='vertical', fontsize=8)
    plt.show()

def word_cloud_graph(file,n_top_words,word_label_in_dict,mask_path=None,background_color="white",height_in_inches=10,width_in_inches=10):

    # Print a histogram containing the top N words, and print them and their counts.
    top_n = n_top_words

    # Load the file
    file = file
    comment_words = ''
    for i in file[word_label_in_dict]:
        i = str(i)
        separate = i.split()
        for j in range(len(separate)):
            separate[j] = separate[j].lower()

        comment_words += " ".join(separate) + " "

    #Image
    if mask_path is None:
        mask = mask_path
        image_colors = None
    else:
        mask = np.array(Image.open(mask_path))
        image_colors = ImageColorGenerator(mask)

    #Word Cloud
    final_wordcloud = WordCloud(width=800, height=800,
                                background_color=background_color,
                                stopwords=STOPWORDS,
                                min_font_size=10,
                                max_words=n_top_words,
                                mask=mask, mode="RGBA").generate(comment_words)

    plt.figure(figsize=(width_in_inches, height_in_inches))
    plt.imshow(final_wordcloud.recolor(color_func=image_colors), interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)

    plt.show()

def dskc_visualizationtext_graphs(file_path,word_label_in_dict,n_top_words_hist,n_top_words_cloud,mask_path,background_color,label_y_axis, height_in_inches,label_x_axis,width_in_inches):
    text_bar_graph(file_path, n_top_words_hist, word_label_in_dict, label_y_axis, height_in_inches, label_x_axis, width_in_inches)
    word_cloud_graph(file_path, mask_path, background_color, n_top_words_cloud, word_label_in_dict, height_in_inches, width_in_inches)