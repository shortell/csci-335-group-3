import re

# functions for preprocessing as needed. or it might just be this.
# @author: selene shen

# No Edge cases. We can work on that later ig.
# for now it might be like the epstein files and may cause random stuff to occ-ur \mao
def preprocesstwt(text):
    # Replace URLs
    text = re.sub(r'http\S+|www\.\S+', '[URL]', text)
    # Remove RT marker
    text = re.sub(r'^RT\s+', '', text)
    # Expand hashtags (remove # but keep word)
    text = re.sub(r'#(\w+)', r'\1', text)
    # Remove emojis / non-ASCII
    text = text.encode('ascii', 'ignore').decode('ascii')
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Lowercase
    text = text.lower()
    return text

# print info and head of a dataframe. just for testing purposes.
def printinfo(dframe, name):
    print(name + " raw info: -------------------------------------")
    dframe.info()
    print("ex: ===========")
    print(dframe.head)
    return