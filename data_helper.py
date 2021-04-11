from pathlib import Path
import pandas as pd
import os
from glob import glob
from collections import defaultdict
import re
import string
from itertools import chain, repeat, islice

#os.chdir('d:\\IG Data\\')
def create_file_paths_dict(data_path):
    file_dict = dict()
    current_dir = data_path
    os.chdir(current_dir)
    all_sub_dir_paths = glob(str(current_dir) + '/*/') #get all the subdirectories in folder
    all_sub_dir_names = [Path(sub_dir).name for sub_dir in all_sub_dir_paths] 
#print(all_sub_dir_names,all_sub_dir_paths)
    for path in all_sub_dir_paths:
        file_dict[Path(path).name] = glob(str(path) + '/*/')
    print('File Paths Saved in Dictionary')
    return file_dict

#Make image and Caption pair of the data set
#Ref: https://stackoverflow.com/questions/26618688/python-iterate-over-a-list-of-files-finding-same-filenames-but-different-exten
def make_paths(mylist,path):
    pair = defaultdict(dict)
    ls = []
    for filename in mylist:
        name, ext = os.path.splitext(filename)
        pair[name][ext] = filename

    text_extentions = set(['.txt'])
    img_extensions = set(['.jpg'])

    for name, files in pair.items():
        files_set = set(files.keys())
        if not files_set & text_extentions:
            continue # no subs
        elif not files_set & img_extensions:
            continue # no movie
        elif len(tuple(files.values()))==2:
            tup = tuple(files.values())
            ls.append({'img':path+tup[0],'caption':path+tup[1]})
    return ls

def create_data_csv(file_dict):
    for dir,dir_paths in file_dict.items():
        files = {}
        for path in dir_paths:
            os.chdir(path)
        #print(os.listdir())
            pairs = make_paths(os.listdir(),path)
        #print(pairs)
            files[Path(path).name] = make_paths(os.listdir(),path)    
        file_dict[dir] = files

    temp = []
    for type_profile,user_dict in file_dict.items():
        for user,data_list in user_dict.items():
            for values in data_list:
                temp.append((type_profile,user,values['img'],values['caption']))
    print('CSV File Created.!')
    print('Contains Four Coulumns:\n')
    print('Type: Type of the Profile || User: username || Image: Full path of image file || Caption: Full path of text file')
    return pd.DataFrame(temp,columns=['Type','User','Image_path','url_path'])

'''Function for cleaning list of words
    @Parameters: List: sequence
    @Return: List:clean List (removing punctuations and numeric values)
'''
def decontracted(phrase):
    # specific
    phrase = re.sub(r'\s+','',phrase)
    #phrase = re.sub(r'<.*?>','', phrase)
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

def pad_infinite(iterable, padding=None):
   return chain(iterable, repeat(padding))

def pad(iterable, size, padding=None):
   return islice(pad_infinite(iterable, padding), size)

def clean_text(sentence):
  decontraction = " ".join([decontracted(word.strip()) for word in sentence])
  clean_regex = re.compile('<.*?>')
  word = re.sub(clean_regex,' ', decontraction)
  #pattern = r"[{}]".format(string.punctuation)
  #pharse = re.sub(pattern,' ',word)
  cleantext = [word for word in word.split() if word]
  cleantext.insert(0,"<SOS>")
  cleantext.insert(len(cleantext),"<EOS>")
  return list(pad(cleantext,10000,'<pad>'))






