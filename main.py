import Keywords_Functions
import warnings
import pandas as pd
warnings.filterwarnings(action="ignore")

## Preprocessor / Keywords_Extraction Set-up

main_dir = '/home/taejin/Desktop/Project/Keywords_Extraction'
preproc = Keywords_Functions.preprocessor(main_dir)
preproc.update()
key_ext = Keywords_Functions.keywords_extraction(main_dir)
key_ext.update()

## Input data

f = open('test_comments.txt', 'r', encoding='UTF-8')
ko_data = f.read().strip().split('\n')
f.close()

## Data Processing Stage

data = []

for comment in ko_data:
    comment = preproc.ko_basic(comment)
    comment = preproc.ko_more(comment)
    comment = preproc.spacer(comment)
    data.append(comment)

ko_data = data
ko_data = preproc.tag_comment(ko_data)

# raw = []
# unknown = []
# for i,_ in enumerate(range(len(ko_data))):
#     if list(ko_data[:,1][i]):
#         raw.append(data[i])
#         unknown.append(list(ko_data[:,1][i]))
# unknown_inside = {'Unknowns':unknown, 'Raw_comment': raw}
# unknown_list = pd.DataFrame(unknown_inside)
#
# del raw
# del unknown

ko_data = ko_data[:, 0].tolist()

## Keywords_Extraction

print(key_ext.keywords_extract(ko_data, 20, gram = (2,2), d = 0.8))