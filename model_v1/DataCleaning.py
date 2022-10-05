import pandas as pd
import glob

txtfiles = []
for file in glob.glob("*.csv"):
    txtfiles.append(file)

"""
for i in txtfiles:
    data = pd.read(i)
    while True:
        data1 = data.filter(['stream_cnt'])
        data2 = data.filter(['uniq_users'])
        if data1 == 0 and data2 == 0:
            data.drop()
"""
print(txtfiles)

