import glob
import pandas as pd

txtfiles = []
for file in glob.glob("*.tsv"):
    txtfiles.append(file)

for i in txtfiles:
	tsv_file = i.lower().replace(" ", "").replace(",", ".")
	out_name = tsv_file.replace(".tsv", ".csv")
	# tsv_file='title.crew.tsv'
	csv_table = pd.read_table(tsv_file, sep='\t',
							  dtype={"isAdult": object, "startYear": object, "isOriginalTitle": object})
	csv_table.to_csv(out_name, index=False, encoding='utf-8-sig')


