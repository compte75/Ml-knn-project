import pandas



def load_data(file_path, target_col="target"):
	if ".csv" in file_path:
		df = pandas.read_csv(file_path)

	elif ".xlsx" in file_path:
		df = pandas.read_excel(file_path)

	else:
		raise Exception("The file path must be a csv or excel file !")

	if target_col not in df.columns:
		raise Exception("The taget column does not exist")

	print(df[target_col].value_counts())
	X = df.drop(columns=[target_col])
	Y = df[target_col]
	X = X.select_dtypes(include=['number'])
	return X, Y 







