import pandas as pd

def merge_results(filenames, output_file):
    dfs = []
    rundf = pd.DataFrame
    for filename in filenames:
        df = pd.read_csv(filename,index_col=0)
        dfs.append(df)

    dfsec =  pd.DataFrame(dfs[0])
    for df in dfs[1:]:
        dfsec = dfsec + df
    dfsec = dfsec / len(filenames)
    dfsec.to_csv(output_file)



