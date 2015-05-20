import pandas as pd

lingpy_df = pd.DataFrame.from_csv("lingpy_fscores.csv")
crp_df = pd.DataFrame.from_csv("crp_fscores.csv")
df = pd.concat([lingpy_df, crp_df])
df.to_csv("combined_fscores.csv")
print("Combined F-scores now in combined_fscores.csv")

mean_df = df.mean(axis=1)
mean_df.sort()
ranked_methods = list(mean_df.keys())
for crp_meth, crp_name in zip(("crp_sca","crp_lex"), ("CRP SCA", "CRP LexStat")):
    beaters = ranked_methods[ranked_methods.index(crp_meth):]
    print("%s has a mean F-score of: %f.  It is outperformed on average by %d methods:" % (crp_name, mean_df[crp_meth],len(beaters)))
    print(" < ".join(beaters))
    for method in ranked_methods:
        if "crp" in method:
            continue
        crp_wins = 0
        for data in df.keys():
            col = df.get(data)
            if col[crp_meth] > col[method]:
                crp_wins += 1
        if crp_wins == 0:
            print("%s is outperformed by %s on EVERY dataset! :(" % (crp_name, method))
        else:
            print("%s outperforms %s on %d datasets." % (crp_name, method, crp_wins))
    print("----------------------------------------")
