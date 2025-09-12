from pandas import Series
def show_target_distribution(target_list):
    t = Series(target_list)

    for k, v in t.value_counts().items():
        print(f"Class : {k}, Per : {v/len(target_list):.5f}")
