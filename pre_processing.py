from fonction import processing, supprimer_replace_nan


def data_clean():
    data = processing()
    data_cleaned = supprimer_replace_nan(data)
    print()
    print("DataFrame Finale".center(50, "="))
    # data_cleaned.to_csv("data1.csv", index=False)
    return data_cleaned
