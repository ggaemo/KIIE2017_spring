from sklearn.preprocessing import OneHotEncoder
import urllib.request

def get_data(preprocess):
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data'
    response = urllib.request.urlopen(url)
    data = pd.read_csv(response, header=None)
    target = data.iloc[:,0]
    target_values = list(set(target))
    target = [target_values.index(x) for x in target]
    target = OneHotEncoder().fit_transform(np.expand_dims(target, 1)).toarray()
    data = data.iloc[:,1:]
    if preprocess == 'embed':
        input_data, unique_dict = preprocess_for_embedding(data)
    elif preprocess == 'onehot':
        input_data, unique_dict = preprocess_for_onehot(data)
    else:
        raise Exception()

    return input_data, target, unique_dict

def preprocess_for_embedding(data):
    unique_dict = dict()
    index_counter = 0
    for i in range(data.shape[1]):
        tmp = data.iloc[:, i]
        unique_cat = set(tmp)
        for val in unique_cat:
            tmp.replace(val, index_counter, inplace=True)
            index_counter += 1

        unique_dict[i] = (unique_cat, index_counter)
    return data, unique_dict

def preprocess_for_onehot(data):
    unique_dict = dict()
    for i in range(data.shape[1]):
        tmp = data.iloc[:, i]
        unique_cat = set(tmp)

        for idx, val in enumerate(unique_cat):
            tmp.replace(val, idx, inplace=True)

        # integer_data.loc[:,i] = tmp

        unique_dict[i] = unique_cat
    onehot = OneHotEncoder().fit_transform(data).toarray()
    return onehot, unique_dict