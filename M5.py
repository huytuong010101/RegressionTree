import numpy as np
import pandas as pd 
import warnings
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error as mape
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

from Node import Node

warnings.filterwarnings("ignore")
np.random.seed(2)

class RegressionTree:
    def __init__(self, k: int=15, min_split: int=4, sd_threshold: float=0.55):
        self.__attrs = []
        self.__target = None
        self.__node = None
        self.__k = k
        self.__min_split = min_split
        self.__sd_threshold = sd_threshold
        
    def calc_node_error(self, node: Node):
        """
        This function use to estimate erorr at curent node
        """
        X, Y = node.dataset[node.model_variable].to_numpy(), node.dataset[self.__target].to_numpy()
        Yhat = node.model.predict(X)
        error = self.calc_error(Y, Yhat, X.shape[1]) 
        return error
    
    def calc_subtree_error(self, node: Node):
        """
        This function use to calc subtree error, following Figure 1 https://researchcommons.waikato.ac.nz/bitstream/handle/10289/1183/uow-cs-wp-1996-23.pdf?sequence=1&isAllowed=y
        """
        if not node.is_leaf():
            l_node, r_node = node.childs
            return (self.calc_subtree_error(l_node) * len(l_node.dataset) + self.calc_subtree_error(r_node) * len(r_node.dataset)) / len(node.dataset)
        else:
            return self.calc_node_error(node)
        
    def create_tree(self, node: Node):
        """
        This function use to construct tree following page 125 https://urdata.net/files/2016_accident_duration_prediction.pdf
        """
        dataset: pd.DataFrame = node.dataset
        # Build regresion model each node
        # TODO: Reimplement Linear regression

        self.simplify_linear_models(node)

        # Stop if have less sample or small SD
        if len(dataset) < self.__min_split or np.std(node.dataset[self.__target].to_numpy()) < self.__sd_threshold*self.__SD:
            print(f">> Reach leaf with {len(node.dataset)} record")
            return
        T = dataset[self.__target].to_numpy()
        # Init sub tree
        max_eer = -np.inf
        left_node = Node(parent=node)
        right_node = Node(parent=node)
        # Loop all attr to find best split
        for attr in self.__attrs:
            sort_dataset = dataset.sort_values(by=[attr]).reset_index()
            # Loop to find threshold
            for i in range(len(sort_dataset) - 1):
                threshold = sort_dataset[attr][i]/2 + sort_dataset[attr][i+1]/2
                # Split data
                left_ds = dataset[dataset[attr] < threshold]
                right_ds = dataset[dataset[attr] >= threshold]
                Ti = [
                    left_ds[self.__target].to_numpy(),
                    right_ds[self.__target].to_numpy(),
                ]
                # calc expected error reduction
                eer = self.calc_EER(T, Ti)
                # If find better -> update
                if eer > max_eer:
                    max_eer = eer
                    left_node.dataset = left_ds
                    right_node.dataset = right_ds
                    node.set_condition(attr, threshold)
        # Create sub tree
        node.childs = (left_node, right_node)
        # print(">> Devide to 2 branch with condition", node.get_condition())
        self.create_tree(left_node)
        self.create_tree(right_node)

    def simplify_linear_models(self, node: Node):
        # Current status
        node.model_variable = self.__attrs 
        node.model = LinearRegression()
        node.model.fit(node.dataset[node.model_variable].to_numpy(), node.dataset[self.__target].to_numpy())
        current_error = self.calc_node_error(node)

        # Greedy simpify linear models
        greedy_variables = node.model_variable.copy()
        while len(greedy_variables) > 1:
            count = len(greedy_variables)
            check_count = count
            for givenIndex in range(len(greedy_variables)):
                temp = greedy_variables[:givenIndex] + greedy_variables[givenIndex + 1:]
                temp_model = LinearRegression() 
                X, Y = node.dataset[temp].to_numpy(), node.dataset[self.__target].to_numpy()
                temp_model.fit(X, Y)
                Yhat = temp_model.predict(X)
                temp_error = self.calc_error(Y, Yhat, X.shape[1]) 
                if temp_error < current_error:
                    check_count -= 1
                    current_error = temp_error
                    greedy_variables = temp.copy()
            if check_count == count:
                break
        # print("Chosen variable:", greedy_variables)
        node.model_variable = greedy_variables
        node.model.fit(node.dataset[node.model_variable].to_numpy(), node.dataset[self.__target].to_numpy())

        
    def prune(self, node: Node):
        """
        This function use to prune the tree following page 125 https://urdata.net/files/2016_accident_duration_prediction.pdf
        """
        if not node.is_leaf():
            # Prune in subtree
            self.prune(node.childs[0])
            self.prune(node.childs[1])
            # prune if subtree error bad then this node
            subtree_error = self.calc_subtree_error(node)
            this_node_error = self.calc_node_error(node)
            if  subtree_error >= this_node_error:
                print(">> Prune tree !!!", subtree_error, this_node_error)
                node.childs = (None, None)
        
    def fit(self, dataset, attrs: list=[], target: str=None):
        self.__attrs = attrs
        self.__target = target
        self.__node = Node(dataset=dataset)
        self.__SD = np.std(dataset[target].to_numpy())
        self.create_tree(self.__node)
        self.prune(self.__node)
            
    def predict(self, x):
        path_to_leaf = self.find_node_path(self.__node, x)
        path_to_leaf.reverse()
        # Get leaf model predict
        leaf_node = path_to_leaf[0]
        pred = leaf_node.model.predict(x[leaf_node.model_variable].to_numpy())[0]
        # Get predict value on each node to root and Smoothing
        for i in range(1, len(path_to_leaf)):
            node = path_to_leaf[i]
            node_predict = node.model.predict(x[node.model_variable].to_numpy())[0]
            ni = len(path_to_leaf[i-1].dataset)
            pred = (ni*pred + self.__k*node_predict) / (self.__k + ni) # Formula 4.2 https://hiof.brage.unit.no/hiof-xmlui/bitstream/handle/11250/293858/15-00486-12%20ThesisReport_HieuHuynh.pdf%20235138_1_1.pdf?sequence=1
        # 
        return pred
    
    def get_rule(self):
        nodes = [([], self.__node)]
        paths = []
        while nodes:
            # pop queue
            path, node = nodes[0]
            del nodes[0]
            # check if reach leaf
            if node.is_leaf():
                str_model = "+".join([f"{w}*{attr}" for w, attr in zip(node.model.coef_, node.model_variable)])
                str_model = str_model + "+" + str(node.model.intercept_)
                paths.append(path + [str_model])
                
            else:
                attr, threshold, _ = node.get_condition()
                nodes.append((path + [f"{attr} < {threshold}"], node.childs[0]))
                nodes.append((path + [f"{attr} >= {threshold}"], node.childs[1]))
        return paths
    
    @classmethod
    def find_node_path(cls, node: Node, x):
        paths = [node]
        while not paths[-1].is_leaf():
            node = paths[-1]
            attr, threshold, _ = node.get_condition()
            if x[attr][0] < threshold:
                paths.append(node.childs[0])
            else:
                paths.append(node.childs[1])
        return paths
    
           
    @staticmethod
    def calc_error(y, yhat, num_param: int):
        n = len(y)
        return np.mean(np.abs(y - yhat)) * (n + num_param) / (n - num_param)
    
    @staticmethod
    def calc_EER(T:np.array, Ti:np.ndarray):
        """
        This function use to find expected error reduction with concept: sd(T) - sum(|Ti|/|T|*sd(Ti)) 
        with Ti is sub-case
        """
        return np.std(T) - np.sum([np.sum(ti) / np.sum(T) * np.std(ti) for ti in Ti])
        
        
if __name__ == "__main__":
    """
    # Train data
    df = pd.DataFrame({
        "x1": [0, -1, 1, 2, 3, 4, 5, 6, 7, 8],
        "x2": [0, 5, 2, 4, 5, 6, 8, 9, 10, 11],
        "y": [1, -1, 3, 5, 7, 17, 21, 25, 29, 33],
    })
    print(">> Train data")
    print(df)
    # Train
    model = RegressionTree(k=1, min_split=4, sd_threshold=0.2) # just set sd=0.2 to debug
    model.fit(df, ["x1", "x2"], "y")
    # Test
    test = pd.DataFrame({
        "x1": [5.5],
        "x2": [7.8]
    })
    print(">> Input")
    print(test)
    print(">> Predict")
    print(model.predict(test))
    
    print("== All RULE ==")
    rules = model.get_rule()
    for rule in rules:
        print(" AND ".join(rule))
    
    """
    
    DATASET = r"clean_dataset_v3.csv"
    dataset = pd.read_csv(DATASET)
    dataset = dataset[["PriceSale",	"Brand", "RamCapacity",	"DisplaySize", "PinCapacity",	"PinCell", "DiskSpace"]]
    # Xoa mau co du lieu gia tien bi trong
    dataset = dataset.dropna(subset=["PriceSale"])
    # Xoa mau co hon 1 tr??ong trong
    dataset["num_nan"] = dataset.isnull().apply(lambda row: row.sum(), axis=1)
    dataset = dataset[dataset.num_nan <= 0]
    # Encode onehot
    one_hot = pd.get_dummies(dataset['Brand'])
    # Drop column B as it is now encoded
    dataset = dataset.drop('Brand',axis = 1)
    # Join the encoded df
    dataset = dataset.join(one_hot)
    train, test = train_test_split(dataset)
    train = train.reset_index()
    test = test.reset_index()
    
    # Nomalize
    for attr in ["RamCapacity",	"DisplaySize", "PinCapacity",	"PinCell"]:
        scaler = MinMaxScaler()
        scaler.fit(train[[attr]])
        train[[attr]] = scaler.transform(train[[attr]])
        test[[attr]] = scaler.transform(test[[attr]])
    
    # scaler = StandardScaler()
    # scaler.fit(train[["PriceSale"]])
    # train[["PriceSale"]] = scaler.transform(train[["PriceSale"]])
    # test[["PriceSale"]] = scaler.transform(test[["PriceSale"]])
    
    model = RegressionTree(k=5, min_split=7, sd_threshold=0.95) 
    model.fit(train, ["RamCapacity",  "DisplaySize",  "PinCapacity",  "PinCell",  "DiskSpace",   "ACER",  "ASUS",  "DELL",  "FUJITSU",  "GIGABYTE",  "HP",  "LENOVO",  "LG",  "MSI"], "PriceSale")
    
    labels = []
    preds = []
    for i, row in tqdm(test.iterrows()):
        sample = pd.DataFrame(row).T.reset_index()
        labels.append(row.PriceSale)
        preds.append(model.predict(sample))

    print(f"MAPE={mape(labels, preds)}")
    
    print("== All RULE ==")
    rules = model.get_rule()
    for rule in rules:
        print(" AND ".join(rule[:-1]))
        print("Model:", rule[-1])
    # print((np.sum(np.abs(np.array(labels) - np.array(preds)))) / len(labels))
    # https://hal.archives-ouvertes.fr/hal-03762155/file/220826%20python-m5p%20-%20Sylvain%20MARIE%201.1.pdf