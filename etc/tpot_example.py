import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv(
    "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
)
data.head()

data.drop(["Ticket", "PassengerId"], axis=1, inplace=True)

gender_mapper = {"male": 0, "female": 1}
data["Sex"].replace(gender_mapper, inplace=True)

data["Title"] = data["Name"].apply(
    lambda x: x.split(",")[1].strip().split(" ")[0]
)
data["Title"] = [
    0 if x in ["Mr.", "Miss.", "Mrs."] else 1 for x in data["Title"]
]
data = data.rename(columns={"Title": "Title_Unusual"})
data.drop("Name", axis=1, inplace=True)

data["Cabin_Known"] = [0 if str(x) == "nan" else 1 for x in data["Cabin"]]
data.drop("Cabin", axis=1, inplace=True)

emb_dummies = pd.get_dummies(
    data["Embarked"], drop_first=True, prefix="Embarked"
)
data = pd.concat([data, emb_dummies], axis=1)
data.drop("Embarked", axis=1, inplace=True)

data["Age"] = data["Age"].fillna(int(data["Age"].mean()))

X = data.drop("Survived", axis=1)
y = data["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

ss = StandardScaler()
X_train_scaled = ss.fit_transform(X_train)
X_test_scaled = ss.transform(X_test)


from tpot import TPOTClassifier

tpot = TPOTClassifier(verbosity=2, max_time_mins=10)
tpot.fit(X_train_scaled, y_train)


cc = 1
