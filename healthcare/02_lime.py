# %% Imports
from utils import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from interpret.blackbox import LimeTabular
from lime.lime_tabular import LimeTabularExplainer
from interpret import show

# %% Load and pre-processing
data_loader = DataLoader()
data_loader.load_dataset()
data_loader.preprocess_data()

# Split the data for evaluation
X_train, X_test, y_train, y_test = data_loader.get_data_split()

# Oversample the train data
X_train, y_train = data_loader.oversample(X_train, y_train)
print(X_train.shape)
print(X_test.shape)

# %% Fit blackbox model
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"F1 Score {f1_score(y_test, y_pred, average='macro')}")
print(f"Accuracy {accuracy_score(y_test, y_pred)}")

# %% Apply lime
class_names = ['No Stroke', 'Stroke']
feature_names = list(X_train.columns)

explainer = LimeTabularExplainer(X_train.values, feature_names=feature_names,class_names=class_names,mode='classification')

num_instances = 5
for instance_index in range(num_instances):
    instance = X_test.iloc[[instance_index]]
    true_class = y_test.iloc[instance_index]

    explanation = explainer.explain_instance(instance.values[0], model.predict_proba, num_features=len(X_train.columns), top_labels=1)

    explanation.show_in_notebook(show_table=True)

# %% Other Way
lime = LimeTabular(model=model,data=X_train, random_state=1)
lime_local = lime.explain_local(X_test[-20:], y_test[-20:], name='LIME')

show(lime_local)
# %%
