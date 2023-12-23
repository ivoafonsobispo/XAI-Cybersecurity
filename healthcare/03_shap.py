# %% Imports
from utils import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
import shap 

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

# %% Create SHAP explainer
class_names = ['No Stroke', 'Stroke']
feature_names = list(X_train.columns)

explainer = shap.TreeExplainer(model)

# Calculate shapley values for test data
start_index = 1
end_index = 27
shap_values = explainer.shap_values(X_test[start_index:end_index])
X_test[start_index:end_index]

# Investigating the values (classification problem)
# class 0 = contribution to class 1
# class 1 = contribution to class 2
print(shap_values[0].shape)
shap_values

# %% >> Visualize local predictions
shap.initjs()

prediction = model.predict(X_test[start_index:end_index])[0]
print(f"The RF predicted: {prediction}")
shap.force_plot(explainer.expected_value[1],
                shap_values[1],
                X_test[start_index:end_index], feature_names=feature_names, matplotlib=True)

# %% >> Visualize global features
shap.summary_plot(shap_values, X_test, feature_names=feature_names,class_names=class_names)

# %% 
shap.plots.force(explainer.expected_value[1],shap_values[1],feature_names=feature_names)

# %% Waterfall Plot
import xgboost

# train an XGBoost model
model = xgboost.XGBRegressor().fit(X_train, y_train)

explainer = shap.Explainer(model)
shap_values = explainer(X_test)

# visualize the first prediction's explanation
shap.plots.waterfall(shap_values[0])

# %%
shap.plots.beeswarm(shap_values)
# %%
