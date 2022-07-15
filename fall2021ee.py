#%%
import pandas as pd
import numpy as np
import altair as alt
from sklearn.ensemble import RandomForestClassifier
from datetime import date
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
#%%
# Most recent f22 data
f21 = pd.read_csv("/Users/12082/Desktop/Pathway/Programming/Endorsement/Data/fall_2021_applicants.csv")
f21 = f21.query("Endorsement == 'Complete' | Endorsement == 'Incomplete'")
# %%
'''
Use PC Complete Sem, Courses, PC Status, GPA (All), GPA (Core), PC Credits (All), 
Provider, Type, Created?, Endorsement(target), Matriculate Type, Major??
'''
#%%
f21["Endorsement"].value_counts() # 29% incomplete
# Will need to oversample lower class a bit so closer to 50/50.

fall = f21.drop(columns = ["Person ID", "Area", "Location", "First Name", "Last Name", "Email", "Phone", "Country", "State", "City", "Application Status", "Submitted", "Transcript", "Completed", "Decided", "Decision", "Enroll Status", "Enrolled", "F21 Credits", "# Students", "Decision Type"])
fall
#%%
fall["GPA (Core)"].mean() # 3.6968396578359077
fall["GPA (All)"].mean() # 3.570722937932804
fall['GPA (Core)'] = fall['GPA (Core)'].fillna(3.6968396578359077)
fall['GPA (All)'] = fall['GPA (All)'].fillna(3.570722937932804)
fall['PC Credits (All)'] = fall['PC Credits (All)'].fillna(0)
fall['PC Status'] = fall['PC Status'].fillna(0)
# But PC Status is opposite now with 1 at where na was and 0 at everything else.
fall.head(10)
# %%
# Factorize
fall["PC Status"] = pd.factorize(fall["PC Status"])[0]
fall["Endorsement"] = pd.factorize(fall["Endorsement"])[0]
# Getting -1's from factorizing

# Subbing
fall['Courses'] = np.where(fall['Courses'].isnull(), 0, 1)
fall['PC Completed Sem'] = np.where(fall['PC Completed Sem'].isnull(), 0, 1)
# fall['PC Status'] = np.where(fall['PC Status'].isnull(), 0, 1)
fall.head(10)

#%%
# One hot encode
fall = pd.get_dummies(fall, columns = ["Matriculate Type", "Type", "Provider", "Major"], drop_first = True)
fall.head()
#%%
# See one PC Credits (All) at 76654.0
high_cred = fall[fall["PC Credits (All)"] > 26]
# 26 still reasonable, but 76654.0 is not, im gonna take out this row.
high_cred
#%%
fall.shape # 9958 rows
# remove that one row
fall = fall[fall["PC Credits (All)"] <27]
fall.shape # 9957, it worked
fall
# %%
# Getting created date column to a numeric value
fall['Created'] = fall['Created'].fillna(0)
fall['Created'] = pd.to_datetime(fall['Created'])
d0 = date(1970,1,1)
fall["base_date"] = d0
fall['base_date'] = pd.to_datetime(fall['base_date'])
fall["Created_in_days"] = (fall["Created"] - fall["base_date"]).dt.days
fall = fall.drop(columns = ["base_date", "Created"])
fall
# Why PC Status have a -1? And a whole bunch a stuff, should just be completed or not.
# %%
fall = fall.drop(columns = "PC Status")
fall1 = fall
#%%
X = fall1.drop(columns = ["Endorsement"])
y = fall1["Endorsement"]
x_train, X_val, Y_train, y_val = train_test_split(X, y, test_size = 0.15, random_state = 20)
X_train, X_test, y_train, y_test = train_test_split(x_train, Y_train, test_size = 0.1765, random_state = 20)
X_train
# Oversample the incomplete endorsements in the training data
ro = RandomOverSampler()

X = X_train
y = y_train
X_train_new, y_train_new = ro.fit_resample(X, y)
bal_training_target = pd.DataFrame(y_train_new)
bal_training_features = pd.DataFrame(X_train_new)
bal_training_target.value_counts()
bal_training_features
#%%
classifier = GaussianNB()
classifier.fit(bal_training_features, bal_training_target)
y_predicted = classifier.predict(X_test)
metrics.accuracy_score(y_test, y_predicted)
# First run had an acc of 77%, 0.7735583684950773 # Whoops on test data though
# On val 76%, 0.7579169598874033
# 77% testing
# %%
# Confusion Matrix
metrics.plot_confusion_matrix(classifier, X_test, y_test)
# Is my initial endorsement encoding backwards? Or is this just bad haha.
# Yes- endorsement at 0 is complete currently in this case.
# So minimizing bottom left in this current case is really good and means I'm only missing 4 of the 409 that need to be contacted. That is Less than 1%!
# Awesome! Can keep like this as long as I explain and interpret it right its all good.
# %%
# With Decision Tree?
classifier_DT = DecisionTreeClassifier()
classifier_DT.fit(bal_training_features, bal_training_target)
y_predicted_DT = classifier_DT.predict(X_test)
metrics.accuracy_score(y_test, y_predicted_DT)
# Acc 77%, 0.7684729064039408
# 77% acc w testing set, 0.7679324894514767
#%%
metrics.plot_confusion_matrix(classifier_DT, X_test, y_test)
# Missing 33%, 133/409
#%%
# With Random Forest
classifier_RF = RandomForestClassifier(n_estimators=10)
classifier_RF.fit(bal_training_features, bal_training_target)
y_predicted_RF = classifier_RF.predict(X_test)
metrics.accuracy_score(y_test, y_predicted_RF)
# Acc 78%, 0.7839549612948628
# 78% acc w testing, 0.7784810126582279

#%%
metrics.plot_confusion_matrix(classifier_RF, X_test, y_test)
# missing 27%, 112/409

#%%
data = [["GaussianNB", 0.48], ["DecisionTree", 35], ["RandomForest", 30]]
new_df = pd.DataFrame(data, columns = ["Model", "Percentage of M issed Students"])
new_df
bars = alt.Chart(data = new_df).mark_bar().encode(
    x = alt.X("Model", sort = None),
    y = "Percentage of M issed Students"
)

text = bars.mark_text(
    align = "center",
    baseline = "middle",
    dy = -5
).encode(
    text = "Percentage of M issed Students"
)

bars + text
# %%
