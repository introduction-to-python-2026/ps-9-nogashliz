# Download the data (כמו שהיה לך)
!wget https://raw.githubusercontent.com/yotam-biu/ps9/main/parkinsons.csv -O /content/parkinsons.csv
!wget https://raw.githubusercontent.com/yotam-biu/python_utils/main/lab_setup_do_not_edit.py -O /content/lab_setup_do_not_edit.py
import lab_setup_do_not_edit

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline # <--- תוספת חשובה
import joblib

# 1. טעינת הנתונים
df = pd.read_csv("parkinsons.csv")
X = df[['PPE', 'DFA']]
y = df['status']

# שימי לב: מחלקים ל-Train/Test את הנתונים המקוריים (לא המנורמלים ידנית)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. יצירת Pipeline שכולל גם את הנרמול וגם את המודל
# המודל הזה ינרמל לבד כל מידע חדש שיגיע אליו
model = make_pipeline(MinMaxScaler(), SVC())

# 3. אימון ה-Pipeline
model.fit(X_train, y_train)

# 4. בדיקה
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy}")

if accuracy >= 0.8:
    print("Great! The accuracy is above 0.8.")
else:
    print("The accuracy is below 0.8.")

# 5. שמירת המודל (שכעת כולל בתוכו את ה-Scaler)
joblib.dump(model, 'my_model.joblib')
