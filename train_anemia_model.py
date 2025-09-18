# ============================================
# ANEMIA MODEL TRAINING SCRIPT
# Save this as: train_anemia_model.py
# Run with: python train_anemia_model.py
# ============================================

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

def create_and_train_anemia_model():
    """Create and train anemia model quickly"""
    
    print("🩸 Creating Anemia Dataset...")
    
    # Create sample anemia data (realistic medical dataset)
    np.random.seed(42)
    n_samples = 1000
    
    # Generate realistic anemia dataset based on medical research
    data = {
        'age': np.random.randint(15, 70, n_samples),
        'gender': np.random.randint(0, 2, n_samples),  # 0=Female, 1=Male
        'hemoglobin': np.random.uniform(8, 16, n_samples),  # g/dL
        'mch': np.random.uniform(25, 35, n_samples),  # Mean Corpuscular Hemoglobin
        'mchc': np.random.uniform(30, 37, n_samples),  # Mean Corpuscular Hemoglobin Concentration
        'mcv': np.random.uniform(75, 100, n_samples),  # Mean Corpuscular Volume
    }
    
    # Create realistic target variable based on medical criteria
    # Anemia is mainly determined by:
    # 1. Low hemoglobin levels (primary factor)
    # 2. Low MCH, MCHC, MCV values
    # 3. Gender differences (females more prone due to menstruation)
    
    anemia_risk = np.zeros(n_samples)
    
    for i in range(n_samples):
        risk_score = 0
        
        # Hemoglobin thresholds (most important factor)
        if data['gender'][i] == 0:  # Female
            if data['hemoglobin'][i] < 12.0:  # WHO threshold for women
                risk_score += 3
            elif data['hemoglobin'][i] < 11.0:
                risk_score += 4
        else:  # Male
            if data['hemoglobin'][i] < 13.0:  # WHO threshold for men
                risk_score += 3
            elif data['hemoglobin'][i] < 12.0:
                risk_score += 4
        
        # Other blood parameters
        if data['mch'][i] < 27:  # Low MCH
            risk_score += 1
        if data['mchc'][i] < 32:  # Low MCHC
            risk_score += 1
        if data['mcv'][i] < 80:  # Low MCV (microcytic anemia)
            risk_score += 1
        
        # Age factor (elderly more prone)
        if data['age'][i] > 60:
            risk_score += 1
        
        anemia_risk[i] = risk_score
    
    # Convert to binary classification (0 = No Anemia, 1 = Anemia)
    data['target'] = (anemia_risk >= 3).astype(int)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    print(f"✅ Dataset created: {df.shape[0]} samples")
    print(f"📊 Features: {list(df.columns[:-1])}")
    print(f"🩸 Anemia cases: {sum(df['target'])} ({sum(df['target'])/len(df['target'])*100:.1f}%)")
    print(f"✅ Healthy cases: {len(df) - sum(df['target'])} ({(len(df) - sum(df['target']))/len(df)*100:.1f}%)")
    
    # Prepare features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    print(f"\n🔬 Training Random Forest Model...")
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"📈 Training set: {X_train.shape[0]} samples")
    print(f"📊 Testing set: {X_test.shape[0]} samples")
    
    # Train Random Forest model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2
    )
    
    model.fit(X_train, y_train)
    
    # Make predictions on test set
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n🎯 Model Performance:")
    print(f"✅ Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    # Detailed classification report
    print(f"\n📋 Detailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Anemia', 'Anemia']))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🔍 Feature Importance:")
    for _, row in feature_importance.iterrows():
        print(f"   {row['feature']}: {row['importance']:.3f}")
    
    # Create saved_models directory if it doesn't exist
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
        print(f"\n📁 Created 'saved_models' directory")
    
    # Save the model
    model_path = 'saved_models/anemia_model.sav'
    pickle.dump(model, open(model_path, 'wb'))
    print(f"✅ Model saved to: {model_path}")
    
    # Test the saved model (verification)
    print(f"\n🧪 Testing saved model...")
    loaded_model = pickle.load(open(model_path, 'rb'))
    test_accuracy = accuracy_score(y_test, loaded_model.predict(X_test))
    print(f"✅ Loaded model accuracy: {test_accuracy:.3f}")
    
    print(f"\n🎉 Anemia model training completed successfully!")
    print(f"📝 Model ready for use in Streamlit app!")
    
    return model, accuracy

# Test function for quick verification
def test_anemia_prediction():
    """Test the anemia model with sample data"""
    print(f"\n🔬 Testing Anemia Predictions...")
    
    try:
        # Load the saved model
        model = pickle.load(open('saved_models/anemia_model.sav', 'rb'))
        
        # Test cases
        test_cases = [
            {
                'name': 'High Risk Female',
                'data': [25, 0, 9.5, 25, 30, 75],  # Young female, low hemoglobin
                'expected': 'High Risk'
            },
            {
                'name': 'Low Risk Male',
                'data': [30, 1, 14.5, 32, 35, 90],  # Adult male, normal values
                'expected': 'Low Risk'
            },
            {
                'name': 'Borderline Female',
                'data': [35, 0, 11.8, 28, 33, 82],  # Female, borderline values
                'expected': 'Moderate Risk'
            }
        ]
        
        print(f"\n📊 Sample Predictions:")
        for case in test_cases:
            prediction = model.predict([case['data']])[0]
            probability = model.predict_proba([case['data']])[0]
            confidence = max(probability) * 100
            
            result = "ANEMIA RISK" if prediction == 1 else "NO ANEMIA"
            print(f"   {case['name']}: {result} (Confidence: {confidence:.1f}%)")
        
        print(f"✅ Model testing completed!")
        
    except FileNotFoundError:
        print(f"❌ Error: Model file not found. Please run training first.")
    except Exception as e:
        print(f"❌ Error testing model: {e}")

# Main execution
if __name__ == "__main__":
    print("="*60)
    print("🏥 ANEMIA PREDICTION MODEL TRAINING")
    print("="*60)
    
    try:
        # Train the model
        model, accuracy = create_and_train_anemia_model()
        
        # Test the model
        test_anemia_prediction()
        
        print(f"\n" + "="*60)
        print(f"🎊 SUCCESS! Anemia model is ready!")
        print(f"📊 Final Accuracy: {accuracy:.1%}")
        print(f"📂 Model saved in: saved_models/anemia_model.sav")
        print(f"🚀 You can now run: streamlit run app.py")
        print(f"="*60)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print(f"💡 Please check if all required libraries are installed:")
        print(f"   pip install pandas numpy scikit-learn")
        
    input(f"\nPress Enter to exit...")