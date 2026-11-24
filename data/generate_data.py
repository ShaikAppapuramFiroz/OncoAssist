import pandas as pd
import numpy as np
import os

def generate_perfectly_balanced_data(samples_per_class=1000):
    """
    Generates a perfectly balanced dataset with 6 distinct clinical signatures:
    UrinaryBladder, Kidney, Prostate, Cystitis, Uterus, and Normal (Healthy).
    """
    np.random.seed(42)
    classes = ['UrinaryBladder', 'Kidney', 'Prostate', 'Cystitis', 'Uterus', 'Normal']
    all_dfs = []
    
    for cls in classes:
        # Base clinical data for all
        data = {
            'Age': np.random.randint(20, 85, samples_per_class),
            'Gender': np.random.choice(['Male', 'Female'], samples_per_class),
            'WBC': np.random.normal(7000, 800, samples_per_class),
            'RBC': np.random.normal(4.8, 0.3, samples_per_class),
            'Hemoglobin': np.random.normal(14, 1.0, samples_per_class),
            'Platelets': np.random.normal(240000, 30000, samples_per_class),
            'ALT': np.random.normal(25, 5, samples_per_class),
            'AST': np.random.normal(22, 4, samples_per_class),
            'ALP': np.random.normal(80, 15, samples_per_class),
            'Bilirubin': np.random.normal(0.5, 0.1, samples_per_class),
            'Creatinine': np.random.normal(0.9, 0.1, samples_per_class),
            'Urea': np.random.normal(12, 2.0, samples_per_class),
            'UricAcid': np.random.normal(5.0, 0.5, samples_per_class),
            'Urinalysis_pH': np.random.normal(6.0, 0.4, samples_per_class),
            'Urinalysis_SpecificGravity': np.random.normal(1.015, 0.003, samples_per_class),
            'BloodInUrine': np.zeros(samples_per_class, dtype=int),
            'FrequentUrination': np.zeros(samples_per_class, dtype=int),
            'PainfulUrination': np.zeros(samples_per_class, dtype=int),
            'PelvicPain': np.zeros(samples_per_class, dtype=int),
            'BackPain': np.zeros(samples_per_class, dtype=int),
        }
        
        df = pd.DataFrame(data)
        
        # Apply class-specific distinct rules
        if cls == 'UrinaryBladder':
            df['Gender'] = 'Male'
            df['Age'] = np.random.randint(65, 90, samples_per_class)
            df['BloodInUrine'] = 1
            df['FrequentUrination'] = 1
            df['WBC'] = np.random.normal(11500, 800, samples_per_class)
            
        elif cls == 'Kidney':
            df['Creatinine'] = np.random.normal(4.2, 0.4, samples_per_class)
            df['Urea'] = np.random.normal(55, 8, samples_per_class)
            df['BackPain'] = 1
            df['Hemoglobin'] = np.random.normal(9.5, 0.8, samples_per_class)
            
        elif cls == 'Prostate':
            df['Gender'] = 'Male'
            df['Age'] = np.random.randint(60, 90, samples_per_class)
            df['FrequentUrination'] = 1
            df['ALP'] = np.random.normal(160, 15, samples_per_class)
            df['UricAcid'] = np.random.normal(8.0, 0.6, samples_per_class)
            
        elif cls == 'Cystitis':
            df['Gender'] = 'Female'
            df['PainfulUrination'] = 1
            df['Urinalysis_pH'] = np.random.normal(8.5, 0.2, samples_per_class)
            df['WBC'] = np.random.normal(17000, 1200, samples_per_class)
            
        elif cls == 'Uterus':
            df['Gender'] = 'Female'
            df['Age'] = np.random.randint(25, 55, samples_per_class)
            df['PelvicPain'] = 1
            df['Hemoglobin'] = np.random.normal(10.5, 0.8, samples_per_class)
            df['AST'] = np.random.normal(50, 4, samples_per_class)
            
        elif cls == 'Normal':
            # Healthy ranges - already set by base distributions
            # Explicitly ensure no symptoms
            df['BloodInUrine'] = 0
            df['FrequentUrination'] = 0
            df['PainfulUrination'] = 0
            df['PelvicPain'] = 0
            df['BackPain'] = 0
            df['WBC'] = np.random.normal(6500, 500, samples_per_class)
            df['Creatinine'] = np.random.normal(0.8, 0.05, samples_per_class)
            
        df['Target'] = cls
        all_dfs.append(df)
        
    final_df = pd.concat(all_dfs, ignore_index=True)
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Add minor missing values (2%)
    for col in final_df.columns:
        if col != 'Target':
            mask = np.random.random(len(final_df)) < 0.02
            final_df.loc[mask, col] = np.nan
            
    os.makedirs('data', exist_ok=True)
    final_df.to_csv('data/dataset.csv', index=False)
    
    print("Dataset Balanced (6 Classes):")
    print(final_df['Target'].value_counts())

if __name__ == "__main__":
    generate_perfectly_balanced_data(1000)
