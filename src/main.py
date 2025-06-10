# Cybersecurity: Suspicious Web Threat Interactions Analysis
# Advanced Data Analytics Project
# Tools: Python, Machine Learning, SQL, Excel Integration

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.cluster import KMeans
import sqlite3
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class CyberThreatAnalyzer:
    def __init__(self):
        self.data = None
        self.processed_data = None
        self.models = {}
        self.scalers = {}

    def create_sample_dataset(self, n_samples=10000):
        np.random.seed(42)
        threat_types = ['Malware', 'Phishing', 'DDoS', 'SQL_Injection', 'XSS', 'Benign']

        data = {
            'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='1H'),
            'source_ip': [f"192.168.{np.random.randint(1,255)}.{np.random.randint(1,255)}" 
                          for _ in range(n_samples)],
            'destination_port': np.random.choice([80, 443, 8080, 3306, 22, 21, 25], n_samples),
            'packet_size': np.random.exponential(1000, n_samples),
            'request_frequency': np.random.poisson(5, n_samples),
            'session_duration': np.random.exponential(300, n_samples),
            'http_status_code': np.random.choice([200, 404, 403, 500, 301, 302], n_samples),
            'payload_entropy': np.random.uniform(0, 8, n_samples),
            'geo_location': np.random.choice(['US', 'CN', 'RU', 'DE', 'UK', 'IN', 'BR'], n_samples),
            'user_agent_anomaly_score': np.random.uniform(0, 1, n_samples),
            'dns_query_count': np.random.poisson(3, n_samples),
            'failed_login_attempts': np.random.poisson(0.5, n_samples),
            'ssl_certificate_valid': np.random.choice([0, 1], n_samples, p=[0.1, 0.9]),
            'detection_types': np.random.choice(threat_types, n_samples, 
                                                p=[0.15, 0.15, 0.10, 0.10, 0.10, 0.40])
        }

        df = pd.DataFrame(data)
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_suspicious'] = (df['detection_types'] != 'Benign').astype(int)

        malicious_mask = df['detection_types'] != 'Benign'
        df.loc[malicious_mask, 'payload_entropy'] *= 1.5
        df.loc[malicious_mask, 'request_frequency'] *= 2
        df.loc[malicious_mask, 'failed_login_attempts'] *= 3

        self.data = df
        print(f"Sample dataset created with {len(df)} records")
        print(f"Threat distribution:\n{df['detection_types'].value_counts()}")

        return df

    def load_external_dataset(self, file_path):
        """
        Loads an external CSV dataset into the analyzer.
        """
        try:
            self.data = pd.read_csv(file_path)
            print(f"Dataset loaded successfully from: {file_path}")
            print(f"Shape: {self.data.shape}")
        except Exception as e:
            print(f"Failed to load dataset: {e}")

    def exploratory_data_analysis(self):
        """
        Comprehensive EDA for cybersecurity data
        """
        if self.data is None:
            print("No data available. Please load or create dataset first.")
            return

        print("=== CYBERSECURITY THREAT ANALYSIS EDA ===\n")

        # Handle timestamp
        if 'timestamp' in self.data.columns:
            self.data['timestamp'] = pd.to_datetime(self.data['timestamp'], errors='coerce')
            self.data['hour'] = self.data['timestamp'].dt.hour
        elif 'time' in self.data.columns:
            self.data['time'] = pd.to_datetime(self.data['time'], errors='coerce')
            self.data['hour'] = self.data['time'].dt.hour
        elif 'creation_time' in self.data.columns:
            self.data['creation_time'] = pd.to_datetime(self.data['creation_time'], errors='coerce')
            self.data['hour'] = self.data['creation_time'].dt.hour
        else:
            print("Timestamp column not found. Cannot extract hour for temporal analysis.")
            self.data['hour'] = np.nan

        # Fix missing columns or create
        if 'is_suspicious' not in self.data.columns:
            if 'detection_types' in self.data.columns:
                self.data['is_suspicious'] = (self.data['detection_types'].str.lower() != 'benign').astype(int)
            elif 'rule_names' in self.data.columns:
                self.data['is_suspicious'] = self.data['rule_names'].notna().astype(int)
            else:
                self.data['is_suspicious'] = 0

        if 'destination_port' not in self.data.columns and 'dst_port' in self.data.columns:
            self.data['destination_port'] = self.data['dst_port']

        if 'geo_location' not in self.data.columns and 'src_ip_country_code' in self.data.columns:
            self.data['geo_location'] = self.data['src_ip_country_code']

        for col in ['payload_entropy', 'request_frequency']:
            if col not in self.data.columns:
                self.data[col] = np.random.normal(0, 1, size=len(self.data))

        # Begin plotting
        plt.figure(figsize=(15, 10))

        # 1. Threat Type Distribution
        plt.subplot(2, 3, 1)
        if 'detection_types' in self.data.columns:
            threat_counts = self.data['detection_types'].value_counts()
            plt.pie(threat_counts.values, labels=threat_counts.index, autopct='%1.1f%%')
            plt.title('Threat Type Distribution')
        else:
            plt.text(0.5, 0.5, "No detection_types column", ha='center', va='center')

        # 2. Suspicious Activity by Hour
        plt.subplot(2, 3, 2)
        if self.data['hour'].notna().any():
            hourly_threats = self.data.groupby('hour')['is_suspicious'].mean()
            plt.plot(hourly_threats.index, hourly_threats.values, marker='o')
            plt.title('Suspicious Activity by Hour')
            plt.xlabel('Hour of Day')
            plt.ylabel('Proportion of Suspicious Activity')
        else:
            plt.text(0.5, 0.5, "No hour data", ha='center', va='center')

        # 3. Threat Rate by Destination Port
        plt.subplot(2, 3, 3)
        if 'destination_port' in self.data.columns:
            port_threats = self.data.groupby('destination_port')['is_suspicious'].mean()
            plt.bar(port_threats.index.astype(str), port_threats.values)
            plt.title('Threat Rate by Destination Port')
            plt.xticks(rotation=45)
        else:
            plt.text(0.5, 0.5, "No destination_port data", ha='center', va='center')

        # 4. Threat Rate by Geographic Location
        plt.subplot(2, 3, 4)
        if 'geo_location' in self.data.columns:
            geo_threats = self.data.groupby('geo_location')['is_suspicious'].mean().sort_values(ascending=False)
            plt.bar(geo_threats.index, geo_threats.values)
            plt.title('Threat Rate by Geographic Location')
            plt.xticks(rotation=45)
        else:
            plt.text(0.5, 0.5, "No geo_location data", ha='center', va='center')

        # 5. Payload Entropy Distribution
        plt.subplot(2, 3, 5)
        plt.hist(self.data[self.data['is_suspicious']==1]['payload_entropy'], alpha=0.7, label='Suspicious', bins=30)
        plt.hist(self.data[self.data['is_suspicious']==0]['payload_entropy'], alpha=0.7, label='Benign', bins=30)
        plt.legend()
        plt.title('Payload Entropy Distribution')
        plt.xlabel('Entropy')

        # 6. Request Frequency Comparison
        plt.subplot(2, 3, 6)
        plt.boxplot([
            self.data[self.data['is_suspicious']==0]['request_frequency'],
            self.data[self.data['is_suspicious']==1]['request_frequency']
        ])
        plt.xticks([1, 2], ['Benign', 'Suspicious'])
        plt.title('Request Frequency Comparison')
        plt.ylabel('Requests per Session')

        plt.tight_layout()
        plt.show()

        # Correlation Matrix
        plt.figure(figsize=(12, 8))
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.data[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.show()
  # Dummy for plotting

    # === Keep rest of your plotting code below this ===

    def feature_engineering(self):
        """
        Advanced feature engineering for cybersecurity analysis
        """
        if self.data is None:
            print("No data to process.")
            return None

        df = self.data.copy()

        # Identify datetime column
        datetime_col = None
        for col_candidate in ['timestamp', 'time', 'creation_time']:
            if col_candidate in df.columns:
                datetime_col = col_candidate
                break

        if datetime_col is None:
            raise ValueError("No datetime column ('timestamp', 'time', or 'creation_time') found in data.")

        # Convert to datetime
        df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')

        # Create time features
        df['day_of_week'] = df[datetime_col].dt.dayofweek
        df['hour'] = df[datetime_col].dt.hour

        # Weekend and night indicators
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)

        # Log-transform features with checks
        if 'packet_size' in df.columns:
            df['packet_size_log'] = np.log1p(df['packet_size'])
        else:
            df['packet_size_log'] = 0
            print("[Warning] 'packet_size' column missing. Filled with 0.")

        if 'session_duration' in df.columns:
            df['session_duration_log'] = np.log1p(df['session_duration'])
        else:
            df['session_duration_log'] = 0
            print("[Warning] 'session_duration' column missing. Filled with 0.")

        # Request frequency anomaly score
        if 'request_frequency' in df.columns:
            df['request_freq_zscore'] = np.abs(
                (df['request_frequency'] - df['request_frequency'].mean()) / df['request_frequency'].std()
            )
        else:
            df['request_freq_zscore'] = 0
            print("[Warning] 'request_frequency' column missing. Filled with 0.")

        # Common ports
        if 'destination_port' in df.columns:
            common_ports = [80, 443, 22, 21]
            df['is_common_port'] = df['destination_port'].isin(common_ports).astype(int)
        else:
            df['is_common_port'] = 0
            print("[Warning] 'destination_port' column missing. Filled with 0.")

        # High-risk geo flag
        if 'geo_location' in df.columns:
            high_risk_countries = ['CN', 'RU']
            df['high_risk_geo'] = df['geo_location'].isin(high_risk_countries).astype(int)

            # Encode geo_location
            le_geo = LabelEncoder()
            df['geo_location_encoded'] = le_geo.fit_transform(df['geo_location'].astype(str))
        else:
            df['high_risk_geo'] = 0
            df['geo_location_encoded'] = 0
            print("[Warning] 'geo_location' column missing. Filled with 0.")

        self.processed_data = df
        print("Feature engineering completed")
        return df

    
    def train_ml_models(self):
        """
        Train multiple machine learning models for threat detection
        """
        if self.processed_data is None:
            self.feature_engineering()

        # Define expected features
        feature_cols = [
            'packet_size_log', 'request_frequency', 'session_duration_log',
            'payload_entropy', 'user_agent_anomaly_score', 'dns_query_count',
            'failed_login_attempts', 'ssl_certificate_valid', 'hour',
            'is_weekend', 'is_night', 'request_freq_zscore',
            'is_common_port', 'high_risk_geo', 'geo_location_encoded'
        ]

        # Filter only available features
        available_cols = [col for col in feature_cols if col in self.processed_data.columns]
        missing_cols = [col for col in feature_cols if col not in self.processed_data.columns]

        if missing_cols:
            print(f"[Warning] Missing features skipped: {missing_cols}")

        X = self.processed_data[available_cols]

        if 'is_suspicious' not in self.processed_data.columns:
            raise ValueError("'is_suspicious' label column is missing in the dataset.")

        y = self.processed_data['is_suspicious']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                            random_state=42, stratify=y)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['standard'] = scaler

        # Train models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42),
            'Isolation Forest': IsolationForest(contamination=0.1, random_state=42)
        }

        results = {}

        for name, model in models.items():
          print(f"\nTraining {name}...")

        if name == 'Isolation Forest':
            # Unsupervised anomaly detection
            model.fit(X_train_scaled)
            y_pred = model.predict(X_test_scaled)
            y_pred = np.where(y_pred == -1, 1, 0)
            accuracy = np.mean(y_pred == y_test)
            auc_score = roc_auc_score(y_test, y_pred)

        else:
            # Supervised learning
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            # Handle binary vs single-class case safely
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_test_scaled)
                if proba.shape[1] == 2:
                    y_pred_proba = proba[:, 1]
                    auc_score = roc_auc_score(y_test, y_pred_proba)
                else:
                    y_pred_proba = proba[:, 0]
                    auc_score = 0.0  # Can't compute meaningful AUC
                    print(f"[Warning] Only one class '{model.classes_[0]}' in training. AUC set to 0.")
            else:
                y_pred_proba = None
                auc_score = 0.0

            accuracy = model.score(X_test_scaled, y_test)

        # Save results
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'auc_score': auc_score,
            'predictions': y_pred,
            'test_labels': y_test
        }

        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC Score: {auc_score:.4f}")
        print(f"Classification Report:\n{classification_report(y_test, y_pred)}")


    def threat_clustering_analysis(self):
        """
        Perform clustering analysis to identify threat patterns
        """
        if self.processed_data is None:
            self.feature_engineering()

        # Select features for clustering
        cluster_features = ['payload_entropy', 'request_frequency', 'session_duration_log',
                            'failed_login_attempts', 'user_agent_anomaly_score']

        # Filter only available features to avoid KeyError
        available_features = [f for f in cluster_features if f in self.processed_data.columns]
        missing_features = [f for f in cluster_features if f not in self.processed_data.columns]
        if missing_features:
            print(f"[Warning] Missing clustering features skipped: {missing_features}")
        if not available_features:
            print("[Error] No available features for clustering.")
            return

        X_cluster = self.processed_data[available_features]

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_cluster)

        # Determine optimal number of clusters (Elbow Method)
        inertias = []
        K_range = range(2, 11)
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)

        # Plot elbow curve
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(K_range, inertias, 'bo-')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia')
        plt.title('Elbow Method for Optimal k')

        # Perform clustering with chosen k (e.g., 5)
        optimal_k = 5
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        cluster_labels = kmeans.fit_predict(X_scaled)

        # Assign cluster labels
        self.processed_data['cluster'] = cluster_labels

        # Analyze clusters (handle missing columns gracefully)
        analysis_cols = ['is_suspicious', 'detection_types', 'payload_entropy',
                         'request_frequency', 'failed_login_attempts']
        analysis_cols = [col for col in analysis_cols if col in self.processed_data.columns]

        agg_funcs = {
            'is_suspicious': 'mean',
            'detection_types': lambda x: x.mode().iloc[0] if not x.mode().empty else 'N/A',
            'payload_entropy': 'mean',
            'request_frequency': 'mean',
            'failed_login_attempts': 'mean'
        }

        cluster_analysis = self.processed_data.groupby('cluster').agg({
            col: agg_funcs[col] for col in analysis_cols
        }).round(3)

        print("Cluster Analysis:")
        print(cluster_analysis)

        # Visualize clusters (only if at least 2 features exist)
        if X_scaled.shape[1] >= 2:
            plt.subplot(1, 2, 2)
            scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=cluster_labels, cmap='viridis')
            plt.xlabel(f'{available_features[0]} (scaled)')
            plt.ylabel(f'{available_features[1]} (scaled)')
            plt.title('Threat Behavior Clusters')
            plt.colorbar(scatter)
            plt.tight_layout()
            plt.show()
        else:
            print("[Warning] Not enough features to plot cluster scatter.")

        return cluster_analysis

    
    def generate_threat_report(self):
        """
        Generate comprehensive threat analysis report
        """
        if self.data is None:
            print("No data available for report generation")
            return

        # Time period
        if 'timestamp' in self.data.columns:
            time_min = self.data['timestamp'].min()
            time_max = self.data['timestamp'].max()
            if hasattr(time_min, 'strftime'):
                time_min = time_min.strftime('%Y-%m-%d %H:%M:%S')
                time_max = time_max.strftime('%Y-%m-%d %H:%M:%S')
            time_period = f"{time_min} to {time_max}"
        else:
            time_period = "N/A (timestamp missing)"

        # Suspicious activities count and percent (assumes 'is_suspicious' exists)
        suspicious_count = self.data['is_suspicious'].sum() if 'is_suspicious' in self.data.columns else 0
        suspicious_pct = (self.data['is_suspicious'].mean() * 100) if 'is_suspicious' in self.data.columns else 0

        # Threat breakdown
        if 'detection_types' in self.data.columns:
            threat_breakdown = self.data['detection_types'].value_counts().to_string()
        else:
            threat_breakdown = "N/A (detection_types missing)"

        # Geographic threat distribution
        if 'geo_location' in self.data.columns and 'is_suspicious' in self.data.columns:
            geo_threat = self.data.groupby('geo_location')['is_suspicious'].agg(['count', 'mean']).round(3).to_string()
        else:
            geo_threat = "N/A (geo_location or is_suspicious missing)"

        # Peak threat hours
        if 'hour' in self.data.columns and 'is_suspicious' in self.data.columns:
            peak_hours = self.data.groupby('hour')['is_suspicious'].mean().nlargest(5).round(3).to_string()
        else:
            peak_hours = "N/A (hour or is_suspicious missing)"

        # High-risk indicators
        suspicious_data = self.data[self.data['is_suspicious'] == 1] if 'is_suspicious' in self.data.columns else None

        avg_entropy = (
            suspicious_data['payload_entropy'].mean()
            if suspicious_data is not None and 'payload_entropy' in suspicious_data.columns else None
        )
        avg_failed_logins = (
            suspicious_data['failed_login_attempts'].mean()
            if suspicious_data is not None and 'failed_login_attempts' in suspicious_data.columns else None
        )
        top_ports = (
            suspicious_data['destination_port'].value_counts().head(3).to_dict()
            if suspicious_data is not None and 'destination_port' in suspicious_data.columns else None
        )

        report = f"""
        ╔══════════════════════════════════════════════════════════════╗
        ║                 CYBERSECURITY THREAT ANALYSIS REPORT         ║
        ╚══════════════════════════════════════════════════════════════╝
        
        📊 DATASET OVERVIEW
        • Total Records: {len(self.data):,}
        • Suspicious Activities: {suspicious_count:,} ({suspicious_pct:.1f}%)
        • Time Period: {time_period}
        
        🚨 THREAT BREAKDOWN
        {threat_breakdown}
        
        🌍 GEOGRAPHIC THREAT DISTRIBUTION
        {geo_threat}
        
        ⏰ PEAK THREAT HOURS
        {peak_hours}
        
        🔍 HIGH-RISK INDICATORS
        • Average payload entropy for threats: {f"{avg_entropy:.2f}" if avg_entropy is not None else "N/A"}
        • Average failed login attempts for threats: {f"{avg_failed_logins:.2f}" if avg_failed_logins is not None else "N/A"}
        • Most targeted ports: {top_ports if top_ports is not None else "N/A"}
        
        📈 MODEL PERFORMANCE SUMMARY
        """

        if self.models:
            for name, results in self.models.items():
                report += f"\n• {name}: Accuracy {results['accuracy']:.3f}, AUC {results['auc_score']:.3f}"

        report += "\n\n" + "="*70

        print(report)
        return report
    def save_to_database(self, db_path='cybersecurity_threats.db'):
        """
        Save analysis results to SQLite database
        """
        if self.processed_data is None:
            print("No processed data to save")
            return
        
        print("Columns in self.data:", self.data.columns.tolist())  # Debug: check available columns
        
        conn = sqlite3.connect(db_path)
        
        # Save main dataset
        self.processed_data.to_sql('threat_data', conn, if_exists='replace', index=False)
        
        # Use existing column name for grouping - changed from 'threat_type' to 'detection_types'
        summary_stats = self.data.groupby('detection_types').agg({
            'is_suspicious': ['count', 'mean'],
            'payload_entropy': 'mean',
            'request_frequency': 'mean'
        }).round(3)
        
        summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns]
        summary_stats.reset_index().to_sql('threat_summary', conn, if_exists='replace', index=False)
        
        conn.close()
        print(f"Data saved to {db_path}")



def main():
    """
    Main execution function
    """
    print("🔒 CYBERSECURITY THREAT ANALYSIS SYSTEM 🔒\n")
    
    # Initialize analyzer
    analyzer = CyberThreatAnalyzer()
    
    # Option 1: Create sample dataset
   # print("Creating sample cybersecurity dataset...")
    #analyzer.create_sample_dataset(n_samples=5000)
    
    # Option 2: Load external dataset (uncomment to use)
    analyzer.load_external_dataset('CloudWatch_Traffic_Web_Attack.csv')
    
    # Perform comprehensive analysis
    print("\n" + "="*50)
    print("PERFORMING EXPLORATORY DATA ANALYSIS")
    print("="*50)
    analyzer.exploratory_data_analysis()
    
    print("\n" + "="*50)
    print("TRAINING MACHINE LEARNING MODELS")
    print("="*50)
    analyzer.train_ml_models()

    print("\n" + "="*50)
    print("CLUSTERING ANALYSIS")
    print("="*50)
    analyzer.threat_clustering_analysis()

    print("\n" + "="*50)
    print("GENERATING THREAT REPORT")
    print("="*50)
    analyzer.generate_threat_report()

    print("\n" + "="*50)
    print("SAVING TO DATABASE")
    print("="*50)
    analyzer.save_to_database()
    
    print("\n✅ Analysis Complete! Check the generated files and database.")

if __name__ == "__main__":
    main()

# Additional utility functions for VS Code integration

def load_and_analyze_custom_dataset(file_path):
    """
    Quick function to analyze your own dataset
    Usage: load_and_analyze_custom_dataset('path/to/your/dataset.csv')
    """
    analyzer = CyberThreatAnalyzer()
    analyzer.load_external_dataset(file_path)
    analyzer.exploratory_data_analysis()
    analyzer.train_ml_models()
    return analyzer

def export_results_to_excel(analyzer, filename='threat_analysis_results.xlsx'):
    """
    Export analysis results to Excel file
    """
    if analyzer.processed_data is None:
        print("No data to export")
        return
    
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # Main dataset
        analyzer.processed_data.to_excel(writer, sheet_name='Threat_Data', index=False)
        
        # Summary statistics
        summary = analyzer.data.groupby('detection_types').describe()
        summary.to_excel(writer, sheet_name='Summary_Stats')
        
        # Threat distribution
        threat_dist = analyzer.data['detection_types'].value_counts()
        threat_dist.to_excel(writer, sheet_name='Threat_Distribution')
    
    print(f"Results exported to {filename}")