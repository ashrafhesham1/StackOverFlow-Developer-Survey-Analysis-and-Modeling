LOG_DATA_PKL    =  "data.pkl"
LOG_MODEL_PKL   =  "model.pkl"
LOG_METRICS_PKL =  "metrics.pkl"
MLFLOW_TRACKING_URI = '../models/mlruns'
MLFLOW_RUN_ID = "7a24a21a88fe4329a986ba8dd6c942cb"
CLUSTERS_YAML_PATH = "../data/processed/fe_cluster_skills_description.yaml"
#-------------------------------------------------------------

import os 
import sklearn
import pickle
import yaml
import re 

import pandas as pd

import mlflow
from mlflow.tracking import MlflowClient

#-------------------------------------------------------------


class JobPrediction:
    """Production Class for predicting the probability of a job from skills"""
    
    def __init__(self, mlflow_uri=MLFLOW_TRACKING_URI, run_id=MLFLOW_RUN_ID, clusters_yaml_path=CLUSTERS_YAML_PATH):

        # Constants
        self.tracking_uri  = mlflow_uri
        self.run_id        = run_id

        # Retrieve model and features
        self.model, self.features_names, self.targets_names = self.load_mlflow_objs()

        # Load clusters config 
        self.path_clusters_config = clusters_yaml_path
        self.skills_clusters_df = self.load_clusters_config(clusters_yaml_path)

    def load_mlflow_objs(self):
        """Load objects from the MLflow run"""

        # Initialize client and experiment
        mlflow.set_tracking_uri(self.tracking_uri)
        client = MlflowClient()

        run = mlflow.get_run(self.run_id)
        artificats_path = re.sub('file:\\\\','',os.path.normpath(run.info.artifact_uri))

        # Load data pkl
        data_path = os.path.join(artificats_path, LOG_DATA_PKL)
        with open(data_path, 'rb') as handle:
            data_pkl = pickle.load(handle)

        # Load model
        model_path = os.path.join(artificats_path, LOG_MODEL_PKL)
        with open(model_path, "rb") as f:
            model_pkl = pickle.load(f)

        # Return model and data labels
        return model_pkl["model_object"], data_pkl["features_names"], data_pkl["targets_names"]
    
    def load_clusters_config(self, path_clusters_config):
        """Load skills clusters developed in 03_feature_engineering.ipynb"""

        # Read YAML
        with open(path_clusters_config, "r") as stream:
            clusters_config = yaml.safe_load(stream)

        # Format into dataframe
        clusters_df = [(cluster_name, skill)
                       for cluster_name, cluster_skills in clusters_config.items()
                       for skill in cluster_skills]

        clusters_df = pd.DataFrame(clusters_df, columns=["cluster_name", "skill"])
        return clusters_df
        
    def get_all_skills(self):
        return self.features_names

    def get_all_jobs(self):
        return self.targets_names
    
    def create_features_array(self, available_skills):
        """Create the features array from a list of the available skills"""

        # Method's helper functions 
        def create_clusters_features(self, available_skills):
            sample_clusters = self.skills_clusters_df.copy()
            sample_clusters["available_skills"] = sample_clusters["skill"].isin(available_skills)
            cluster_features = sample_clusters.groupby("cluster_name")["available_skills"].sum()
            return cluster_features

        def create_skills_features(self, available_skills, exclude_features):
            all_features = pd.Series(self.features_names.copy())
            skills_names = all_features[~all_features.isin(exclude_features)]
            ohe_skills = pd.Series(skills_names.isin(available_skills).astype(int).tolist(), index=skills_names)
            return ohe_skills
        
        clusters_features = create_clusters_features(self, available_skills)
        skills_features   = create_skills_features(self, available_skills, 
                                                   exclude_features=clusters_features.index)
        
        features = pd.concat([skills_features, clusters_features])
        features = features[self.features_names]
        return features.values 
    
    def predict_jobs_probabilities(self, available_skills):
        '''Returns probabilities of the different jobs according to the skills'''

        features_array = self.create_features_array(available_skills)

        # Predict and format
        predictions = self.model.predict_proba([features_array])
        predictions = [prob[0][1] for prob in predictions] # Keep positive probs 
        predictions = pd.Series(predictions, index=self.targets_names)

        return predictions

    def recommend_new_skills(self, available_skills, target_job, threshold=0):
        # Calculate base probability
        base_predictions = self.predict_jobs_probabilities(available_skills)

        # Get all possible additional skills
        all_skills = pd.Series(self.get_all_skills())
        new_skills = all_skills[~all_skills.isin(available_skills)].copy()

        # Simulate new skills
        simulated_results = []
        for skill in new_skills:
            additional_skill_prob = self.predict_jobs_probabilities([skill] + available_skills)
            additional_skill_uplift = (additional_skill_prob - base_predictions) / base_predictions
            additional_skill_uplift.name = skill
            simulated_results.append(additional_skill_uplift)

        simulated_results = pd.DataFrame(simulated_results)

        # Recommend new skills
        target_results = simulated_results[target_job].sort_values(ascending=False)
        positive_mask = (target_results > threshold)
        return target_results[positive_mask]