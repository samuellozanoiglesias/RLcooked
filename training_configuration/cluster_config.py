"""
Cluster Configuration Module

Handles cluster-specific resource allocation and paths.
"""

def get_cluster_config(cluster_name):
    """
    Get cluster-specific configuration.
    
    Args:
        cluster_name: Name of the cluster ('brigit', 'cuenca', 'local')
        
    Returns:
        dict: Configuration dictionary with resource allocation and paths
    """
    cluster_name = cluster_name.lower()
    
    configs = {
        'brigit': {
            'local_path': '/mnt/lustre/home/samuloza',
            'num_gpus': 1.0,
            'num_cpus': 24,
            'num_env_workers': 8,
            'num_learner_workers': 1,
        },
        'cuenca': {
            'local_path': '',
            'num_gpus': 0.1,
            'num_cpus': 12,
            'num_env_workers': 8,
            'num_learner_workers': 1,
        },
        'local': {
            'local_path': 'D:/OneDrive - Universidad Complutense de Madrid (UCM)/Doctorado',
            'num_gpus': 0.0,
            'num_cpus': 1,
            'num_env_workers': 8,
            'num_learner_workers': 1,
        }
    }
    
    if cluster_name not in configs:
        raise ValueError(f"Invalid cluster specified. Choose from {list(configs.keys())}.")
    
    return configs[cluster_name]
