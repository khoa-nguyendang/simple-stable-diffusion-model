import yaml

app_configs = None
def load_config():
    if app_configs is not None:
        return app_configs
    with open('app_configs.yaml') as f:
        app_configs = yaml.safe_load(f)
    return app_configs