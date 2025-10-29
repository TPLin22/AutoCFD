# input information
import os
import sys
import yaml
from utils.util import log_with_time

# Function to load configuration from HIT.yaml
def load_config(file_path):
    global config
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Define paths
Src_PATH = os.path.dirname(os.path.abspath(__file__))
Base_PATH = os.path.dirname(Src_PATH)
Case_PATH = ''

config_file_path = os.getenv('CONFIG_FILE_PATH', '')

log_with_time(f"config_file_path: {config_file_path}")

config = load_config(config_file_path)


def get_config_value(*keys, default=""):
    for key in keys:
        if key in config and config[key] not in (None, ""):
            return config[key]
    return default


# Set variables from config
case_name = config_file_path.split('/')[-1].split('.')[0]
description = get_config_value('description')
mesh_path = get_config_value('mesh_path')
runfile_path = get_config_value('runfile_path')
input_file_list = get_config_value('input_file_list')
usr_requirment = description + "<mesh_path>" + mesh_path
max_loop = get_config_value('max_loop', default=10)
temperature = get_config_value('temperature', default=0.7)
run_times = get_config_value('run_times', default=1)
MetaGPT_PATH = get_config_value('MetaGPT_PATH')
model = get_config_value('model')
openfoam_llm_base_url = get_config_value('openfoam_llm_base_url')
Run_PATH = f'{Base_PATH}/run'  # Modify to the actual path
should_stop = False
status = ''

writter_prompt = ''
writter_system = ''
latest_inputfiles_rsp = ''

# Set environment variables from config (support lowercase/alternate keys)
api_key = get_config_value("API_KEY", "api_key", "METAGPT_API_KEY")
proxy = get_config_value("PROXY", "proxy")
base_url = get_config_value("BASE_URL", "base_url")
api_type = get_config_value("API_TYPE", "api_type", default="openai")

os.environ["API_KEY"] = api_key
os.environ["PROXY"] = proxy
os.environ["BASE_URL"] = base_url
os.environ["METAGPT_API_KEY"] = api_key
os.environ["OPENAI_API_KEY"] = api_key

if proxy:
    os.environ["http_proxy"] = proxy
    os.environ["https_proxy"] = proxy
else:
    os.environ.pop("http_proxy", None)
    os.environ.pop("https_proxy", None)

# Add MetaGPT_PATH to sys.path
sys.path.append(MetaGPT_PATH)
sys.path.append(Src_PATH)

log_with_time("Configuration loaded successfully:")
log_with_time(f"usr_requirment: {usr_requirment}")

# Extract MetaGPT_PATH
config2_yaml_path = os.path.join(MetaGPT_PATH, "config/config2.yaml")

# Check if config2.yaml exists
if not os.path.exists(config2_yaml_path):
    raise FileNotFoundError(f"{config2_yaml_path} does not exist")

with open(config2_yaml_path, 'r') as file:
    config2_data = yaml.safe_load(file)

new_config2_data = {
    "llm": {
        "api_type": api_type or "openai",
        "model": model,
        "proxy": proxy,
        "base_url": base_url,
        "api_key": api_key,
    }
}

# Write the modified config back to config2.yaml
with open(config2_yaml_path, 'w') as file:
    yaml.dump(new_config2_data, file, default_flow_style=False)

log_with_time(f"{config2_yaml_path} has been updated successfully.")