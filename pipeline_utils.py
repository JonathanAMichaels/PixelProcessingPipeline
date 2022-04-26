import os
import fnmatch
import shutil

def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
        for name in dirs:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result

def create_config(script_folder, folder):
    shutil.copyfile(script_folder + '/config_template.yaml', folder + '/config.yaml')


