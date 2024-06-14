import os
import glob
import importlib.util

# Get the current directory
current_dir = os.path.dirname(__file__)

# Get all Python files in the current directory
modules = glob.glob(os.path.join(current_dir, "*.py"))

# Import all functions from each module
for module in modules:
    module_name = os.path.basename(module)[:-3]
    if module_name != "__init__":
        module_spec = importlib.util.spec_from_file_location(module_name, module)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        globals().update({name: getattr(module, name) for name in dir(module) if not name.startswith('_')})