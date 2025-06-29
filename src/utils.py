import os
import sys
import dill
from src.exceptional import log_exception
def save_object(file_path, object):
    try:
        directory_path = os.path.dirname(file_path)
        os.makedirs(directory_path,exist_ok=True)
        with open(file_path,'wb') as file:
            dill.dump(object,file)
        
    except Exception:
        exc_type,exc_value,exc_traceback = sys.exc_info()
        log_exception(exc_type,exc_value,exc_traceback)
    