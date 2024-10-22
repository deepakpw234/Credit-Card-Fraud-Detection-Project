import os
import sys
import logging
from datetime import datetime

filename = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

os.makedirs(os.path.join(os.getcwd(),"logs"),exist_ok=True)

log_file_path = os.path.join(os.getcwd(),"logs",filename)

print((filename))


logging.basicConfig(
    filename=log_file_path,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

