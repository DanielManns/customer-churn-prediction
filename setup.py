import os
import src.utility.config as c

con = c.config()


def create_dirs():
    if not os.path.isdir(con.data_dir):
        os.makedirs(con.data_dir)
    if not os.path.isdir(con.model_dir):
        os.makedirs(con.model_dir)
    if not os.path.isdir(con.exp_dir):
        os.makedirs(con.exp_dir)
    if not os.path.isdir(con.checkpoint_dir):
        os.makedirs(con.checkpoint_dir)

print(con)
create_dirs()