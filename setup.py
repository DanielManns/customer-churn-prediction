import os
import src.utility.config as c

con = c.config()


def create_dirs():
    if not os.path.isdir(con.data_dir):
        os.makedirs(con.data_dir)
    if not os.path.isdir(con.exp_dir):
        os.makedirs(con.exp_dir)


create_dirs()