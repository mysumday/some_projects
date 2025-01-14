import pandas as pd
import private.env
from src.interface.ai_interface import AIInterface
import src.transforms.trasform_funcs  as trasform_funcs
df = pd.read_csv("./some_data.csv")

test = AIInterface(load_modules=[trasform_funcs])

request = ("Rename FruitCol column to MyFavorite column then drop rows with missing values and "
           "also drop BoolCol2 column, filter rows so only rows were integerCol is more than 50 "
           "are shown, and save "
           "this "
           "into test.csv ")

test.execute_commands(df, request)