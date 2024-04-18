import pandas as pd
from procesar_datos import *

df = pd.read_csv("datos.csv")
print(df["diagnostico_preuci"].unique())
