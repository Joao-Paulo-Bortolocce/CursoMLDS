import plotly.express as px  # pyright: ignore[reportMissingImports]
import plotly.graph_objects as go # pyright: ignore[reportMissingImports]
import seaborn as sns # pyright: ignore[reportMissingModuleSource]
import matplotlib.pyplot as plt # pyright: ignore[reportMissingModuleSource]
import pandas as pd # pyright: ignore[reportMissingModuleSource]
import numpy as np # pyright: ignore[reportMissingImports]
import os # pyright: ignore[reportMissingImports]


base_dir = os.path.dirname(__file__)  # Serve para localizar o arquivo .py, para assim poder criar o caminho relativo até os dados em qualquer local
caminho = os.path.join(base_dir, "..", "dados", "plano_saude.csv")

base_plano_saude = pd.read_csv(caminho)
print(base_plano_saude)