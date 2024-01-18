import pickle
import re
from pathlib import Path
import sklearn


__version__ = '1.1.1'

BASE_DIR = Path(__file__).resolve(strict=True).parent

with open(f'{BASE_DIR}/pipeline_model-{__version__}.pkl', 'rb') as f:
    model = pickle.load(f)

classes = ['Arabic', 'Danish', 'Dutch', 'English', 'French', 'German',
       'Greek', 'Hindi', 'Italian', 'Kannada', 'Malayalam', 'Portugeese',
       'Russian', 'Spanish', 'Sweedish', 'Tamil', 'Turkish']

def predict_pipeline(text):
    text = re.sub(r'[!@#$(),\n"%^*?\:;~`0-9]', " ", text)
    text = re.sub(r"[[]]", " ", text)
    text = text.lower()
    pred = model.predict([text])
    return classes[pred[0]]
