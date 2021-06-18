import random
from IPython.core.display import display, HTML
import matplotlib.pyplot as plt

data = """Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."""

def highlighter(word):
    colors = ["#FF3333", "#DD9999", "#9999DD", '#3333FF']
    i = random.randint(0, 18)
    if i < len(colors):
        color =  colors[i]
        word = '<span style="background-color:' +color+ '">' +word+ '</span>'
    return word

text = ' '.join([highlighter(word) for word in data.split()])

display(HTML(text))
plt.show()