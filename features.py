import numpy as np

def code_to_features(code):

    vec = np.zeros(10)

    # características simples
    vec[0] = code.count("for")
    vec[1] = code.count("while")
    vec[2] = code.count("if")
    vec[3] = code.count("def")
    vec[4] = code.count("return")

    # recursión simple
    lines = code.splitlines()

    for line in lines:
        if "def " in line:
            fname = line.split("def ")[1].split("(")[0]

            if fname in code.replace(line, ""):
                vec[5] = 1

    return vec.reshape(1, -1)