import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# --- Fonctions d’activation ---
def relu(x): return np.maximum(0, x)
def sigmoid(x): return 1 / (1 + np.exp(-x))
def tanh(x): return np.tanh(x)
def softmax(x): exp_x = np.exp(x - np.max(x)); return exp_x / exp_x.sum()

activation_functions = {
    "ReLU": relu,
    "Sigmoïde": sigmoid,
    "Tanh": tanh,
    "Softmax": softmax
}

# --- Fonctions de coût ---
def mse(y_true, y_pred): return np.mean((y_true - y_pred)**2)
def cross_entropy(y_true, y_pred): return -np.sum(y_true * np.log(y_pred+1e-9)) / y_true.shape[0]

cost_functions = {
    "MSE": mse,
    "Cross-Entropy": cross_entropy
}

st.set_page_config(page_title="Réseau de neurones interactif", layout="wide")
st.title("🧠 Laboratoire interactif de réseaux de neurones")

# --- Configuration ---
st.sidebar.header("⚙️ Configuration du réseau")
n_input = st.sidebar.slider("Neurones en entrée", 1, 5, 3)
n_hidden_layers = st.sidebar.slider("Couches cachées", 1, 3, 1)
n_output = st.sidebar.slider("Neurones en sortie", 1, 5, 2)

mode = st.sidebar.radio("Mode de configuration :", ["Manuel", "Aléatoire"])
seed = st.sidebar.number_input("Graine aléatoire (si aléatoire)", value=42)
np.random.seed(seed)

st.sidebar.header("Realiser par Sony Tchouaou")

# Valeurs d’entrée
if mode == "Manuel":
    st.subheader("Valeurs des neurones d’entrée")
    X = np.array([st.number_input(f"Entrée {i+1}", value=0.0, key=f"x{i}") for i in range(n_input)])
else:
    X = np.random.randn(n_input)
    st.sidebar.write("Entrée générée :", X)

# Définition des couches cachées
hidden_layers = []
for l in range(n_hidden_layers):
    st.subheader(f"Couche cachée {l+1}")
    n_hidden = st.slider(f"Neurones couche cachée {l+1}", 1, 10, 4, key=f"hidden_{l}")
    if mode == "Manuel":
        W = np.zeros((n_input if l==0 else hidden_layers[l-1]['n'], n_hidden))
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                W[i,j] = st.number_input(f"W{l+1}[{i},{j}]", value=0.1, key=f"W{l}_{i}_{j}")
        b = np.array([st.number_input(f"Biais {l+1}-{j+1}", value=0.0, key=f"b{l}_{j}") for j in range(n_hidden)])
    else:
        W = np.random.randn(n_input if l==0 else hidden_layers[l-1]['n'], n_hidden)
        b = np.random.randn(n_hidden)
    act = st.selectbox(f"Activation couche cachée {l+1}", list(activation_functions.keys()), key=f"act{l}")
    hidden_layers.append({"n": n_hidden, "W": W, "b": b, "act": act})

# Couche de sortie
st.subheader("Couche de sortie")
if mode == "Manuel":
    W_out = np.zeros((hidden_layers[-1]['n'], n_output))
    for j in range(W_out.shape[0]):
        for k in range(W_out.shape[1]):
            W_out[j,k] = st.number_input(f"W_out[{j},{k}]", value=0.1, key=f"Wout_{j}_{k}")
    b_out = np.array([st.number_input(f"Biais sortie {k+1}", value=0.0, key=f"bout_{k}") for k in range(n_output)])
else:
    W_out = np.random.randn(hidden_layers[-1]['n'], n_output)
    b_out = np.random.randn(n_output)
act_out = st.selectbox("Activation sortie", list(activation_functions.keys()), key="act_out")

# --- VISUALISATION IMMÉDIATE ---
st.subheader("📊 Architecture du réseau")
G = nx.DiGraph()
input_nodes = [f"I{i+1}" for i in range(n_input)]
G.add_nodes_from(input_nodes)

hidden_nodes_layers = []
for l, layer in enumerate(hidden_layers):
    hidden_nodes = [f"H{l+1}_{j+1}" for j in range(layer["n"])]
    hidden_nodes_layers.append(hidden_nodes)
    G.add_nodes_from(hidden_nodes)
    prev_nodes = input_nodes if l == 0 else hidden_nodes_layers[l-1]
    for pn in prev_nodes:
        for hn in hidden_nodes:
            G.add_edge(pn, hn)

output_nodes = [f"O{k+1}" for k in range(n_output)]
G.add_nodes_from(output_nodes)
for hn in hidden_nodes_layers[-1]:
    for on in output_nodes:
        G.add_edge(hn, on)

pos = {}
for i, node in enumerate(input_nodes): pos[node] = (0, i)
for l, hidden_nodes in enumerate(hidden_nodes_layers):
    for j, node in enumerate(hidden_nodes): pos[node] = (2*(l+1), j)
for k, node in enumerate(output_nodes): pos[node] = (2*(len(hidden_nodes_layers)+1), k)

fig, ax = plt.subplots(figsize=(14,6))
nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=2000, font_size=10, arrows=True, ax=ax)
st.pyplot(fig)

# --- Exécution ---
if st.button("▶️ Exécuter le réseau"):
    A = X
    outputs = []
    for l, layer in enumerate(hidden_layers):
        Z = np.dot(A, layer["W"]) + layer["b"]
        A = activation_functions[layer["act"]](Z)
        outputs.append((f"H{l+1}", A))
        st.write(f"Couche cachée {l+1} → {layer['act']} :", A)

    Z_out = np.dot(A, W_out) + b_out
    Y = activation_functions[act_out](Z_out)
    outputs.append(("O", Y))

    st.success("Exécution terminée ✅")
    st.write("### Sortie finale :", Y)

    # Affichage des valeurs de sortie Y sur les neurones
    fig, ax = plt.subplots(figsize=(14,6))
    nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=2000, font_size=10, arrows=True, ax=ax)
    for k, node in enumerate(output_nodes):
        ax.text(pos[node][0]+0.3, pos[node][1], f"Y={Y[k]:.2f}", color="green", fontsize=10)
    st.pyplot(fig)

    st.markdown("### 📝 Commentaire")
    st.write("")

    st.subheader("📋 Tableau des sorties")
    st.table({f"Sortie O{k+1}": [Y[k]] for k in range(n_output)})

# --- Courbes d’activation ---


# --- Courbes d’activation ---
st.subheader("📈 Courbes des fonctions d’activation")
x_vals = np.linspace(-10, 10, 400)
fig, axes = plt.subplots(1, len(hidden_layers)+1, figsize=(15,5))

# Courbes des couches cachées
for l, layer in enumerate(hidden_layers):
    axes[l].plot(x_vals, activation_functions[layer["act"]](x_vals))
    axes[l].set_title(f"Activation {layer['act']}")
    axes[l].grid(True)

# Courbe de la couche de sortie
axes[-1].plot(x_vals, activation_functions[act_out](x_vals), color="red")
axes[-1].set_title(f"Activation sortie : {act_out}")
axes[-1].grid(True)

# Si exécution faite, afficher la position de Y sur la courbe
if 'Y' in locals():
    for k in range(len(Y)):
        axes[-1].scatter(Z_out[k], Y[k], color="green", s=80, label=f"Y{k+1}={Y[k]:.2f}")
    axes[-1].legend()

st.pyplot(fig)
