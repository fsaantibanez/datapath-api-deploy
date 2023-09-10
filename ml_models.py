from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, make_scorer
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import classification_report
import pandas as pd
import pickle

def evaluate_models(models, X, y, cv=5):
    """
    Evalúa diferentes modelos utilizando validación cruzada.

    Args:
    models (list of tuples): Lista de tuplas donde cada tupla contiene el nombre del modelo y el modelo.
    X (pd.DataFrame): Conjunto de características.
    y (pd.Series): Variable objetivo.
    cv (int): Número de divisiones en la validación cruzada (por defecto es 5).

    Returns:
    pd.DataFrame: Un DataFrame que muestra F1-Score y Recall para cada modelo.
    """
    results = []

    for model_name, model in models:
        # Calcula F1-Score y Recall utilizando validación cruzada
        f1_scorer = make_scorer(f1_score)
        recall_scorer = make_scorer(recall_score)

        f1_scores = cross_val_score(model, X, y, cv=cv, scoring=f1_scorer)
        recall_scores = cross_val_score(model, X, y, cv=cv, scoring=recall_scorer)

        # Calcula la media de los puntajes
        mean_f1 = f1_scores.mean()
        mean_recall = recall_scores.mean()

        # Agrega los resultados a la lista
        results.append({
            'Modelo': model_name,
            'F1-Score Promedio': mean_f1,
            'Recall Promedio': mean_recall
        })

    # Crea un DataFrame a partir de los resultados
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by = 'Recall Promedio', ascending = False)

    return results_df

def generate_classification_report(y_true, y_pred):
    """
    Genera un informe de clasificación y lo devuelve como un DataFrame.

    Args:
    y_true (array-like): Etiquetas verdaderas.
    y_pred (array-like): Etiquetas predichas.

    Returns:
    pd.DataFrame: Informe de clasificación en forma de DataFrame.
    """
    report = classification_report(y_true, y_pred, output_dict=True)
    df_classification_report = pd.DataFrame(report).transpose()
    return df_classification_report


def save_model_to_pickle(model, file_path):
    """
    Guarda un modelo entrenado en un archivo pickle.

    Args:
    model: El modelo entrenado que deseas guardar.
    file_path (str): La ruta del archivo pickle donde deseas guardar el modelo.

    Returns:
    None
    """
    with open(file_path, 'wb') as file:
        pickle.dump(model, file)

def load_model_from_pickle(file_path):
    """
    Carga un modelo entrenado desde un archivo pickle.

    Args:
    file_path (str): La ruta del archivo pickle desde donde deseas cargar el modelo.

    Returns:
    model: El modelo cargado.
    """
    with open(file_path, 'rb') as file:
        model = pickle.load(file)
    return model