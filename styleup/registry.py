import glob
import os
import time
import pickle
from colorama import Fore, Style
from tensorflow.keras import Model, models

# safe and load model here



############ Save the model ###############
model_save_path = os.path.join("..","saved-models")

def save_model(model: Model = None,
               params: dict = None,
               metrics: dict = None,
              model_save_path = model_save_path) -> None:
    """
    persist trained model, params and metrics
    """
    model_save_path = os.path.join("..","saved-models")

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    print(Fore.BLUE + "\nSave model to local disk..." + Style.RESET_ALL)

    # save params
    if params is not None:
        params_path = os.path.join(model_save_path, "params", timestamp + ".pickle")
        print(f"- params path: {params_path}")
        with open(params_path, "wb") as file:
            pickle.dump(params, file)

    # save metrics
    if metrics is not None:
        metrics_path = os.path.join(model_save_path, "metrics", timestamp + ".pickle")
        print(f"- metrics path: {metrics_path}")
        with open(metrics_path, "wb") as file:
            pickle.dump(metrics, file)

    # save model
    if model is not None:
        model_path = os.path.join(model_save_path, "models", timestamp)
        print(f"- model path: {model_path}")
        model.save(model_path)

    print("\n✅ data saved locally")

    return None




#######  Load the model  ######
def load_model_rn(save_copy_locally=False,model_save_path=model_save_path):
    """
    load the latest saved model, return None if no model found
    """
    print(Fore.BLUE + "\nLoad model from local disk..." + Style.RESET_ALL)
    model_save_path = os.path.join("..","saved-models")

    # get latest model version
    model_directory = os.path.join(model_save_path, "models")

    results = glob.glob(f"{model_directory}/*")
    if not results:
        return None

    model_path = sorted(results)[-1]
    print(f"- path: {model_path}")

    model = models.load_model(model_path)
    print("\n✅ model loaded from disk")

    return model
