import tensorflow as tf
import tf2onnx

# Chemin vers votre modèle Keras
model_path = "../model/model.keras"
model = tf.keras.models.load_model(model_path)

# Définir output_names si inexistant : basé sur les noms des tenseurs de sortie
if not hasattr(model, 'output_names') or not model.output_names:
    model.output_names = [output.name.split(":")[0] for output in model.outputs]

# Définir la signature d'entrée pour le modèle.
spec = (tf.TensorSpec((None, 100), tf.float32, name="input"),)

# Chemin pour sauvegarder le modèle ONNX
onnx_model_path = "model/model.onnx"

# Conversion du modèle vers ONNX
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=onnx_model_path)

print("Conversion en ONNX réussie :", onnx_model_path)