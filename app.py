from flask import Flask
from flask_restful import reqparse
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

prediction_args = reqparse.RequestParser()
prediction_args.add_argument(
    'gender', type=int, help='request body \'gender\' is required', required=True)
prediction_args.add_argument(
    'age', type=int, help='request body \'age\' is required', required=True)
prediction_args.add_argument(
    'weight', type=float, help='request body \'weight\' is required', required=True)
prediction_args.add_argument(
    'height', type=float, help='request body \'height\' is required', required=True)
prediction_args.add_argument(
    'duration', type=float, help='request body \'duration\' is required', required=True)
prediction_args.add_argument(
    'heart_rate', type=float)
prediction_args.add_argument(
    'body_temp', type=float)


@app.route('/predictions', methods=['POST'])
def get_prediction():
    args = prediction_args.parse_args()
    model = None

    if not args['body_temp']:
        model = tf.keras.models.load_model('./model/model_6.h5')
        x = np.array([
            args['duration'],
            args['heart_rate'],
            args['age'],
            args['weight'],
            args['gender'],
            args['height'],
        ])
    else:
        model = tf.keras.models.load_model('./model/model_7.h5')
        x = np.array([
            args['duration'],
            args['heart_rate'],
            args['body_temp'],
            args['age'],
            args['weight'],
            args['gender'],
            args['height'],
        ])

    try:
        scaler = StandardScaler()
        x = scaler.fit_transform(x.reshape(-1,1))
        pred = model.predict(np.array([x,]))
        res = pred.flat[0]
    except ValueError:
        return {'error': 'value error exception'}, 400

    return {
        'result': res
    }


if __name__ == "__main__":
    app.run(debug=True)
