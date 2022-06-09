from flask import Flask
from flask_restful import reqparse
from keras.models import load_model
import numpy as np

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

    if not args['heart_rate'] or not args['body_temp']:
        # load model kedua
        # model = load_model('path-ke-model-tanpa-bodytemp-heartrate')
        pass

    model = load_model('./model/model1.h5')
    x = np.array([
        args['gender'],
        args['age'],
        args['weight'],
        args['height'],
        args['duration'],
        args['heart_rate'],
        args['body_temp'],
    ])
    # try:
    res = model.predict(x)
    # except ValueError:
    #     return {'error': 'value error exception'}, 400

    return {
        'result': res
    }


if __name__ == "__main__":
    app.run(debug=True)
