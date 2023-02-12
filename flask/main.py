# Binding.pry equiv
# import code; code.interact(local=dict(globals(), **locals()))

from flask import Flask, request, jsonify

from graph_recommend import get_recommendations

app = Flask(__name__)


@app.route("/")
def home():
    return jsonify({"body": "Invalid Arguments"}), 400


@app.route("/recommend")
def recommend():
    """
    Returns a list of recommended card_ids

    Parameters
    ---
    card_id: int
        The card_id of the user to get recommendations for
    min: int
        The minimum number of followers the user should have
    max: int
        The maximum number of followers the user should have

    Returns
    ---
    list: json object
        A list of recommended card_ids
    """
    try:
        card_id = request.args.get('card_id', type=int)
        min_fol = request.args.get('min',  type=int)
        max_fol = request.args.get('max', type=int)
        if card_id is None:
            return jsonify({"body": "Invalid Arguments"}), 400
        else:
            responses = get_recommendations(card_id, min_fol, max_fol)
            return jsonify({"body":responses}), 200
    except:
        return jsonify({"body": "Server Error"}), 500


if __name__ == "__main__":
    # repalce with your own host and port
    app.run(host="xxxxx.us-west-2.compute.amazonaws.com", port=8080, debug=True)
