import base64
import io
import pickle
import re
from io import BytesIO

import matplotlib.pyplot as plt
import pandas as pd
from flask import (Flask, jsonify, redirect, render_template, request,
                   send_file, url_for)
from flask_cors import CORS
from flask_login import (LoginManager, UserMixin, current_user, login_required,
                         login_user, logout_user)
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from textblob import TextBlob
from werkzeug.security import check_password_hash, generate_password_hash

STOPWORDS = set(stopwords.words("english"))

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///your_database.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SECRET_KEY"] = "Sai Kumar"

CORS(app)

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"

migrate = Migrate(app, db)


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)

    data = db.relationship('UserData', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class UserData(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    input_text = db.Column(db.Text, nullable=False)
    result = db.Column(db.Text, nullable=False)

    def __init__(self, user_id, input_text, result):
        self.user_id = user_id
        self.input_text = input_text
        self.result = result


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        data = request.get_json()
        username = data.get("username")
        email = data.get("email")
        password = data.get("password")

        if not username or not email or not password:
            return jsonify({"error": "Missing username, email, or password"}), 400

        if (
                User.query.filter_by(username=username).first()
                or User.query.filter_by(email=email).first()
        ):
            return jsonify({"error": "Username or email already exists"}), 400

        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        return jsonify({"message": "User created successfully"}), 201
    else:
        return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        data = request.get_json()
        username = data.get("username")
        password = data.get("password")

        if not username or not password:
            return jsonify({"error": "Missing username or password"}), 400

        user = User.query.filter_by(username=username).first()

        if not user or not user.check_password(password):
            return jsonify({"error": "Invalid username or password"}), 401

        login_user(user)
        return jsonify({"message": "Logged in successfully"}), 200
    else:
        return render_template("login.html")


@app.route("/test", methods=["GET"])
def test():
    return "Test request received successfully. Service is running."


@app.route("/", methods=["GET", "POST"])
@login_required
def home():
    return render_template("landing.html")


@app.route("/predict", methods=["POST"])
@login_required
def predict():
    # Select the predictor to be loaded from Models folder
    predictor = pickle.load(open(r"Models/model_xgb.pkl", "rb"))
    scaler = pickle.load(open(r"Models/scaler.pkl", "rb"))
    cv = pickle.load(open(r"Models/countVectorized.pkl", "rb"))
    try:
        # Check if the request contains a file (for bulk prediction) or text input
        if "file" in request.files:
            # Bulk prediction from CSV file
            file = request.files["file"]
            data = pd.read_csv(file)

            # Perform sentiment analysis for each text input in the CSV
            predictions = []
            for text_input in data["text_column_name"]:  # Replace "text_column_name" with the actual column name
                blob = TextBlob(text_input)
                sentiment_polarity = blob.sentiment.polarity
                predicted_sentiment = "POSITIVE" if sentiment_polarity > 0 else "NEGATIVE"
                predictions.append(predicted_sentiment)

            # Add predictions to the DataFrame
            data["predicted_sentiment"] = predictions

            # Convert DataFrame to CSV string
            predictions_csv = data.to_csv(index=False)

            # Create response with predictions CSV
            response = io.BytesIO()
            response.write(predictions_csv.encode())
            response.seek(0)

            # Add headers for graph data if needed
            response.headers["X-Graph-Exists"] = "false"  # Assuming no graph data for bulk prediction

            return send_file(
                response,
                mimetype="text/csv",
                as_attachment=True,
                download_name="Predictions.csv",
            )
        elif "text" in request.json:
            # Single string prediction
            # text_input = request.json["text"]
            # predicted_sentiment = single_prediction(predictor, scaler, cv, text_input)
            text_input = request.json["text"]
            # Perform sentiment analysis using TextBlob
            blob = TextBlob(text_input)
            # Get the sentiment polarity
            sentiment_polarity = blob.sentiment.polarity
            # Assign sentiment based on polarity
            predicted_sentiment = "POSITIVE" if sentiment_polarity > 0 else "NEGATIVE"

            user = current_user
            # Save the input text and prediction result to the UserData table
            new_data = UserData(user_id=str(user), input_text=text_input, result=predicted_sentiment)
            db.session.add(new_data)
            db.session.commit()
            db.session.refresh(new_data)
            print(predicted_sentiment)
            return jsonify({"prediction": predicted_sentiment})

    except Exception as e:
        return jsonify({"error": str(e)})


def single_prediction(predictor, scaler, cv, text_input):
    corpus = []
    stemmer = PorterStemmer()
    review = re.sub("[^a-zA-Z]", " ", text_input)
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if not word in STOPWORDS]
    review = " ".join(review)
    corpus.append(review)
    X_prediction = cv.transform(corpus).toarray()
    X_prediction_scl = scaler.transform(X_prediction)
    y_predictions = predictor.predict_proba(X_prediction_scl)
    y_predictions = y_predictions.argmax(axis=1)[0]
    print(y_predictions)

    return "Positive" if y_predictions == 1 else "Negative"


def bulk_prediction(predictor, scaler, cv, data):
    corpus = []
    stemmer = PorterStemmer()
    for i in range(0, data.shape[0]):
        review = re.sub("[^a-zA-Z]", " ", data.iloc[i]["Sentence"])
        review = review.lower().split()
        review = [stemmer.stem(word) for word in review if not word in STOPWORDS]
        review = " ".join(review)
        corpus.append(review)

    X_prediction = cv.transform(corpus).toarray()
    X_prediction_scl = scaler.transform(X_prediction)
    y_predictions = predictor.predict_proba(X_prediction_scl)
    y_predictions = y_predictions.argmax(axis=1)
    y_predictions = list(map(sentiment_mapping, y_predictions))

    data["Predicted sentiment"] = y_predictions
    predictions_csv = BytesIO()

    data.to_csv(predictions_csv, index=False)
    predictions_csv.seek(0)

    graph = get_distribution_graph(data)

    return predictions_csv, graph


def get_distribution_graph(data):
    fig = plt.figure(figsize=(5, 5))
    colors = ("green", "red")
    wp = {"linewidth": 1, "edgecolor": "black"}
    tags = data["Predicted sentiment"].value_counts()
    explode = (0.01, 0.01)

    tags.plot(
        kind="pie",
        autopct="%1.1f%%",
        shadow=True,
        colors=colors,
        startangle=90,
        wedgeprops=wp,
        explode=explode,
        title="Sentiment Distribution",
        xlabel="",
        ylabel="",
    )

    graph = BytesIO()
    plt.savefig(graph, format="png")
    plt.close()

    return graph


def sentiment_mapping(x):
    if x == 1:
        return "Positive"
    else:
        return "Negative"


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(port=5000, debug=True)
