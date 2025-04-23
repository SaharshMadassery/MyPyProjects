from flask import Flask, request, jsonify, render_template
from logic import get_recommendation_response  

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')  # Changed to lowercase

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    query = data.get("recomnquery_val", "")
    recommendation = get_recommendation_response(query)
    return jsonify({"result": recommendation})  # Changed key to "result"

if __name__ == '__main__':
    app.run(debug=True)