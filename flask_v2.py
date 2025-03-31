from flask import Flask, request, render_template, jsonify
from rag_pipeline import run_rag_pipeline

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/query", methods=["POST"])
def handle_query():
    user_input = request.form.get("query")
    print("📨 Query received from frontend:", user_input)
    result = run_rag_pipeline(user_input)
    return jsonify({"query": user_input, "response": result})

if __name__ == "__main__":
    app.run(debug=True)
