import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
import textwrap
from langgraph.graph import END, StateGraph
from langchain_groq import ChatGroq
from openbb import obb
from langchain_core.messages import HumanMessage, SystemMessage
from duckduckgo_search import DDGS
from langgraph.prebuilt import tools_condition, ToolNode
import warnings
from PIL import Image

warnings.filterwarnings("ignore")

# Load environment variables from the .env file
load_dotenv()

# Get API keys and other sensitive data from environment variables
CHATGROQ_API_KEY = os.getenv("CHATGROQ_API_KEY")
OPENBB_PAT = os.getenv("OPENBB_PAT")

MODEL = "llama-3.1-8b-instant"
# MODEL = "llama-3.3-70b-versatile"

# Initialize ChatGroq and OpenBB with the API keys
llm = ChatGroq(
    temperature=0,
    model_name=MODEL,
    api_key=CHATGROQ_API_KEY,  # Use the API key from the environment
)
obb.obb.account.login(pat=OPENBB_PAT)  # Use the OpenBB PAT from the environment
obb.obb.user.preferences.output_type = "dataframe"

# Flask app
app = Flask(__name__)

# Your extension ID
extension_id = "afebmagfbkjhjpgjohcgpplmnjlmpaam"
# Enable CORS for only your extension
CORS(app, origins=[f"chrome-extension://{extension_id}"])

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json()
        user_query = data.get("user_query", "")
        print("Received user query:", user_query)

        if not user_query:
            return jsonify({"error": "user_query is required"}), 400

        # Run the LangGraph workflow
        state = graph_app.invoke({"user_query": user_query})
        print("LangGraph state:", state)

        # Prepare response data
        response_data = {"final_response": state["final_response"][-1].content}

        if ticker_check(state) == "yes":
            reports = {
                "price_analyst_report": state.get("price_analyst_report", ""),
                "news_analyst_report": state.get("news_analyst_report", ""),
                "final_report": (
                    {
                        "action": state["final_report"].action,
                        "score": state["final_report"].score,
                        "trend": state["final_report"].trend,
                        "sentiment": state["final_report"].sentiment,
                        "price_predictions": state["final_report"].price_predictions,
                        "summary": state["final_report"].summary,
                    }
                    if state.get("final_report")
                    else None
                ),
            }
            response_data.update(reports)

        return jsonify(response_data)

    except Exception as e:
        print("Error occurred:", str(e))
        traceback.print_exc()  # Make sure to import traceback at the top if you want to log exceptions
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)
