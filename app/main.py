# app/main.py
import sys
import uvicorn
from app.chatbot import Chatbot


def run_cli_chat():
    """Start the AI Twin in CLI interactive chat mode."""
    bot = Chatbot()
    print("==================================================")
    print("🤖 Smart AI Twin Assistant (CLI Mode)")
    print("Type your message below. Type 'quit' or 'exit' to exit.")
    print("==================================================")

    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.strip().lower() in {"quit", "exit"}:
                print("Goodbye! 👋 Stay mindful of your digital wellbeing.")
                break

            response = bot.get_response(user_input)
            print(f"\n{response['bot_name']}: {response['reply']}")
            print(f"[Detected Mood: {response['detected_emotion']} | Sentiment: {response['sentiment']}]")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting CLI chat. Goodbye! 👋")
            break


def run_api_server(host: str = "127.0.0.1", port: int = 8000):
    """Launch the FastAPI server with Uvicorn."""
    print(f"🚀 Starting Smart AI Twin REST API server on http://{host}:{port}...")
    uvicorn.run("app.api:app", host=host, port=port, reload=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        run_cli_chat()
    else:
        run_api_server()
