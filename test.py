from flask import Flask

# Create Flask app instance
app = Flask(__name__)

# Define route and view function
@app.route('/')
def hello():
    return 'Hello, World!'

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)